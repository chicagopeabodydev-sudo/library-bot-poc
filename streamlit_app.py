#!/usr/bin/env python3
"""Streamlit UI for querying the indexed library content."""

from __future__ import annotations

from typing import Any

import streamlit as st
try:
    from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
except ImportError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

from src.guardrails import run_guardrailed_query
from src.query import build_chat_engine, reset_chat_history
from src.rag import COLLECTION_NAME, get_required_env

APP_TITLE = "Library Information Desk"
APP_CAPTION = "What would you like to know about our wonderful library?"
SOURCE_EXPANDER_LABEL = "Retrieved sources"

st.set_page_config(page_title=APP_TITLE)


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Render retrieved source snippets below an answer."""
    if not sources:
        return

    with st.expander(SOURCE_EXPANDER_LABEL):
        for index, source in enumerate(sources, start=1):
            label = source.get("label", f"Source {index}")
            score = source.get("score")
            metadata = source.get("metadata", {})
            excerpt = source.get("excerpt", "")

            heading = f"{index}. {label}"
            if isinstance(score, (int, float)):
                heading = f"{heading} (score: {score:.3f})"

            st.markdown(f"**{heading}**")

            file_path = metadata.get("file_path") or metadata.get("source")
            url = metadata.get("url")
            if file_path:
                st.caption(str(file_path))
            elif url:
                st.caption(str(url))

            st.write(excerpt or "No excerpt available.")


def _initial_messages() -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": "Ask a question about the indexed library content.",
            "sources": [],
        }
    ]


def get_session_query_pipeline() -> Any:
    """Build one shared chat pipeline per browser session."""
    if "query_pipeline" not in st.session_state:
        database_url = get_required_env("DATABASE_URL")
        st.session_state.query_pipeline = build_chat_engine(database_url=database_url)
    return st.session_state.query_pipeline


def main() -> None:
    """Render the Streamlit RAG chat UI."""
    load_dotenv(override=True)

    st.title(APP_TITLE)
    st.caption(APP_CAPTION)

    with st.sidebar:
        st.write(f"Collection: `{COLLECTION_NAME}`")
        st.write("Configuration is read from `.env`.")
        st.write("The CLI now starts an interactive chat with optional `QUERY_TEXT` bootstrap.")
        if st.button("Reset conversation", use_container_width=True):
            if "query_pipeline" in st.session_state:
                reset_chat_history(st.session_state.query_pipeline)
            st.session_state.messages = _initial_messages()
            st.rerun()

    try:
        query_pipeline = get_session_query_pipeline()
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    except Exception:
        st.error("Unable to load the query pipeline. Check configuration and try again.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = _initial_messages()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            render_sources(message.get("sources", []))

    if prompt := st.chat_input("Ask about the indexed content"):
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching the library index..."):
                try:
                    result = run_guardrailed_query(prompt, query_pipeline=query_pipeline)
                except ValueError as exc:
                    st.error(str(exc))
                    return
                except Exception:
                    st.error("Sorry, the app couldn't answer that question right now.")
                    return

            answer_text = result.answer_text
            sources = result.sources

            st.write(answer_text)
            render_sources(sources)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer_text,
                "sources": sources,
            }
        )


if __name__ == "__main__":
    main()
