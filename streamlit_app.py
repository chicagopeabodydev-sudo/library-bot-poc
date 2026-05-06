#!/usr/bin/env python3
"""Streamlit UI for querying the indexed library content."""

from __future__ import annotations

from pathlib import Path
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
ASSISTANT_IMAGE_PATH = (
    Path(__file__).resolve().parent / "assets" / "images" / "S_Calvin_Information_Desk.png"
)

st.set_page_config(page_title=APP_TITLE, layout="wide")


def apply_ui_styles() -> None:
    """Hide the default assistant avatar icon."""
    st.markdown(
        """
        <style>
        [data-testid="stChatMessageAvatarAssistant"] {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
    return []


def get_session_query_pipeline() -> Any:
    """Build one shared chat pipeline per browser session."""
    if "query_pipeline" not in st.session_state:
        database_url = get_required_env("DATABASE_URL")
        st.session_state.query_pipeline = build_chat_engine(database_url=database_url)
    return st.session_state.query_pipeline


def render_assistant_image() -> None:
    """Render the assistant image centered in the left column."""
    if not ASSISTANT_IMAGE_PATH.exists():
        st.info("Assistant image not found.")
        return

    st.image(str(ASSISTANT_IMAGE_PATH), use_container_width=True)


def build_chat_turns(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group chat history into question/answer turns."""
    turns: list[list[dict[str, Any]]] = []
    index = 0

    while index < len(messages):
        message = messages[index]
        turn = [message]

        if (
            message.get("role") == "user"
            and index + 1 < len(messages)
            and messages[index + 1].get("role") == "assistant"
        ):
            turn.append(messages[index + 1])
            index += 2
        else:
            index += 1

        turns.append(turn)

    return turns


def render_chat_panel(query_pipeline: Any) -> None:
    """Render the chat caption, history, and chat input."""
    st.caption(APP_CAPTION)

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
        try:
            result = run_guardrailed_query(prompt, query_pipeline=query_pipeline)
        except ValueError as exc:
            st.session_state.messages.append(
                {"role": "assistant", "content": str(exc), "sources": []}
            )
        except Exception:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Sorry, the app couldn't answer that question right now.",
                    "sources": [],
                }
            )
        else:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": result.answer_text,
                    "sources": result.sources,
                }
            )

    for turn in reversed(build_chat_turns(st.session_state.messages)):
        for message in turn:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                render_sources(message.get("sources", []))


def main() -> None:
    """Render the Streamlit RAG chat UI."""
    load_dotenv(override=True)
    apply_ui_styles()

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

    image_column, chat_column = st.columns([3, 4], gap="medium")
    with image_column:
        render_assistant_image()
    with chat_column:
        render_chat_panel(query_pipeline)


if __name__ == "__main__":
    main()
