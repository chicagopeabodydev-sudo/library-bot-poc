#!/usr/bin/env python3
"""
Interactive chat with the Supabase-backed LlamaIndex store.

Reads configuration from environment variables:
  DATABASE_URL    - PostgreSQL connection string for Supabase (required)
  OPENAI_API_KEY  - Used for retrieval and response generation (required)
  QUERY_TEXT      - Optional first-turn bootstrap for the interactive CLI
"""

from dataclasses import dataclass, field
import logging
import os
import sys
from typing import Any
from pathlib import Path

try:
    from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
except ImportError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

from llama_index.core import get_response_synthesizer  # pyright: ignore[reportMissingImports]
from llama_index.core.base.response.schema import Response  # pyright: ignore[reportMissingImports]
from llama_index.core.chat_engine import CondenseQuestionChatEngine  # pyright: ignore[reportMissingImports]
from llama_index.core.llms import ChatMessage, MessageRole  # pyright: ignore[reportMissingImports]
from llama_index.core.query_engine import CustomQueryEngine  # pyright: ignore[reportMissingImports]
from llama_index.core.response_synthesizers import ResponseMode  # pyright: ignore[reportMissingImports]
from llama_index.core.schema import QueryBundle  # pyright: ignore[reportMissingImports]

# Ensure project root is on path so we can import sibling script modules.
_project_root = Path(__file__).resolve().parent.parent
sys.path = [path for path in sys.path if path != str(_project_root)]
sys.path.insert(0, str(_project_root))

from src.rag import COLLECTION_NAME, configure_llamaindex, get_required_env, load_vector_index

SIMILARITY_TOP_K = 5
CHAT_MODE = "condense_question"
CLI_EXIT_COMMANDS = {"exit", "quit"}
CLI_RESET_COMMANDS = {"clear", "reset"}
EVENT_QUERY_TERMS = (
    "event",
    "events",
    "program",
    "programs",
    "calendar",
    "storytime",
    "book club",
    "author talk",
    "game night",
    "workshop",
    "class",
)
TARGET_AGE_GROUP_ALIASES = {
    "adult": "adult",
    "adults": "adult",
    "teen": "teen",
    "teens": "teen",
    "teenager": "teen",
    "teenagers": "teen",
    "young adult": "teen",
    "young adults": "teen",
    "kid": "kids",
    "kids": "kids",
    "child": "kids",
    "children": "kids",
}
LIBRARY_TOPIC_HINTS = (
    "library",
    "books",
    "book",
    "hours",
    "catalog",
    "events",
    "event",
    "program",
    "storytime",
    "meeting room",
    "borrow",
    "card",
    "wilmette",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ChatPipeline:
    """Stateful chat wrapper around a LlamaIndex query engine."""

    query_engine: Any
    chat_history: list[ChatMessage] = field(default_factory=list)


@dataclass(frozen=True)
class QueryIntent:
    """Normalized query hints that can shape retrieval post-processing."""

    is_event_query: bool
    target_age_group: str | None = None


class LibraryChatQueryEngine(CustomQueryEngine):
    """Custom query engine that preserves project-specific retrieval behavior."""

    retriever: Any
    response_synthesizer: Any

    def custom_query(self, query_str: str) -> Response:
        normalized_query = normalize_query_text(query_str)
        approved_nodes = retrieve_nodes(
            normalized_query,
            retriever=self.retriever,
        )
        if not approved_nodes:
            return Response(response=None, source_nodes=[])

        response = synthesize_response(
            normalized_query,
            approved_nodes,
            response_synthesizer=self.response_synthesizer,
        )
        response_metadata = getattr(response, "metadata", None)
        source_nodes = list(getattr(response, "source_nodes", []) or approved_nodes)
        return Response(
            response=str(response),
            source_nodes=source_nodes,
            metadata=response_metadata,
        )


def normalize_query_text(query_text: str) -> str:
    """Validate and normalize user query text."""
    normalized_query = query_text.strip()
    if not normalized_query:
        raise ValueError("Query text must not be empty")
    return normalized_query


def analyze_query_intent(query_text: str) -> QueryIntent:
    """Detect simple query intent hints for later metadata-aware retrieval."""
    normalized_query = normalize_query_text(query_text).lower()
    is_event_query = any(term in normalized_query for term in EVENT_QUERY_TERMS)

    target_age_group = None
    for alias, canonical in TARGET_AGE_GROUP_ALIASES.items():
        if alias in normalized_query:
            target_age_group = canonical
            break

    return QueryIntent(
        is_event_query=is_event_query,
        target_age_group=target_age_group,
    )


def _metadata_tokens(metadata: dict[str, Any], key: str) -> set[str]:
    """Split pipe-delimited metadata fields into normalized tokens."""
    raw_value = metadata.get(key)
    if raw_value in (None, ""):
        return set()

    if isinstance(raw_value, str):
        parts = raw_value.split("|")
    else:
        parts = [str(raw_value)]

    return {part.strip().lower() for part in parts if str(part).strip()}


def _node_text(source_node: Any) -> str:
    node = getattr(source_node, "node", source_node)
    try:
        return str(node.get_content()).strip()
    except (AttributeError, TypeError):
        return str(getattr(node, "text", "")).strip()


def _node_metadata(source_node: Any) -> dict[str, Any]:
    node = getattr(source_node, "node", source_node)
    return dict(getattr(node, "metadata", {}) or {})


def _node_has_library_signals(source_node: Any) -> bool:
    text = _node_text(source_node).lower()
    metadata = _node_metadata(source_node)
    metadata_blob = " ".join(str(value).lower() for value in metadata.values())

    if not text:
        return False

    for hint in LIBRARY_TOPIC_HINTS:
        if hint in text or hint in metadata_blob:
            return True

    return bool(text and (metadata.get("file_name") or metadata.get("source") or metadata.get("url")))


def filter_retrieved_nodes(nodes: list[Any]) -> list[Any]:
    """Remove empty or clearly off-topic retrieved nodes before synthesis."""
    return [node for node in nodes if _node_has_library_signals(node)]


def select_nodes_for_query(query_text: str, nodes: list[Any]) -> list[Any]:
    """Prefer structured event nodes when the user is asking about events."""
    intent = analyze_query_intent(query_text)
    if not intent.is_event_query or not nodes:
        return nodes

    structured_nodes = [
        node
        for node in nodes
        if bool(_node_metadata(node).get("has_structured_events"))
    ]
    if not structured_nodes:
        return nodes

    if intent.target_age_group:
        age_group_matches = [
            node
            for node in structured_nodes
            if intent.target_age_group in _metadata_tokens(
                _node_metadata(node),
                "event_target_age_groups",
            )
        ]
        if age_group_matches:
            return age_group_matches

    return structured_nodes


def build_retriever(*, database_url: str | None = None, similarity_top_k: int = SIMILARITY_TOP_K) -> Any:
    """Create a retriever backed by the shared Supabase vector store."""
    configure_llamaindex(enable_llm=False)
    index = load_vector_index(database_url=database_url)
    return index.as_retriever(similarity_top_k=similarity_top_k)


def build_response_synthesizer() -> Any:
    """Create the shared response synthesizer used after retrieval."""
    configure_llamaindex(enable_llm=True)
    return get_response_synthesizer(response_mode=ResponseMode.COMPACT)


def build_query_engine(*, database_url: str | None = None, similarity_top_k: int = SIMILARITY_TOP_K) -> Any:
    """Build the shared metadata-aware query engine used by the chat engine."""
    return build_chat_engine(
        database_url=database_url,
        similarity_top_k=similarity_top_k,
    )


def build_chat_engine(*, database_url: str | None = None, similarity_top_k: int = SIMILARITY_TOP_K) -> ChatPipeline:
    """Create the shared chat pipeline with retained conversation history."""
    configure_llamaindex(enable_llm=True)
    index = load_vector_index(database_url=database_url)
    retriever = index.as_retriever(similarity_top_k=similarity_top_k)
    response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT)
    return ChatPipeline(
        query_engine=LibraryChatQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
    )


def build_turn_chat_engine(query_pipeline: ChatPipeline) -> Any:
    """Build a fresh per-turn chat engine from the committed chat history."""
    return CondenseQuestionChatEngine.from_defaults(
        query_engine=query_pipeline.query_engine,
        chat_history=list(query_pipeline.chat_history),
        verbose=False,
    )


def append_chat_turn(query_pipeline: ChatPipeline, user_message: str, assistant_message: str) -> None:
    """Record a committed user/assistant exchange for future follow-up turns."""
    query_pipeline.chat_history.extend(
        [
            ChatMessage(role=MessageRole.USER, content=normalize_query_text(user_message)),
            ChatMessage(role=MessageRole.ASSISTANT, content=assistant_message.strip()),
        ]
    )


def reset_chat_history(query_pipeline: ChatPipeline) -> None:
    """Clear the retained chat history for a fresh conversation."""
    query_pipeline.chat_history.clear()


def retrieve_nodes(
    query_text: str,
    *,
    retriever: Any | None = None,
    database_url: str | None = None,
    similarity_top_k: int = SIMILARITY_TOP_K,
) -> list[Any]:
    """Retrieve and post-process approved source nodes for a normalized query."""
    normalized_query = normalize_query_text(query_text)
    active_retriever = retriever or build_retriever(
        database_url=database_url,
        similarity_top_k=similarity_top_k,
    )
    nodes = active_retriever.retrieve(QueryBundle(query_str=normalized_query))
    nodes = filter_retrieved_nodes(nodes)
    return select_nodes_for_query(normalized_query, nodes)


def synthesize_response(
    query_text: str,
    nodes: list[Any],
    *,
    response_synthesizer: Any | None = None,
) -> Any:
    """Synthesize a final answer from the approved retrieved nodes."""
    normalized_query = normalize_query_text(query_text)
    active_synthesizer = response_synthesizer or build_response_synthesizer()
    return active_synthesizer.synthesize(
        query=QueryBundle(query_str=normalized_query),
        nodes=nodes,
    )


def run_query(
    query_text: str,
    *,
    query_engine: Any | None = None,
    database_url: str | None = None,
    similarity_top_k: int = SIMILARITY_TOP_K,
) -> Any:
    """Run a single chat turn and optionally commit it to the retained history."""
    return run_chat_turn(
        query_text,
        query_pipeline=query_engine,
        database_url=database_url,
        similarity_top_k=similarity_top_k,
        commit_history=True,
    )


def run_chat_turn(
    query_text: str,
    *,
    query_pipeline: ChatPipeline | Any | None = None,
    database_url: str | None = None,
    similarity_top_k: int = SIMILARITY_TOP_K,
    commit_history: bool = True,
) -> Any:
    """Run a single chat turn against the retained conversation history."""
    normalized_query = normalize_query_text(query_text)
    pipeline = query_pipeline or build_chat_engine(
        database_url=database_url,
        similarity_top_k=similarity_top_k,
    )

    if hasattr(pipeline, "chat") and not isinstance(pipeline, ChatPipeline):
        return pipeline.chat(normalized_query)

    turn_chat_engine = build_turn_chat_engine(pipeline)
    response = turn_chat_engine.chat(normalized_query)
    if commit_history:
        append_chat_turn(pipeline, normalized_query, str(response))
    return response


def extract_sources(response: Any, *, max_excerpt_chars: int = 500) -> list[dict[str, Any]]:
    """Extract source metadata and excerpts from a LlamaIndex response."""
    sources: list[dict[str, Any]] = []

    for index, source_node in enumerate(getattr(response, "source_nodes", []), start=1):
        node = getattr(source_node, "node", source_node)
        metadata = dict(getattr(node, "metadata", {}) or {})

        try:
            content = node.get_content().strip()
        except (AttributeError, TypeError):
            content = str(getattr(node, "text", "")).strip()

        excerpt = content[:max_excerpt_chars].strip()
        if len(content) > max_excerpt_chars:
            excerpt = f"{excerpt}..."

        source_label = (
            metadata.get("file_name")
            or metadata.get("source")
            or metadata.get("url")
            or f"Source {index}"
        )

        sources.append(
            {
                "label": str(source_label),
                "score": getattr(source_node, "score", None),
                "excerpt": excerpt,
                "metadata": metadata,
            }
        )

    return sources


def run_guardrailed_query_for_cli(
    query_text: str,
    *,
    query_pipeline: ChatPipeline | Any | None = None,
    database_url: str | None = None,
    similarity_top_k: int = SIMILARITY_TOP_K,
) -> Any:
    """Run the shared guardrailed chat flow for CLI callers."""
    from src.guardrails import run_guardrailed_query

    return run_guardrailed_query(
        query_text,
        query_pipeline=query_pipeline,
        database_url=database_url,
        similarity_top_k=similarity_top_k,
    )


def _print_cli_result(result: Any) -> None:
    print(result.answer_text)
    sources = getattr(result, "sources", [])
    if not sources:
        return

    print("\nSources:")
    for source in sources:
        print(f"- {source.get('label', 'Unknown source')}")


def _run_cli_turn(prompt: str, *, query_pipeline: ChatPipeline, database_url: str) -> None:
    result = run_guardrailed_query_for_cli(
        prompt,
        query_pipeline=query_pipeline,
        database_url=database_url,
        similarity_top_k=SIMILARITY_TOP_K,
    )
    logger.info("Query complete. Retrieved %d approved source nodes", len(getattr(result, "source_nodes", [])))
    if getattr(result, "blocked", False):
        logger.info("Guardrails blocked the query at stage '%s'", getattr(result, "block_stage", "unknown"))
    _print_cli_result(result)


def main() -> None:
    """Run an interactive CLI chat against the vector store."""
    # Let CLI-provided env vars such as QUERY_TEXT win over .env defaults.
    load_dotenv(override=False)

    try:
        database_url = get_required_env("DATABASE_URL")
    except ValueError as exc:
        logger.error("%s", exc)
        raise SystemExit(1)

    logger.info("Connecting to Supabase collection '%s'", COLLECTION_NAME)
    logger.info("Starting chat with similarity_top_k=%d using %s mode", SIMILARITY_TOP_K, CHAT_MODE)

    try:
        query_pipeline = build_chat_engine(
            database_url=database_url,
            similarity_top_k=SIMILARITY_TOP_K,
        )
    except Exception as exc:
        logger.error("Unable to initialize chat pipeline: %s", exc)
        raise SystemExit(1)

    bootstrap_query = os.environ.get("QUERY_TEXT", "").strip()
    if bootstrap_query:
        try:
            _run_cli_turn(
                bootstrap_query,
                query_pipeline=query_pipeline,
                database_url=database_url,
            )
        except Exception as exc:
            logger.error("Query failed: %s", exc)
            raise SystemExit(1)

    print("Interactive library chat ready. Type a question, or use 'reset', 'quit', or 'exit'.")

    while True:
        try:
            prompt = input("You: ")
        except EOFError:
            break
        except KeyboardInterrupt:
            print()
            break

        normalized_prompt = prompt.strip()
        if not normalized_prompt:
            continue

        command = normalized_prompt.lower()
        if command in CLI_EXIT_COMMANDS:
            break
        if command in CLI_RESET_COMMANDS:
            reset_chat_history(query_pipeline)
            print("Chat history cleared.")
            continue

        try:
            _run_cli_turn(
                normalized_prompt,
                query_pipeline=query_pipeline,
                database_url=database_url,
            )
        except Exception as exc:
            logger.error("Query failed: %s", exc)
            print("Sorry, the chat couldn't answer that question right now.")


if __name__ == "__main__":
    main()
