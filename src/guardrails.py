#!/usr/bin/env python3
"""Shared NeMo Guardrails orchestration for the RAG query flow."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import sys
from typing import Any

try:
    from dotenv import load_dotenv  # pyright: ignore[reportMissingImports]
except ImportError:
    def load_dotenv(*args: Any, **kwargs: Any) -> bool:
        return False

from nemoguardrails import LLMRails, RailsConfig  # pyright: ignore[reportMissingImports]
from nemoguardrails.rails.llm.options import RailStatus, RailsResult  # pyright: ignore[reportMissingImports]

# Ensure project root is on path so we can import sibling modules/packages.
_project_root = Path(__file__).resolve().parent.parent
sys.path = [path for path in sys.path if path != str(_project_root)]
sys.path.insert(0, str(_project_root))

from guardrails.actions import (
    check_library_input,
    check_library_output,
)
from src import query

DEFAULT_GUARDRAILS_DIR = "guardrails"
DEFAULT_BLOCKED_INPUT_MESSAGE = "Sorry, I can only help with safe questions about the indexed library website."
DEFAULT_NO_APPROVED_CONTEXT_MESSAGE = "Sorry, I couldn't find approved source material to answer that safely."
DEFAULT_BLOCKED_OUTPUT_MESSAGE = "Sorry, I can't provide that response."


@dataclass
class GuardrailedQueryResult:
    """Structured result for a guardrailed query pipeline run."""

    approved_question: str | None
    answer_text: str
    blocked: bool
    block_stage: str | None
    source_nodes: list[Any]
    sources: list[dict[str, Any]]
    rail_name: str | None = None


def _is_guardrails_enabled() -> bool:
    value = os.environ.get("GUARDRAILS_ENABLED", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def _guardrails_config_dir(config_dir: str | None = None) -> str:
    configured = config_dir or os.environ.get("GUARDRAILS_CONFIG_DIR", DEFAULT_GUARDRAILS_DIR)
    return str(Path(configured).resolve())


@lru_cache(maxsize=4)
def load_guardrails_app(config_dir: str | None = None) -> LLMRails:
    """Load and cache the shared NeMo Guardrails application."""
    resolved_dir = _guardrails_config_dir(config_dir)
    config = RailsConfig.from_path(resolved_dir)
    return LLMRails(config)


def _run_guardrail_action(coro: Any) -> Any:
    """Run an async custom guardrail action from the sync app entrypoints."""
    return asyncio.run(coro)


def apply_input_guardrails(question: str, *, config_dir: str | None = None) -> RailsResult:
    """Run the project input guardrail policy against the raw user question."""
    normalized_question = query.normalize_query_text(question)
    is_allowed = _run_guardrail_action(
        check_library_input({"last_user_message": normalized_question})
    )
    if not is_allowed:
        return RailsResult(
            status=RailStatus.BLOCKED,
            content=DEFAULT_BLOCKED_INPUT_MESSAGE,
            rail="CheckLibraryInputAction",
        )
    return RailsResult(status=RailStatus.PASSED, content=normalized_question)


def apply_output_guardrails(
    question: str,
    answer_text: str,
    *,
    config_dir: str | None = None,
) -> RailsResult:
    """Run the project output guardrail policy against a synthesized bot response."""
    normalized_question = query.normalize_query_text(question)
    is_allowed = _run_guardrail_action(
        check_library_output(
            {
                "last_user_message": normalized_question,
                "bot_message": answer_text,
            }
        )
    )
    if not is_allowed:
        return RailsResult(
            status=RailStatus.BLOCKED,
            content=DEFAULT_BLOCKED_OUTPUT_MESSAGE,
            rail="CheckLibraryOutputAction",
        )
    return RailsResult(status=RailStatus.PASSED, content=answer_text)


def run_guardrailed_query(
    query_text: str,
    *,
    query_pipeline: query.ChatPipeline | Any | None = None,
    database_url: str | None = None,
    similarity_top_k: int = query.SIMILARITY_TOP_K,
    config_dir: str | None = None,
) -> GuardrailedQueryResult:
    """Run the shared chat flow with input, retrieval, and output guardrails."""
    load_dotenv(override=True)
    normalized_query = query.normalize_query_text(query_text)

    if _is_guardrails_enabled():
        input_result = apply_input_guardrails(normalized_query, config_dir=config_dir)
        if input_result.status == RailStatus.BLOCKED:
            return GuardrailedQueryResult(
                approved_question=None,
                answer_text=DEFAULT_BLOCKED_INPUT_MESSAGE,
                blocked=True,
                block_stage="input",
                source_nodes=[],
                sources=[],
                rail_name=input_result.rail,
            )

        approved_question = input_result.content or normalized_query
    else:
        approved_question = normalized_query

    pipeline = query_pipeline or query.build_chat_engine(
        database_url=database_url,
        similarity_top_k=similarity_top_k,
    )

    response = query.run_chat_turn(
        approved_question,
        query_pipeline=pipeline,
        database_url=database_url,
        similarity_top_k=similarity_top_k,
        commit_history=False,
    )
    approved_nodes = list(getattr(response, "source_nodes", []) or [])

    if not approved_nodes:
        return GuardrailedQueryResult(
            approved_question=approved_question,
            answer_text=DEFAULT_NO_APPROVED_CONTEXT_MESSAGE,
            blocked=True,
            block_stage="retrieval",
            source_nodes=[],
            sources=[],
        )

    answer_text = str(response)

    if _is_guardrails_enabled():
        output_result = apply_output_guardrails(
            approved_question,
            answer_text,
            config_dir=config_dir,
        )
        if output_result.status == RailStatus.BLOCKED:
            return GuardrailedQueryResult(
                approved_question=approved_question,
                answer_text=DEFAULT_BLOCKED_OUTPUT_MESSAGE,
                blocked=True,
                block_stage="output",
                source_nodes=[],
                sources=[],
                rail_name=output_result.rail,
            )
        final_answer = output_result.content
    else:
        final_answer = answer_text

    if isinstance(pipeline, query.ChatPipeline):
        query.append_chat_turn(pipeline, approved_question, str(final_answer))

    return GuardrailedQueryResult(
        approved_question=approved_question,
        answer_text=final_answer,
        blocked=False,
        block_stage=None,
        source_nodes=approved_nodes,
        sources=query.extract_sources(response),
    )
