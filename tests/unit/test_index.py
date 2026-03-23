"""Unit tests for markdown loading and event sidecar metadata merging."""

from __future__ import annotations

import json
from pathlib import Path
import sys

from llama_index.core.schema import TextNode

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import index


def test_load_markdown_documents_merges_event_sidecar_metadata(tmp_path: Path) -> None:
    """Markdown documents should absorb same-basename event sidecar metadata."""
    md_path = tmp_path / "events.md"
    md_path.write_text("# Events\nStorytime this Friday.", encoding="utf-8")
    md_path.with_suffix(".json").write_text(
        json.dumps(
            {
                "events": [
                    {
                        "event_type": "storytime",
                        "event_title": "Toddler Storytime",
                        "date_time": "2026-03-20T10:00:00",
                        "target_age_group": "kids",
                        "location": "Children's Room",
                        "description": "Stories and songs for toddlers.",
                        "link_to_details": "https://example.com/events/storytime",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    documents = index.load_markdown_documents(tmp_path)

    assert len(documents) == 1
    metadata = documents[0].metadata
    assert metadata["has_structured_events"] is True
    assert metadata["event_count"] == 1
    assert metadata["event_titles"] == "Toddler Storytime"
    assert metadata["event_target_age_groups"] == "kids"
    assert "structured_events_json" in metadata


def test_load_markdown_documents_keeps_non_event_markdown_usable(tmp_path: Path) -> None:
    """Markdown files without sidecars should still load with default metadata."""
    md_path = tmp_path / "hours.md"
    md_path.write_text("# Hours\nWeekdays 9 AM to 6 PM.", encoding="utf-8")

    documents = index.load_markdown_documents(tmp_path)

    assert len(documents) == 1
    metadata = documents[0].metadata
    assert metadata["file_name"] == "hours.md"
    assert metadata["has_structured_events"] is False
    assert metadata["event_count"] == 0


def test_chunk_nodes_for_embeddings_preserves_event_metadata() -> None:
    """Chunking should keep event metadata on the resulting text nodes."""
    source_node = TextNode(
        text="Storytime is every Friday at 10 AM in the children's room. " * 40,
        metadata={
            "file_name": "events.md",
            "has_structured_events": True,
            "event_target_age_groups": "kids",
        },
    )

    chunked_nodes = index.chunk_nodes_for_embeddings([source_node])

    assert chunked_nodes
    assert chunked_nodes[0].metadata["has_structured_events"] is True
    assert chunked_nodes[0].metadata["event_target_age_groups"] == "kids"
