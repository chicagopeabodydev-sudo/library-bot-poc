"""Unit tests for crawl-time event extraction helpers and file output."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import crawl


class FakeMarkdownResult:
    def __init__(self, raw_markdown: str, fit_markdown: str | None = None) -> None:
        self.raw_markdown = raw_markdown
        self.fit_markdown = fit_markdown


class FakeCrawlResult:
    def __init__(
        self,
        *,
        url: str,
        markdown: Any = None,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
        extracted_content: str | None = None,
        error_message: str | None = None,
    ) -> None:
        self.url = url
        self.markdown = markdown
        self.success = success
        self.metadata = metadata or {}
        self.extracted_content = extracted_content
        self.error_message = error_message


class FakeAsyncWebCrawler:
    def __init__(self, responses: list[Any]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, Any]] = []

    async def __aenter__(self) -> "FakeAsyncWebCrawler":
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None

    async def arun(self, url: str, config: Any) -> Any:
        self.calls.append((url, config))
        if not self._responses:
            raise AssertionError("No fake crawler responses remaining")
        return self._responses.pop(0)


def test_is_event_like_page_matches_event_cues() -> None:
    """Event-like pages should be detected from URL/title/content hints."""
    result = FakeCrawlResult(
        url="https://example.com/events/storytime",
        metadata={"title": "Kids Storytime Events"},
    )

    assert crawl.is_event_like_page(result, "# Storytime\nJoin us for a kids program at the library.")


@pytest.mark.asyncio
async def test_crawl_main_writes_markdown_and_event_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Event-like pages should produce enriched markdown plus a JSON sidecar."""
    page_result = FakeCrawlResult(
        url="https://example.com/events/storytime",
        markdown=FakeMarkdownResult(
            raw_markdown="# Storytime\nJoin us for a kids event at the library."
        ),
        metadata={"title": "Kids Storytime Events"},
    )
    extraction_result = FakeCrawlResult(
        url=page_result.url,
        extracted_content=json.dumps(
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
    )
    fake_crawler = FakeAsyncWebCrawler(responses=[[page_result], extraction_result])

    monkeypatch.setattr(crawl, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(crawl, "AsyncWebCrawler", lambda: fake_crawler)
    monkeypatch.setattr(crawl, "run_index", lambda: None)
    monkeypatch.setenv("CRAWL_URL", "https://example.com")
    monkeypatch.setenv("CRAWL_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("CRAWL_RUN_INDEX", "0")
    monkeypatch.setenv("CRAWL_CLEAR_OUTPUT", "1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    await crawl.main()

    md_files = list(tmp_path.glob("*.md"))
    json_files = list(tmp_path.glob("*.json"))

    assert len(md_files) == 1
    assert len(json_files) == 1
    markdown_text = md_files[0].read_text(encoding="utf-8")
    sidecar = json.loads(json_files[0].read_text(encoding="utf-8"))

    assert "## Extracted Event Records" in markdown_text
    assert sidecar["events"][0]["event_title"] == "Toddler Storytime"
    assert len(fake_crawler.calls) == 2


@pytest.mark.asyncio
async def test_crawl_main_clears_stale_markdown_and_json_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Output cleanup should remove old markdown and JSON artifacts before crawling."""
    stale_md = tmp_path / "stale.md"
    stale_json = tmp_path / "stale.json"
    stale_md.write_text("old markdown", encoding="utf-8")
    stale_json.write_text("{}", encoding="utf-8")

    page_result = FakeCrawlResult(
        url="https://example.com/hours",
        markdown=FakeMarkdownResult(raw_markdown="# Hours\nThe library is open weekdays."),
        metadata={"title": "Library Hours"},
    )
    fake_crawler = FakeAsyncWebCrawler(responses=[[page_result]])

    monkeypatch.setattr(crawl, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(crawl, "AsyncWebCrawler", lambda: fake_crawler)
    monkeypatch.setattr(crawl, "run_index", lambda: None)
    monkeypatch.setenv("CRAWL_URL", "https://example.com")
    monkeypatch.setenv("CRAWL_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("CRAWL_RUN_INDEX", "0")
    monkeypatch.setenv("CRAWL_CLEAR_OUTPUT", "1")

    await crawl.main()

    assert not stale_md.exists()
    assert not stale_json.exists()
    assert len(list(tmp_path.glob("*.md"))) == 1
    assert len(list(tmp_path.glob("*.json"))) == 0
