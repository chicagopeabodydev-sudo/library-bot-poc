"""Integration test for the crawl script. Runs a real crawl using .env config."""

import os
import warnings
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.crawl import main


def _is_crawl_configured() -> bool:
    """Check if .env has CRAWL_URL configured (required for integration test)."""
    load_dotenv()
    return bool(os.environ.get("CRAWL_URL"))


@pytest.mark.asyncio
async def test_crawl_completes_and_writes_markdown_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run crawl with CRAWL_OUTPUT_DIR pointing to tmp_path; verify .md files are written."""
    if not _is_crawl_configured():
        warnings.warn(
            "CRAWL_URL not set in .env; skipping crawl integration test. "
            "Copy .env.example to .env and set CRAWL_URL to run this test.",
            UserWarning,
            stacklevel=2,
        )
        pytest.skip(
            "CRAWL_URL not configured in .env. Copy .env.example to .env and set CRAWL_URL to run this test."
        )

    monkeypatch.setenv("CRAWL_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setenv("CRAWL_RUN_INDEX", "0")  # Skip indexing; test only verifies crawl output

    await main()

    md_files = list(tmp_path.glob("*.md"))
    assert len(md_files) >= 1, f"Expected at least 1 .md file in {tmp_path}, got {len(md_files)}"

    contents = [f.read_text(encoding="utf-8") for f in md_files]
    assert any(len(c) > 100 for c in contents), "Expected at least one file with substantial content (>100 chars)"
