# AGENT.md

This file provides AI coding assistants with context about this project's conventions, architecture, and rules.

The source code lives in `library_bot_poc/` relative to this file. The inner `README.md` has full setup and usage documentation.

---

## Persona

This project is a **general-purpose RAG chatbot POC**. It crawls any website (configured via `CRAWL_URL`) and answers user questions against that content using retrieval-augmented generation. The Wilmette Public Library website is the current example data source — it is not a constraint on the system's purpose or design.

Key implications:
- Do not assume content is always library-related. The guardrail topic-hint keyword lists and the NeMo Guardrails config are tuned to the current crawl target and should be updated when `CRAWL_URL` changes.
- This is a POC. Prefer simple, targeted implementations. Do not add abstractions or features beyond what the task requires.
- Avoid domain-specific assumptions in new code unless they are explicitly scoped to the current example site.

**Stack**: Crawl4AI · LlamaIndex · Supabase pgvector · Streamlit · NeMo Guardrails · Pydantic · OpenAI
- LLM: `gpt-4o-mini`
- Embeddings: `text-embedding-3-small` (1536 dimensions)
- Vector collection: `website_docs` (defined in `src/rag.py`)

---

## Project Layout

```
library_bot_poc/          # inner project root — run all commands from here
  src/
    crawl.py              # Crawl4AI web crawler + event extraction
    index.py              # LlamaIndex ingestion + Supabase pgvector storage
    query.py              # Chat pipeline, retrieval, response synthesis
    guardrails.py         # NeMo Guardrails orchestration (input/output rails)
    rag.py                # Shared LlamaIndex/Supabase configuration
    models/
      event.py            # LibraryEvent, LibraryEventExtractionResult Pydantic models
  guardrails/
    config.yml            # NeMo Guardrails Colang 2.x configuration
    actions.py            # Custom guardrail action functions
  tests/
    unit/                 # Unit tests (no external calls)
    integration/          # Integration tests (real HTTP, skipped if .env not set)
  streamlit_app.py        # Streamlit chat UI
  website-markdown/       # Output directory for crawled markdown + event JSON sidecars
  .env.example            # All supported environment variables with descriptions
  requirements.txt
  pytest.ini
```

---

## Environment Variables

See `.env.example` for the full list. Key variables:

| Variable | Required | Description |
|---|---|---|
| `CRAWL_URL` | Crawl | Home page URL to crawl |
| `CRAWL_MAX_DEPTH` | No | Max crawl depth (default: 2) |
| `CRAWL_MAX_PAGES` | No | Max pages to crawl (default: 30) |
| `OPENAI_API_KEY` | Yes | Used for embeddings, LLM, and event extraction |
| `DATABASE_URL` | Index/Query | Supabase PostgreSQL connection string |
| `GUARDRAILS_ENABLED` | No | Set to `0` to disable guardrails (default: enabled) |
| `GUARDRAILS_CONFIG_DIR` | No | Path to guardrails config dir (default: `guardrails`) |

---

## Logging Preferences

All modules use Python's standard `logging` — never `print()` for diagnostic output.

**Setup pattern** (used consistently across all `src/` modules):
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
```

**Level conventions**:
- `INFO` — pipeline milestones: crawl start/complete, files written, index progress, guardrail pass/block decisions
- `WARNING` — recoverable issues: skipped pages, invalid event sidecars, empty markdown, extraction failures
- `ERROR` — unrecoverable failures immediately before `raise SystemExit(1)`: missing required env vars, broken connections

**In tests**: use the `caplog` pytest fixture at `INFO` level to assert log output. Do not assert on `print()` for anything that should be logged.

---

## Testing Rules

**Run tests** from `library_bot_poc/` (the inner project root):
```bash
.venv/bin/pytest tests/ -v
.venv/bin/pytest tests/unit/ -v          # unit only
.venv/bin/pytest tests/integration/ -v  # integration only
```

### Rules

**No conftest.py** — all fixtures and helpers are defined inline within each test file. Do not create a shared `conftest.py`.

**Mocking: use `monkeypatch` only** — never `unittest.mock.patch`, `Mock`, or `MagicMock`. Use `monkeypatch.setattr()` to swap dependencies.

**Fake classes over magic mocks** — define minimal inline fake objects for LlamaIndex nodes, crawl results, chat engines, etc. Name them `FakeThing` and keep them in the same file as the tests that use them.

**Pytest fixtures in use**: `tmp_path`, `monkeypatch`, `capsys`, `caplog` — rely on these built-ins rather than third-party fixture libraries.

**Async tests**: mark with `@pytest.mark.asyncio`. The `pytest.ini` sets `asyncio_mode = auto`, so the decorator is still required but the event loop is managed automatically.

**No real external calls in unit tests**: always patch `load_dotenv`, `AsyncWebCrawler`, Supabase/DB connections, and OpenAI calls. Unit tests must run offline.

**Integration tests skip gracefully**: use `pytest.skip()` with a message when required env vars (e.g., `CRAWL_URL`) are not configured. Check via a helper like `_is_crawl_configured()` at the top of the test.

**File naming**:
- `tests/unit/test_<module>.py`
- `tests/integration/test_integration_<area>.py`

**Guardrails cache**: call `load_guardrails_app.cache_clear()` in teardown when a test exercises the `@lru_cache`-decorated `load_guardrails_app()` function in `src/guardrails.py`.

---

## Key Architectural Rules

**Guardrailed query flow is strictly ordered**: `input guardrails → retrieval → output guardrails`. Never skip or reorder stages.

**Blocked turns must not enter chat history**: only successful user/assistant exchanges are committed to `ChatPipeline.chat_history` via `append_chat_turn()`. A blocked or empty-retrieval result must return early without calling `append_chat_turn()`.

**Event model invariant**: always use `LibraryEvent` and `LibraryEventExtractionResult` from `src/models/event.py` for structured event data. Do not define ad-hoc dicts for events anywhere in the pipeline.

**Guardrail keyword lists are site-specific**: `LIBRARY_TOPIC_HINTS` in `src/query.py` and `guardrails/actions.py`, and `EVENT_HINT_TERMS` in `src/crawl.py`, are tuned to the current crawl target. Update them when `CRAWL_URL` changes to a different domain.
