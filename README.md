# library_bot_poc

A RAG system with a chatbot. Content is crawled from a website and indexed for retrieval.

The runtime now applies NeMo Guardrails to the question-answering flow and can also extract structured library event data during crawling.

## Streamlit UI

Ask questions in a local browser UI backed by the existing Supabase/LlamaIndex index:

```bash
# After indexing
streamlit run streamlit_app.py
```

The Streamlit app:
- reads `DATABASE_URL` and `OPENAI_API_KEY` from `.env`
- accepts each question in the browser instead of from `QUERY_TEXT`
- retains conversation history for follow-up questions within the browser session
- queries the existing `website_docs` vector collection through a LlamaIndex chat engine
- prefers structured event results when the question is about library events
- shows the answer and the retrieved source snippets
- lets you reset the conversation from the sidebar

Expected flow:
1. `python src/crawl.py`
2. `python src/index.py`
3. `streamlit run streamlit_app.py`

## Crawling

Crawl a website and save markdown to `website-markdown/`:

```bash
# Create venv and install dependencies
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Install browser (required for crawl4ai)
playwright install chromium

# Configure and run
cp .env.example .env
# Edit .env: set CRAWL_URL, CRAWL_MAX_DEPTH, CRAWL_MAX_PAGES
python src/crawl.py
```

Environment variables (see `.env.example`):
- `CRAWL_URL` - Home page URL to crawl (required)
- `CRAWL_MAX_DEPTH` - Max depth levels (default: 2)
- `CRAWL_MAX_PAGES` - Max pages to crawl (default: 30)
- `CRAWL_OUTPUT_DIR` - Directory for crawled markdown and event sidecars (default: `website-markdown`)
- `CRAWL_CLEAR_OUTPUT` - Clear existing markdown and event sidecars before crawling (default: `1`)
- `CRAWL_RUN_INDEX` - Run indexing automatically after crawl (default: `1`)
- `OPENAI_API_KEY` - Also used for event extraction on event-like pages during crawl

### Event Extraction During Crawl

When a crawled page looks like it contains library event information, the crawl step performs an additional structured extraction pass using Crawl4AI plus the shared Pydantic event schema in `src/models/event.py`.

For those event-like pages, the crawl output includes:
- the normal `.md` file, enriched with a structured `Extracted Event Records` section for RAG
- a same-basename `.json` sidecar file containing the extracted event records in machine-readable form

For non-event pages, the crawl step still writes only markdown.

## Indexing

Load markdown from `website-markdown/`, create embeddings, and store in Supabase pgvector:

```bash
# After crawling (website-markdown/ must contain .md files)
# Configure .env with DATABASE_URL and OPENAI_API_KEY
python src/index.py
```

Environment variables:
- `DATABASE_URL` - PostgreSQL connection string for Supabase (required)
- `OPENAI_API_KEY` - Used for embeddings (required)

Get the connection string from Supabase: Project Settings → Database → Connection string (URI).

During indexing, each markdown file can be paired with a same-basename event sidecar. When present, structured event fields are merged into document and chunk metadata so event queries can later prefer pages with extracted events.

## Querying

Chat with the existing Supabase-backed index from the command line:

```bash
# After indexing
# Configure .env with DATABASE_URL and OPENAI_API_KEY
python src/query.py
```

Optional first-turn bootstrap:

```bash
QUERY_TEXT="when is the library open on sundays" python src/query.py
```

Environment variables:
- `DATABASE_URL` - PostgreSQL connection string for Supabase (required)
- `OPENAI_API_KEY` - Used for retrieval and response generation (required)
- `QUERY_TEXT` - Optional first-turn bootstrap for the interactive CLI chat

The CLI script now starts an interactive chat loop. The Streamlit app and the CLI both retain conversation history for follow-up questions, but they do not yet implement token streaming.

Event-aware querying behavior:
- questions about events prefer retrieved nodes with structured event metadata
- age-group hints such as `kids`, `teen`, or `adult` are used to narrow event results when matching metadata exists
- if no structured event matches are available, the query flow falls back to normal retrieval results

## Guardrails Configuration

The repository is prepared for a NeMo Guardrails configuration that will live in `guardrails/` by default.

Install dependencies from `requirements.txt`, which now includes `nemoguardrails[openai]` so NeMo Guardrails uses the existing OpenAI model setup for this project.

Environment variables:
- `GUARDRAILS_ENABLED` - Enables or disables guardrail integration once the runtime wiring is added. Suggested values are `1` or `0`.
- `GUARDRAILS_CONFIG_DIR` - Directory that will contain the NeMo Guardrails config files. Default example value: `guardrails`

Planned guardrailed runtime flow:
1. Accept the user question in Streamlit or from `QUERY_TEXT`.
2. Apply input guardrails to the question.
3. Retrieve candidate context from the existing Supabase-backed index.
4. Apply retrieval guardrails to the retrieved RAG content.
5. Generate an answer with the existing OpenAI-backed query model.
6. Apply output guardrails to the final answer before returning it to the user.

The current runtime applies the guardrailed flow in both the CLI and Streamlit UI while retaining only successful user/assistant turns in chat history.

## Tests

Run tests from the project root:

```bash
"/Users/peabody/Documents/repos/library_bot_poc/library_bot_poc/.venv/bin/pytest" tests/ -v
# or run only integration tests:
"/Users/peabody/Documents/repos/library_bot_poc/library_bot_poc/.venv/bin/pytest" tests/integration/ -v
```

Useful focused test runs:
- `"/Users/peabody/Documents/repos/library_bot_poc/library_bot_poc/.venv/bin/pytest" tests/unit/test_query.py tests/unit/test_guardrails.py -v`
- `"/Users/peabody/Documents/repos/library_bot_poc/library_bot_poc/.venv/bin/pytest" tests/integration/test_integration_crawl.py tests/unit/test_index.py -v`

The crawl integration test (`tests/integration/test_integration_crawl.py`) performs real HTTP requests and requires `.env` with `CRAWL_URL` configured. If `.env` is missing or `CRAWL_URL` is not set, the test is skipped with a warning.