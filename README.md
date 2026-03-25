# library_bot_poc

This is a POC chatbot that uses retrieval augmented generation (RAG). The RAG content is from a local library website.

Tech stack:
1. **Crawl4ai**- used to crawl the library website from the home page (it automatically follows internal links) and generates markdown files from the crawled content
2. **LlamaIndex** - used to index the markdown files (i.e. chunking and embedding)
3. **pgVector Collection on Supabase** - stores the indexing results
4. **Streamlit** - simple chatbot UI
5. **NeMo Guardrails** - guards against malicious, inappropriate, or off-topic questions and also verifies LLM responses are appropriate
6. **Pydantic Models** - to standardize data about "events" at the library (used by crawl4ai)


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
### NOTE: plumbing for NeMo guardrails is in place but configs are NOT prod-level
The application currently uses NeMo Guardrails in both the CLI chat flow and the Streamlit UI. The guardrail configuration lives in `guardrails/` by default.

Install dependencies from `requirements.txt`, which now includes `nemoguardrails[openai]` so NeMo Guardrails uses the existing OpenAI model setup for this project.

### Configuration

Environment variables:
- `OPENAI_API_KEY` - Required by the configured NeMo Guardrails OpenAI-backed model
- `GUARDRAILS_ENABLED` - Enables or disables guardrail integration once the runtime wiring is added. Suggested values are `1` or `0`.
- `GUARDRAILS_CONFIG_DIR` - Directory that will contain the NeMo Guardrails config files. Default example value: `guardrails`

### Runtime Behavior

The current guardrailed runtime flow is:
1. Accept the user question in Streamlit or from `QUERY_TEXT`.
2. Apply input guardrails to the question.
3. If the question is off-topic, unsafe, or not understandable, return a clarification-style message instead of sending it into the history-aware chat engine.
4. Retrieve candidate context from the existing Supabase-backed index.
5. Generate an answer with the existing OpenAI-backed query model.
6. Apply output guardrails to the final answer before returning it to the user.

Behavior notes:
- only successful user/assistant turns are retained in chat history
- blocked or unclear questions are not committed to history
- short contextual follow-up questions are allowed when they appear related to the existing library conversation
- if a question does not clearly reference library topics, the app performs a lightweight retrieval preview before allowing the history-aware chat flow to answer it
- blocked input returns a clarification message asking the user to rephrase or stay on-topic

## Tests

Run tests from the project root:

```bash
.venv/bin/pytest tests/ -v
# or run only integration tests:
.venv/bin/pytest tests/integration/ -v
```

Useful focused test runs:
- `.venv/bin/pytest tests/unit/test_query.py tests/unit/test_guardrails.py -v`
- `.venv/bin/pytest tests/integration/test_integration_crawl.py tests/unit/test_index.py -v`

The crawl integration test (`tests/integration/test_integration_crawl.py`) performs real HTTP requests and requires `.env` with `CRAWL_URL` configured. If `.env` is missing or `CRAWL_URL` is not set, the test is skipped with a warning.