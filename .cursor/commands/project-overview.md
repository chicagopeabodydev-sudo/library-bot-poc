# project-overview

## Overview
This project is a chatbot for a local library website. It uses retrieval-augmented generation (RAG) to answer user questions based on content scraped from the library website.

## RAG Steps
The content available to the RAG process is generated in two steps:

1. `crawl4ai` connects to the local library website, crawls its pages, and generates markdown files for those pages.
   The home page URL is provided through an environment variable.
2. The markdown files created by `crawl4ai` are indexed by `LlamaIndex` using an embeddings model, and the resulting vectors are stored in a `pgvector` collection hosted on Supabase.

## Chatbot Steps
Once crawling, indexing, and embedding storage are complete, a chatbot UI built with Streamlit handles user questions. The chatbot applies guardrails at multiple points in the flow to make the application safer and more secure by blocking inappropriate, off-topic, or malicious user inputs or LLM responses. `NeMo Guardrails` is used to apply these guardrails.

The chatbot steps are:

1. The user enters a plain-text question into the Streamlit UI.
2. The question is checked against all configured input guardrails to verify that it is safe and on topic.
   On topic means related to the content scraped from the local library website.
3. If the question fails any input guardrails, an appropriate error message is returned through the Streamlit interface.
4. If the question passes the guardrails, `LlamaIndex` retrieves the appropriate content from the `pgvector` store and passes that content, along with the question, to an LLM to generate a response.
5. After the LLM returns a response, that response is checked against all configured output guardrails.
6. If the response fails any output guardrails, an error message is returned through the Streamlit interface.
7. If the response passes the output guardrails, the response is returned to the user through the Streamlit interface.

---

## Streamlit UI
The streamlit UI has two vertical columns. The column on the left displays an image that represents the chatbot. The column on the right displays the actual text of the user-inputted chats/questions as well as the RAG-supported responses.

---

## Parsing "Event" Models
Some of the content scraped from the local library website will describe events, such as a live book reading by an author or a game night for teenagers.

Event content should support at least these properties:

- `event_type`: string
- `event_title`: string
- `date_time`: date or datetime
- `target_age_group`: string with one of these values: `adult`, `teen`, or `kids`
- `location`: string describing the room or place in the library
- `description`: string
- `link_to_details`: string

### Use Pydantic models to capture events
When generating indexed content that will be used by RAG, event records should be identifiable and queryable through a predefined Pydantic schema.

For example, if the user asks, `What are kids events at the library?`, the relevant event data in the result should be mappable to that Pydantic event schema so it can be returned in a consistent structured format.
