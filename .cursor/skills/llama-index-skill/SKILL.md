---
name: llama-index-skill
description: Used to supplement the data LLMs can use when answering questions by supplying them with custom data generated and managed by llama-index.
---

# When to use this skill
Use this skill to take custom data in various formats, such as .pdf or .md (markdown), and convert it into a format that can be consumed by LLMs when answering questions. This process is known as retrieval augmented generation (RAG).

## Key steps when using LlamaIndex for RAG
- 1. LOADING data is ingesting supplied data, such as markdown files, and processing them into "documents" and "nodes".
- 2. INDEXING then processes the documents and nodes using an embedding model to generate an "index".
- 3. STORING the index (embedding results), typically in a vector database.
- 4. QUERYING an LLM and adding to this related data pulled from the index.


## Loading data is completed with a data connector, also called a "Reader".
### Result of loading is a "document" which is a container around any data source - for instance, a pdf file, an API output, or retrieve data from a database.
### Nodes are atomic units of data in LlamaIndex and represent “chunks” of a source document.
### Node parsers are a simple abstraction that take a list of documents and chunk them into node objects, such that each node is a specific chunk of the parent document.
### Example syntax using the markdown parser to get nodes:

```python
from llama_index.core.node_parser import MarkdownNodeParser

parser = MarkdownNodeParser()
nodes = parser.get_nodes_from_documents(markdown_docs)
```


## Indexing means creating a data structure that allows for semantic-querying of the data later.
### For LLMs this nearly always means creating vector embeddings using an embedding model.
### Indexes can be created from nodes or from documents (by using "from_documents()").
- Example to create an index from a document:

```python
index = VectorStoreIndex.from_documents(documents)
```

- Example to create an index from nodes:

```python
index = VectorStoreIndex(nodes)
```


## Storing an index typically uses a Vector Store.
### LlamaIndex supports many types of Vector Stores including pgvector on Supabase or chomadb.
- Example syntax (assumes the "vector_store" and "storage_context" already exist):

```python
# step 1 - load your index from stored vectors
index = VectorStoreIndex.from_vector_store(
    vector_store, storage_context=storage_context
)

# step 2 - create a query engine
query_engine = index.as_query_engine()
response = query_engine.query("What is llama2?")
```


## Querying is a prompt call to an LLM using a QueryEngine
### Optionally QueryEngines can be assigned an array of QueryTools, with each tool designed for a different type of question.
### similarity_top_k defines the number of top-scoring relevant document chunks (nodes) retrieved from an index to answer a query, and it determines how many semantic matches (using cosine similarity) are passed to the LLM.
    - Typically set to 2–10
    - Small similarity_top_k (1-3) is good for direct, specific, or fact-based queries.
    - Large similarity_top_k (5-10+) is better for complex, summary-based, or open-ended questions.
        - Higher values may improve context recall, but increase costs, latency, and increases odds of inaccurate results.
### 3 steps of querying:
    - 1. Retrieval - when you find and return the most relevant documents for your query from your Index. The most common type of retrieval is “top-k” semantic retrieval.
    - 2. Postprocessing (optional) - when the Nodes retrieved are reranked, transformed, or filtered, for example requiring that nodes have metadata matching certain keywords.
    - 3. Response Synthesis - when your query, your most-relevant data, and your prompt are combined and sent to your LLM to get a response.

    - Example creating a retriever:

```python
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)
```

## After a retriever fetches relevant nodes, a BaseSynthesizer synthesizes the final response by combining the information.
### To configure use the "response_mode" property of the retriever
### Options are:
| Mode | Description | When to Use |
|-----|-------------|-------------|
| **default** | Sequentially **creates and refines** an answer by iterating through each retrieved Node. A separate LLM call is made for every Node. | Best when accuracy and depth are more important than speed or cost. |
| **compact** | Attempts to **pack as many Node text chunks as possible into each prompt** sent to the LLM. | Useful when there are many small chunks and you want to reduce the number of LLM calls. |
| **tree_summarize** | Builds a **hierarchical summary tree** from the Nodes and returns the final summary from the root of the tree. | Ideal for summarizing large sets of documents or long collections of text chunks. |
| **no_text** | Runs the **retriever only**, without sending the Node content to the LLM for generation. | Useful for debugging, inspection, or retrieval evaluation. |
| **accumulate** | Applies the query to **each Node independently**, accumulating all responses into an array and returning them as a concatenated result. | Helpful when each chunk should be processed individually rather than merged into a single synthesized answer. |


## Query Engine compared to Chat Engine
Chat engine is a high-level interface for having a conversation with your data (multiple back-and-forth instead of a single question & answer). Chat engine is a stateful analogy of a Query Engine. By keeping track of the conversation history, it can answer questions with past context in mind.


## Additional Resources
- For usage examples, see [examples.md](examples.md)
- [LlamaIndex documentation](https://docs.llamaindex.ai/)
- Loading data documentation [Loading-Documents](https://developers.llamaindex.ai/python/framework/understanding/rag/loading/)
- Indexing documentation [Indexing-Documents](https://developers.llamaindex.ai/python/framework/understanding/rag/indexing/)
- Storing embedding results documentation [Storing-Indexed-Data](https://developers.llamaindex.ai/python/framework/understanding/rag/storing/)
- Querying LLMs [Querying](https://developers.llamaindex.ai/python/framework/understanding/rag/querying/)
- Chat Engine [Chat-engine](https://developers.llamaindex.ai/python/framework/module_guides/deploying/chat_engines/)