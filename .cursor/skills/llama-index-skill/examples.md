# llama-index-skill Examples

## Example 1 - Creating nodes from markdown using the MarkdownNodeParser
from llama_index.core.node_parser import MarkdownNodeParser

```python
parser = MarkdownNodeParser()
nodes = parser.get_nodes_from_documents(markdown_docs)

```

---

## Example 2 - Load, index, store, and query steps using a "chromadb" open source vector store

```python
import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

# step 1 - load some documents
documents = SimpleDirectoryReader("./data").load_data()

# step 2 - initialize client, setting path to save data
db = chromadb.PersistentClient(path="./chroma_db")

# step 3 - create collection
chroma_collection = db.get_or_create_collection("quickstart")

# step 4 - assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# step 5 - create your index
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# step 6 - create a query engine and query
query_engine = index.as_query_engine()
response = query_engine.query("What is the meaning of life?")
print(response)

```

---

## Example 3 - Querying using post-processing

```python
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor

# step 1 - build the index
index = VectorStoreIndex.from_documents(documents)

# step 2 - configure a retriever
retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=10,
)

# step 3 - configure a response synthesizer
response_synthesizer = get_response_synthesizer()

# step 4 - assemble query engine
query_engine = RetrieverQueryEngine(
    retriever=retriever,
    response_synthesizer=response_synthesizer,
    node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
)

# step 5 - execute the query
response = query_engine.query("What did the author do growing up?")
print(response)

```

---

## Example 3 - Defining and using query_engines and query_tools

QueryEngineTool defines 2 things:
1. How to query and return results (which is a "query_engine")
2. "description" which indicates when the query engine is a good choice

If multiple tools are provided, the LLM can choose which "query_tool" to use based on the content of the question beng asked (foir example, summary questions can use a query_tool designed for this)

```python
from llama_index.core.tools import QueryEngineTool
from typing import List
from llama_index.core.vector_stores import FilterCondition

# define a query_engine and a query_tool that uses it
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
summary_tool = QueryEngineTool.from_defaults(
    name="summary_tool",
    query_engine=summary_query_engine,
    description=(
        "Useful if you want to get summarizations"
    ),
)

# define a second query_engine and a query_tool that uses it
def vector_query(
    query: str, 
    page_numbers: List[str]
) -> str:
    """Perform a vector search over an index.
    
    query (str): the string query to be embedded.
    page_numbers (List[str]): Filter by set of pages. Leave BLANK if we want to perform a vector search
        over all pages. Otherwise, filter by the set of specified pages.
    
    """

    metadata_dicts = [
        {"key": "page_label", "value": p} for p in page_numbers
    ]
    
    query_engine = vector_index.as_query_engine(
        similarity_top_k=2,
        filters=MetadataFilters.from_dicts(
            metadata_dicts,
            condition=FilterCondition.OR
        )
    )
    response = query_engine.query(query)
    return response
    

vector_query_tool = FunctionTool.from_defaults(
    name="vector_tool",
    fn=vector_query
)

# pass in both query_tool's to the LLM and let it pick which to use with "predict_and_call(...)"
response = llm.predict_and_call(
    [vector_query_tool, summary_tool], 
    "What is a summary of the paper?", 
    verbose=True
)

```

---

## Example 4 - Chat Engine to retain coversation history

```python
chat_engine = index.as_chat_engine()
streaming_response = chat_engine.stream_chat("Tell me a joke.")
for token in streaming_response.response_gen:
    print(token, end="")

```

---
