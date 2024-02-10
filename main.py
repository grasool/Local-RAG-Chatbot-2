#main.py
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings import resolve_embed_model
#from llama_index.llms import Ollama

from llama_index.llms import OpenAI

documents = SimpleDirectoryReader("data").load_data()

# bge-m3 embedding model
embed_model = resolve_embed_model("local:BAAI/bge-small-en-v1.5")


llm = OpenAI(temperature=0.0, api_base="http://localhost:1234/v1", api_key="not-needed")

service_context = ServiceContext.from_defaults(
    embed_model=embed_model, llm=llm
)

index = VectorStoreIndex.from_documents(
    documents, service_context=service_context
)

query_engine = index.as_query_engine()
response = query_engine.query("What are PCR and FISH and how are these related to glioblastomas?")
print(response)

