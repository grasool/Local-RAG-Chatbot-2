#create-persistent-storage.py


import chromadb
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import resolve_embed_model
from llama_index.llms import OpenAI

from llama_index.embeddings import HuggingFaceEmbedding

llm = OpenAI(temperature=0.7, api_base="http://localhost:1234/v1", api_key="not-needed")

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# load some documents
documents = SimpleDirectoryReader("./data-grants").load_data()

db = chromadb.PersistentClient(path="./chroma_db")

# get collection
chroma_collection = db.get_or_create_collection("quickstart")

service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)


vector_store = ChromaVectorStore(chroma_collection=chroma_collection, embed_model=embed_model)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# create your index
index = VectorStoreIndex.from_documents(
    documents, service_context=service_context, storage_context=storage_context
)


# create a query engine and query
query_engine = index.as_query_engine()
response = query_engine.query("Wht types of proejcts were proposed by the PI?")
print(response)