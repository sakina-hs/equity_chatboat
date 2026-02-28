import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
load_dotenv()


def _get_client():
    return chromadb.PersistentClient(path="./equity_research_db")

def _get_embedding():
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )

def get_vector_collection():
    return _get_client().get_or_create_collection(
        name="equity_reports",
        embedding_function=_get_embedding()
    )

def get_memory_collection():
    return _get_client().get_or_create_collection(
        name="user_memory",
        embedding_function=_get_embedding()
    )

