import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()

def get_vector_collection():
    #Shared connection logic for ChromaDB.
    api_key = os.getenv("OPENAI_API_KEY")
    
   
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small"
    )
    
    # Persistent storage 
    chroma_client = chromadb.PersistentClient(path="./equity_research_db")
    
    return chroma_client.get_or_create_collection(
        name="equity_reports",
        embedding_function=openai_ef
    )