from fastapi import FastAPI
from pydantic import BaseModel
from collections import defaultdict
from pathlib import Path
from vector_store import get_vector_collection, get_memory_collection
from llm_prompts import  generate_answer, get_longterm_memory
from ingest_pipeline import ingest_file_multimodal
from retrieval_utils import extract_query_entities, rerank_with_cross_encoder


app = FastAPI()
DATA_DIR = Path("./test")
SESSION_STORE = defaultdict(list)


class ChatRequest(BaseModel):
    message: str
    session_id: str

def build_chroma_where(filters: dict):
    if not filters:
        return None   
    conditions = []   
    if "company_name" in filters:
       
        conditions.append({"company_name": str(filters["company_name"]).strip()})
    if "fiscal_year" in filters:
        
        conditions.append({"fiscal_year": str(filters["fiscal_year"]).strip()})
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
from pathlib import Path


DATA_DIR = Path("./test") 

@app.on_event("startup")
def startup():
    if not DATA_DIR.exists():
        print("Error: The test folder does not exist!")
        return
    extensions = ("*.pdf", "*.html")
    files_found = []
    
    for ext in extensions:
        files_found.extend(DATA_DIR.rglob(ext))

    print(f"Found {len(files_found)} total files.")

    for file in files_found:
        print(f"Processing: {file.relative_to(DATA_DIR)}")
        ingest_file_multimodal(file)

@app.post("/chat")
def chat(request: ChatRequest):
    
    filters = extract_query_entities(request.message)
    print("filters",filters)
    where_clause = build_chroma_where(filters)
    print("where_clause",where_clause)
    
   
    results = get_vector_collection().query(
    query_texts=[request.message],
    where=where_clause, 
    n_results=7
)

# If no results found with filters, fallback to semantic search only
    if not results["documents"] or not results["documents"][0]:
        print("Warning: Strict filters returned 0 docs. Falling back to unfiltered search.")
        results = get_vector_collection().query(
            query_texts=[request.message],
            n_results=7
        )
   
    docs = results["documents"][0]
    
    reranked_docs = rerank_with_cross_encoder(
        request.message,
        docs,
        top_k=3
    )
    print("reranked_docs",reranked_docs)
    context = "\n\n".join(reranked_docs)
    # 2. Search Memory
    mem_results = get_memory_collection().query(
        query_texts=[request.message],
        where={"session_id": request.session_id},
        n_results=3
    )
    memory_context = "\n".join(mem_results["documents"][0]) if mem_results["documents"] else ""
    print("memory_context"+memory_context)
    history = SESSION_STORE[request.session_id]
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history[-5:]])

    answer = generate_answer(context, memory_context, history_text, request.message)
    
    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": answer})
    get_longterm_memory(request.message, answer, request.session_id, len(history))

    return {
        "reply": answer, 
        "sources": list(set(m.get("source") for m in results["metadatas"][0]))
    }

@app.get("/memory/{session_id}")
def get_all_memory(session_id: str):
    results = get_memory_collection().get(where={"session_id": session_id})
    memory_list = []
    if results["ids"]:
        for doc, meta in zip(results["documents"], results["metadatas"]):
            memory_list.append({"content": doc, "type": meta.get("type")})
    return {"memory": memory_list}

if __name__=="__main__":
    import uvicorn
    uvicorn.run("backend:app", host="127.0.0.1", port=8000, reload=True)