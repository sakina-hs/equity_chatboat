import os
from openai import OpenAI
from vector_store import get_vector_collection
from ingest_pipeline import process_file
from pathlib import Path

client = OpenAI()

def run_ingestion():
    #Scans the /data folder and processes any new files.
    print("\nScanning for documents...")
    extensions = ["*.pdf", "*.html", "*.htm"]
    data_path = Path("./data")
    files = []
    for ext in extensions:
        files.extend(list(data_path.rglob(ext)))
    
    if not files:
        print("No files found in ./data/")
        return

    for file_path in files:
        process_file(file_path)
    print("\nIngestion check complete.")

def chat_loop():
    # Starts an interactive session to query data.
    collection = get_vector_collection()
    print("\n--- Equity Analyst Chat Active ---")
    print("Type 'exit' to quit or 'ingest' to update data.\n")

    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            break
        if user_input.lower() == 'ingest':
            run_ingestion()
            continue
        if not user_input:
            continue

        # 1. Semantic Search (Retrieval)
        # We ask Chroma to find the top 5 most relevant chunks
        results = collection.query(
            query_texts=[user_input],
            n_results=5
        )

        # 2. Context Preparation
        context = "\n\n".join(results['documents'][0])
        sources = list(set([m.get('source', 'Unknown') for m in results['metadatas'][0]]))

        # 3. AI Generation (Augmentation)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a senior financial analyst. Answer questions using ONLY the provided context. If the answer isn't there, say you don't know. Always list the sources used."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {user_input}"}
            ]
        )

        print(f"\nAI: {response.choices[0].message.content}")
        print(f"📍 Sources: {', '.join(sources)}\n")

if __name__ == "__main__":
    # You can choose to run ingestion automatically at startup
    # run_ingestion() 
    chat_loop()