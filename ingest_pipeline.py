import os
from pathlib import Path
import pymupdf
import pymupdf4llm
from openai import OpenAI
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vector_store import get_vector_collection

client = OpenAI()

class EquityMetadata(BaseModel):
    ticker: str
    company_name: str
    report_type: str
    fiscal_year: int

def extract_metadata_ai(file_path: str):
    doc = pymupdf.open(file_path)
    cover_text = "".join([page.get_text() for page in doc[:2]])
    doc.close()

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract financial metadata from the cover."},
            {"role": "user", "content": f"Text: {cover_text[:3000]}"}
        ],
        response_format=EquityMetadata,
    )
    return completion.choices[0].message.parsed.model_dump()

def process_file(file_path: Path):
    collection = get_vector_collection()
    
    # 1. Peek at Metadata to generate ID
    metadata = extract_metadata_ai(str(file_path))
    first_chunk_id = f"{metadata['ticker']}_{metadata['fiscal_year']}_p0_s0"
    
    # 2. Check if already indexed (Saves credits!)
    existing = collection.get(ids=[first_chunk_id], include=[])
    if existing and existing.get("ids"):
        print(f" Skipping {file_path.name} (Already in DB)")
        return

    # 3. Full Parse
    print(f" Processing: {file_path.name}...")
    raw_pages = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=400)
    all_docs, all_metas, all_ids = [], [], []

    for i, page in enumerate(raw_pages):
        text = page["text"].strip()
        if not text: continue
        
        chunks = text_splitter.split_text(text)
        for sub_i, chunk_text in enumerate(chunks):
            meta = {**metadata, **page["metadata"], "source": file_path.name}
            all_docs.append(chunk_text)
            all_metas.append({k: str(v) for k, v in meta.items()})
            all_ids.append(f"{metadata['ticker']}_{metadata['fiscal_year']}_p{i}_s{sub_i}")

    # 4. Batch Upload (Stay under 300k token limit)
    batch_size = 50
    for j in range(0, len(all_docs), batch_size):
        collection.add(
            documents=all_docs[j : j + batch_size],
            metadatas=all_metas[j : j + batch_size],
            ids=all_ids[j : j + batch_size]
        )
    print(f" Indexed {len(all_docs)} chunks for {metadata['ticker']}")