import fitz
import pymupdf4llm
import hashlib
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vector_store import get_vector_collection
from llm_prompts import extract_page_metadata, explain_image_semantically
import time

# 1. Define Chunking Strategy
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100) 
collection = get_vector_collection()

def compute_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def safe_metadata(meta):
    # Ensure metadata is string, int, or float
    return {str(k): (", ".join(v) if isinstance(v, list) else v) for k, v in meta.items() if v is not None}

def ingest_file_multimodal(file_path: Path):
    file_hash = compute_file_hash(file_path)
    if collection.get(where={"file_hash": file_hash}, limit=1)["ids"]:
        print(f"Skipping {file_path.name}, already in database.")
        return

    doc = None
    try:
        doc = fitz.open(file_path)
        # Using a higher DPI for better OCR, but low resolution for image description to save tokens
        md_pages = pymupdf4llm.to_markdown(file_path, page_chunks=True)
        
        all_chunks = []

        # 2. Process each page
        for i, page in enumerate(doc):
            try:
                # 3. Vision Analysis
                pix = page.get_pixmap(dpi=100)
                img_bytes = pix.tobytes("png")
                
                # Check for images/drawings
                if len(page.get_images()) > 0 or len(page.get_drawings()) > 0:  
                    print(f"  -> Analyzing visuals for page {i+1}...")
                    img_ctx = explain_image_semantically(img_bytes)
                else:
                    img_ctx = "" 

                # 4. Combine Text + Vision
                text = md_pages[i]["text"] if i < len(md_pages) else ""
                combined_content = f"{text}\n{img_ctx}"
                
                # 5. Metadata Extraction
                page_meta = extract_page_metadata(combined_content)
                
                d = Document(
                    page_content=combined_content,
                    metadata=safe_metadata({
                        **page_meta, 
                        "source": file_path.name, 
                        "file_hash": file_hash, 
                        "page": i + 1
                    })
                )
                all_chunks.extend(splitter.split_documents([d]))

            except Exception as page_err:
                print(f"  ! Error on page {i}: {page_err}")
                continue

        # 6. Save to Vector Store in SMALL BATCHES to avoid token limits
        if all_chunks:
            print(f"Uploading {len(all_chunks)} chunks in batches...")
            
            # --- UPDATED BATCHING LOGIC ---
            BATCH_SIZE = 50 # Start small, maybe 50-100
            for i in range(0, len(all_chunks), BATCH_SIZE):
                batch = all_chunks[i:i + BATCH_SIZE]
                
                print(f"  -> Uploading batch {i//BATCH_SIZE + 1}...")
                collection.add(
                    documents=[c.page_content for c in batch],
                    metadatas=[c.metadata for c in batch],
                    ids=[f"{file_hash}_{i+idx}" for idx in range(len(batch))]
                )
                # Sleep briefly to avoid hitting rate limits (TPM/RPM)
                time.sleep(1)
            
            print(f"Successfully finished: {file_path.name}")

    except Exception as e:
        print(f"CRITICAL ERROR processing {file_path.name}: {e}")
    finally:
        if doc:
            doc.close()

if __name__=="__main__":
    ingest_file_multimodal("test/2023-Financial-Report.pdf")