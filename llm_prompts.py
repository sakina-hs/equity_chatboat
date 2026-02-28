import json
import io
import re
from typing import Dict, Optional, List,Literal
from pydantic import BaseModel, Field
from openai import OpenAI
from PIL import Image
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from vector_store import get_memory_collection
import base64
client = OpenAI()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

class MemoryExtraction(BaseModel):
    type: Literal["preference", "none"] = Field(description="Is this a user preference or no relevant info?")
    value: str = Field(description="The summarized investment preference or profile info.")

class PageFinancialMetadata(BaseModel):
    document_section: str = Field(description="e.g., Risk Factors, MD&A, Financial Statements")
    primary_metrics: List[str] = Field(description="List of metrics: EBITDA, Revenue, etc.")
    data_focus: str = Field(description="Main topic: e.g., Debt Profile, Segment Growth")
    contains_tables: bool
    ticker: str
    company_name: str
    report_type: str
    fiscal_year: str

    
def extract_document_metadata(vision_descriptions: str) -> dict:
    """
    This function now receives the visual descriptions of the first 3 pages.
    """
    prompt = f"""
    Based on the following visual descriptions of a document's cover pages, 
    extract the structured metadata.
    
    DESCRIPTIONS:
    {vision_descriptions}
    """
    
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a financial data extractor. Use the visual descriptions provided to identify the company and report details. If the info is missing, use 'Unknown'."},
            {"role": "user", "content": prompt}
        ],
        response_format=EquityMetadata
    )
    return completion.choices[0].message.parsed.model_dump()
'''
def plan_query(user_query: str):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "Return JSON: {search_query: str, filters: dict}"}, 
                  {"role": "user", "content": user_query}]
    )
    try:
        return json.loads(clean_json_string(response.choices[0].message.content))
    except:
        return {"search_query": user_query, "filters": {}}
'''

import re
def generate_answer(context: str, memory_context: str, history_text: str, question: str):
    system_prompt = (
        "You are a professional financial auditor. Use the provided context to answer questions accurately. "
        "If the answer is not in the context, state that you do not have enough information. "
        "Maintain a formal tone and reference specific financial figures."
    )

    user_content = f"""
### USER PROFILE & PREFERENCES:
{memory_context if memory_context else "No specific preferences known."}

### RECENT CONVERSATION:
{history_text if history_text else "This is the start of the conversation."}

### DOCUMENT CONTEXT:
{context}

### QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0
    )
    return response.choices[0].message.content

def explain_image_semantically(image_bytes: bytes) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system", 
                "content": "You are a specialized data extractor. If the image contains important text, financial data, or meaningful information, extract it clearly. If the image is just a logo, seal, decoration, or contains no useful data, respond with an empty string: ''"
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract all meaningful information from this image. If none, return nothing."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}","detail": "low"}}
                ]
            }
        ],
        temperature=0 
    )
    print(response.usage)
    content = response.choices[0].message.content.strip()
    
    
    if "unable to" in content.lower() or "sorry" in content.lower():
        return ""
        
    return content

def extract_page_metadata(page_content: str) -> dict:
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": page_content[:4000]}],
        response_format=PageFinancialMetadata
    )
    return completion.choices[0].message.parsed.model_dump()

def get_longterm_memory(user_input, answer, session_id, history_len):
    prompt = f"""
    Analyze the conversation and extract any specific user investment preferences, 
    risk tolerance, or financial profile information.
    
    User: {user_input}
    AI: {answer}
    """

    try:
       
        completion = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a memory module for a financial advisor AI. Extract user profile details into structured JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format=MemoryExtraction
        )

        
        data = completion.choices[0].message.parsed

        if data.type != "none" and data.value:
            get_memory_collection().add(
                documents=[data.value],
                metadatas=[{"session_id": session_id, "type": data.type}],
                ids=[f"{session_id}_{history_len}"]
            )
            print(f"✅ MEMORY STORED: {data.value}")
        else:
            print("ℹ️ MEMORY SKIPPED: No relevant info found.")

    except Exception as e:
        print(f" MEMORY ERROR: {e}")