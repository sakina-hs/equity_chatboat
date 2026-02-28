import pandas as pd
import os
from dotenv import load_dotenv
from datasets import Dataset

# Ragas v0.2+ Standard Imports
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from vector_store import get_vector_collection
from llm_prompts import generate_answer
from retrieval_utils import extract_query_entities, rerank_with_cross_encoder

load_dotenv()

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

def run_evaluation(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"Error: Could not find {csv_path}")
        return
        
    df_testset = pd.read_csv(csv_path)
    collection = get_vector_collection()
    
    
    eval_data = {
        "question": [], 
        "contexts": [], 
        "answer": [], 
        "ground_truth": []
    }

    print(f"Starting evaluation on {len(df_testset)} samples...")

    for i, row in df_testset.iterrows():
        
        question = row["user_input"]
        ground_truth = row["reference"]
        
        # 1. NER Extraction (GLiNER)
        filters = extract_query_entities(question)
        where_clause = build_chroma_where(filters)
        
        # 2. Vector Search with Metadata Filtering
        results = collection.query(
            query_texts=[question],
            where=where_clause,
            n_results=7 # Top 7 before reranking
        )

        # Fallback if filters are too strict 
        if not results["documents"] or not results["documents"][0]:
            print(f"Warning: Filters returned 0 docs for '{question[:30]}...'. Falling back.")
            results = collection.query(query_texts=[question], n_results=7)
        
        initial_docs = results["documents"][0]
        
        # 3. Cross-Encoder Reranking
        reranked_docs = rerank_with_cross_encoder(
            question,
            initial_docs,
            top_k=3 # Matching your production top_k
        )
        
        # 4. Generation
        # Context is joined for the LLM but Ragas needs the list for 'contexts'
        context_string = "\n\n".join(reranked_docs)
        answer = generate_answer(
            context=context_string,
            memory_context="", 
            history_text="", 
            question=question
        )

        # 5. Populate Eval Data
        eval_data["question"].append(question)
        eval_data["contexts"].append(reranked_docs) # List of strings
        eval_data["answer"].append(answer)
        eval_data["ground_truth"].append(ground_truth)
        
        print(f"[{i+1}/{len(df_testset)}] Processed: {question[:50]}...")

    # 6. Setup Ragas Evaluator (Modern v0.2+ Syntax)
    dataset = Dataset.from_dict(eval_data)
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
    evaluator_embeddings = OpenAIEmbeddings()

    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextPrecision(llm=evaluator_llm),
        ContextRecall(llm=evaluator_llm)
    ]

    # 7. Run Evaluation
    print("\nCalculating Ragas Metrics...")
    result = evaluate(dataset=dataset, metrics=metrics)
    
    print("\n" + "="*30)
    print("FINAL AUDITOR BOT EVALUATION")
    print(result)
    
    return result.to_pandas()

if __name__ == "__main__":
    input_csv = "testset_2023-Financial-Report.csv" 
    results_df = run_evaluation(input_csv)
    
    if results_df is not None:
        results_df.to_csv("evaluation_report_gliner.csv", index=False)
        print("Success! Results saved to evaluation_report_gliner.csv")