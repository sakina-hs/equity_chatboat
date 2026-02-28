import os
from pathlib import Path
from dotenv import load_dotenv

from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader

load_dotenv()

def generate_financial_testset(file_path: str, test_size: int = 10):
    # 1. Load the PDF
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    
    # 2. Setup the Models (Required for v0.3+)
    generator_llm = LangchainLLMWrapper(CChatOpenAI(model="gpt-4o-mini"))
    generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    # 3. Initialize the Generator
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings
    )

    # 4. Generate the testset
    # This helper handles the Knowledge Graph building automatically
    dataset = generator.generate_with_langchain_docs(
        documents,
        testset_size=test_size
    )

    # 5. Export to CSV
    df = dataset.to_pandas()
    output_path = f"testset_{Path(file_path).stem}.csv"
    df.to_csv(output_path, index=False)
    
    print(f" Success! Generated {test_size} questions at {output_path}")
    return df

if __name__ == "__main__":
    target_pdf = "./test/2023-Financial-Report.pdf"
    if os.path.exists(target_pdf):
        generate_financial_testset(target_pdf)
    else:
        # Create the directory if it doesn't exist to help you out
        os.makedirs("./test", exist_ok=True)
        print(f"File not found. Please put your PDF at: {target_pdf}")