import os
from dotenv import load_dotenv


load_dotenv(dotenv_path=r"E:\project\.env")

import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI


gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY environment variable not found. Make sure E:\\project\\.env exists!")

def run_autonomous_summarizer(pdf_path: str):
    print(f"--- 1. Ingesting PDF: {pdf_path} ---")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"Could not find the file at path: {pdf_path}")

    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            extracted = page.extract_text()
            if extracted:
                full_text += f"\n--- Page {page_num} ---\n" + extracted

    print(f"Extracted {len(full_text)} characters from the document.")

    
    print("--- 2. Splitting text into chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = text_splitter.split_text(full_text)
    print(f"Created {len(chunks)} text chunks.")

    
    print("--- 3. Generating Embeddings locally & Storing in FAISS Index ---")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)

    
    retrieved_docs = vector_store.similarity_search("abstract summary key architecture transformer results", k=5)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

    
    print("--- 4. Generating Summary using Gemini ---")
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)
    
    prompt = f"""
    You are an expert AI research assistant. Summarize the following academic research paper based on the retrieved context below.

    Provide a structured summary containing:
    1. **Title & Core Objective**
    2. **Key Innovation / Model Architecture**
    3. **Main Results & Findings**
    4. **Conclusion**

    Context:
    {context_text}
    """

    response = llm.invoke(prompt)

    print("\n" + "="*40 + " SUMMARY OUTPUT " + "="*40 + "\n")
    
    
    if isinstance(response.content, list):
        summary_text = "".join([item.get("text", "") for item in response.content if isinstance(item, dict)])
        print(summary_text)
    else:
        print(response.content)

    print("\n" + "="*96)

if __name__ == "__main__":
    PDF_FILE_PATH = r"C:\Users\Chethana Sailaja\Downloads\attention-is-all-you-need-Paper-2.pdf"
    
    run_autonomous_summarizer(PDF_FILE_PATH)
