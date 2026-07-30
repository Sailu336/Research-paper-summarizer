# 📄 Research Paper Summarizer

An automated RAG-based (Retrieval-Augmented Generation) pipeline that extracts, indexes, and summarizes academic research papers using **LangChain**, **FAISS**, **HuggingFace Embeddings**, and **Google Gemini**.


## 📌 Architecture & Tech Stack

* **PDF Parsing:** `pdfplumber`
* **Text Chunking:** `RecursiveCharacterTextSplitter`
* **Embeddings:** HuggingFace (`all-MiniLM-L6-v2`)
* **Vector Store:** FAISS (Facebook AI Similarity Search)
* **LLM Reasoning:** Google Gemini API (`langchain-google-genai`)

## 🚀 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install required packages:

bash
git clone [https://github.com/Sailu336/Research-paper-summarizer.git](https://github.com/Sailu336/Research-paper-summarizer.git)
cd Research-paper-summarizer
pip install pdfplumber langchain langchain-community langchain-huggingface langchain-google-genai faiss-cpu python-dotenv sentence-transformers

##  Program Output

### 1. Ingestion & RAG Execution
![Execution Log](./Screenshot%202026-07-29%20.png)

### 2. Structured Summary Output
![Summary Output](./Screenshot%202026-07-29%20233316.png)
