# 📄 Local Document ChatGPT (RAG)

A fully offline Retrieval-Augmented Generation chatbot that allows users to upload PDFs and ask questions using:

- Llama3 (Ollama)
- Nomic Embeddings
- FAISS Vector Search
- Streamlit UI

## Features
✅ Works without API keys  
✅ Runs locally  
✅ Source chunk preview  
✅ Accurate retrieval with MMR  

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3
ollama pull nomic-embed-text
streamlit run app.py
