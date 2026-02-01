import streamlit as st
from src.loader import load_pdf
from src.embeddings import get_embeddings
from src.rag_pipeline import build_vectorstore, get_answer
from src.utils import save_uploaded_file

st.set_page_config(page_title = "Local RAG chatbot", layout = "wide")

st.title("📄 Local RAG Document Chatbot")
st.markdown("Runs fully offline using **Llama3 + Nomic Enbeddings**")

uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])

if uploaded_file:
    st.info("saving pdf...")
    pdf_path = save_uploaded_file(uploaded_file)

    st.info("Loading document...")
    documents = load_pdf(pdf_path)

    st.info("Creating embeddings + FAISS index...")
    embeddings = get_embeddings()
    vectorstore = build_vectorstore(documents, embeddings)

    question = st.text_input("Ask a question about your document:")

    if question:
        with st.spinner("Thinking..."):
            answer, sources = get_answer(vectorstore, question)

            st.subheader("🤖 Answer:")
            st.write(answer)

            with st.expander("Sources"):
                for i, doc in enumerate(sources):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content[:500])
                    st.markdown("---")