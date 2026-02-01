from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama

def build_vectorstore(documents, embeddings):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def get_answer(vectorstore, question: str):
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    llm = Ollama(model="llama3:8b")
    prompt = f""" You are a helpful AI assistant. 
        Answer ONLY using the context below.
        Context: 
        {context}

        Question: 
        {question}
    Answer Clearly  """
    return llm.invoke(prompt), docs

