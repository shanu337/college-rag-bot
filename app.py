import os
import streamlit as str  # Using 'str' or another safe alias to avoid namespace issues
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- CONFIGURATION & PATHS ---
DOCS_DIR = "docs"
DB_DIR = "chroma_db"
MODEL_NAME = "qwen-hermes:latest"
EMBED_MODEL_NAME = "nomic-embed-text"

# Ensure the docs directory exists
os.makedirs(DOCS_DIR, exist_ok=True)

st.set_page_config(page_title="Hermes Academic Advisor", page_icon="🎓", layout="wide")
st.title("🎓 Hermes AI Academic Advisor")
st.subheader("Your localized B.Tech CSE Syllabus & Textbook Brain")

# --- INITIALIZE LLM & EMBEDDINGS ---
@st.cache_resource
def init_llm_components():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL_NAME)
    llm = OllamaLLM(model=MODEL_NAME)
    return embeddings, llm

embeddings, llm = init_llm_components()

# --- VECTOR DATABASE INITIALIZATION ---
def get_vector_store():
    # If the database directory exists and has files, load it directly
    if os.path.exists(DB_DIR) and len(os.listdir(DB_DIR)) > 0:
        st.info("⚡ Loading existing knowledge base from storage...")
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # Otherwise, check for PDFs to build a new one
    else:
        pdf_files = [f for f in os.listdir(DOCS_DIR) if f.endswith('.pdf')]
        if not pdf_files:
            st.warning("📥 Your knowledge base is empty. Please drop your B.Tech syllabus or textbook PDFs into the 'docs/' directory to get started.")
            return None
        
        with st.spinner("📚 Processing new PDFs and generating local embeddings... This might take a second."):
            loader = PyPDFDirectoryLoader(DOCS_DIR)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            
            # Create and persist the vector store
            vector_store = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings, 
                persist_directory=DB_DIR
            )
            st.success(f"✅ Successfully processed {len(pdf_files)} files into {len(chunks)} knowledge chunks!")
            return vector_store

vector_store = get_vector_store()

# --- RAG CHAIN SETUP ---
if vector_store:
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    template = """You are a brilliant AI academic advisor helping a college student. 
    Use the following pieces of retrieved context from the syllabus/textbook to answer the question. 
    If you don't know the answer or if it's not in the context, say that you don't know based on the provided documents. Do not hallucinate.

    Context:
    {context}

    Question: {question}
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # --- CHAT INTERFACE ---
    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display past chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if user_query := st.chat_input("Ask me anything about your upcoming semester courses..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            with st.spinner("Hermes is scanning documents..."):
                response = rag_chain.invoke(user_query)
            response_placeholder.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})
else:
    st.info("ℹ️ Once you drop files into the 'docs/' folder, refresh this web page to compile your engine.")
