import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load PDFs from the docs directory
print("📚 Loading syllabus/textbook PDFs...")
loader = PyPDFDirectoryLoader("docs")
docs = loader.load()

if not docs:
    print("❌ No PDFs found in the 'docs' folder! Add some files first.")
    exit()

# 2. Split text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)
print(f"✂️ Split documents into {len(chunks)} chunks.")

# 3. Setup local Embeddings and Vector DB
# Note: Ensure your Ollama server is running in the background
print("🧠 Generating local embeddings and building Vector DB...")
embeddings = OllamaEmbeddings(model="qwen-hermes:latest") # Or your exact Hermes/Qwen model tag
vector_store = Chroma.from_documents(chunks, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# 4. Define the prompt structure for Hermes/Qwen
template = """You are a brilliant AI academic advisor helping a college student. 
Use the following pieces of retrieved context from the syllabus/textbook to answer the question. 
If you don't know the answer or if it's not in the context, say that you don't know based on the provided documents. Do not hallucinate.

Context:
{context}

Question: {question}
Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# 5. Initialize your local LLM via Ollama
# Replace with your exact model string if it's named 'hermes3' or similar
llm = Ollama(model="qwen-hermes:latest") 

# 6. Construct the RAG Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Interactive Chat Loop
print("\n🚀 System Ready! Ask your local AI anything about your college documents (Type 'exit' to quit):\n")
while True:
    query = input("🧔 You: ")
    if query.lower() == 'exit':
        break
    if not query.strip():
        continue
    
    print("\n🤖 AI is thinking...")
    response = rag_chain.invoke(query)
    print(f"\n🤖 AI:\n{response}\n")
    print("-" * 50)
