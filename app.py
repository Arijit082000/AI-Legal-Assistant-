import os
import time
import streamlit as st
import streamlit.components.v1 as components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Sleep Mode Fix ---
# If inactive for a few hours on Streamlit Cloud, the app goes to sleep.
# This JavaScript code sends a signal from the browser every 5 minutes to keep the session active.
def prevent_sleep_mode():
    components.html(
        """
        <script>
        setInterval(function() {
            window.parent.document.dispatchEvent(new Event('mousemove'));
            console.log("Anti-sleep ping sent.");
        }, 300000); // Every 5 minutes
        </script>
        """,
        height=0,
        width=0,
    )

st.set_page_config(page_title="AI Legal Assistant", layout="wide")
prevent_sleep_mode()

st.title("⚖️ AI Legal Assistant (Constitution & BNS)")

# --- API Key Setup ---
api_key = st.secrets.get("GOOGLE_API_KEY")
if not api_key:
    st.error("Please add GOOGLE_API_KEY to your Streamlit secrets.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key

# --- Data Loading and Caching ---
@st.cache_resource
def load_and_process_documents():
    # Note: These files must be present in your local directory or GitHub repository
    CONSTITUTION_PDF_PATH = "data/The constitution of India.pdf"
    BNS_PDF_PATH = "data/BNS.pdf" # Your previous code had IPC, BNS is used here

    def load_pdf(file_path, category_name):
        try:
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            docs = text_splitter.split_documents(pages)
            for doc in docs:
                doc.metadata["source_category"] = category_name
            return docs
        except Exception as e:
            st.error(f"Error loading {file_path}: {e}")
            return []

    const_docs = load_pdf(CONSTITUTION_PDF_PATH, "Constitution")
    bns_docs = load_pdf(BNS_PDF_PATH, "BNS")
    
    return const_docs, bns_docs

const_docs, bns_docs = load_and_process_documents()
all_docs = const_docs + bns_docs

@st.cache_resource
def setup_retrievers(_docs_list):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Creating Chroma Vector Store
    vector_store = Chroma.from_documents(
        documents=_docs_list,
        embedding=embeddings,
        collection_name="legal_docs_local_final_db"
    )
    
    return vector_store, _docs_list

# --- Filtering Logic based on Dropdown ---
selection = st.selectbox("Select Domain:", ["Both", "Constitution", "BNS"])

# Filter documents based on selection
if selection == "Constitution":
    filtered_docs = const_docs
elif selection == "BNS":
    filtered_docs = bns_docs
else:
    filtered_docs = all_docs

# Setup retriever based on filtered data
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(documents=filtered_docs, embedding=embeddings)

bm25_retriever = BM25Retriever.from_documents(filtered_docs)
bm25_retriever.k = 10
vector_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
)

# --- LLM and Prompt Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash") # Stable version for Streamlit

template = """
You are an expert AI Legal Assistant specializing in the Constitution of India, and BNS.

First, carefully review the retrieved context below (pay attention to the [Source: ...] tags).
Try to answer the user's question completely based on this context.

⚠️ CRITICAL FALLBACK RULE (THE INDEX TRAP BYPASS):
If the context is incomplete, missing, or only shows the index/heading, DO NOT refuse to answer.
Instead, gracefully use your internal expert training to provide the exact legal text and explanation.
If you use your internal knowledge, you MUST append this exact disclaimer at the bottom:
"*(Note: Retrieved PDF context was limited by the Table of Contents trap. This full detail is provided from my internal legal knowledge base.)*"

Context:
{context}

Question: {question}

Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    clean_texts = []
    for doc in docs:
        source = doc.metadata.get("source_category", "Unknown Source")
        safe_text = f"[Source: {source}]\n{doc.page_content}"
        clean_texts.append(safe_text)
    return "\n\n---\n\n".join(clean_texts)

# --- Chat Interface ---
user_question = st.text_input("🧑‍⚖️ Your Question:")

if st.button("Get Answer"):
    if user_question:
        with st.spinner("Analyzing legal documents..."):
            try:
                docs = retriever.invoke(user_question)
                context_text = format_docs(docs)
                
                chain = prompt | llm | StrOutputParser()
                response = chain.invoke({"context": context_text, "question": user_question})
                
                st.markdown("### ✅ Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")

