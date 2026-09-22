import os
import time
import streamlit as st
import streamlit.components.v1 as components
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Sleep Mode Prevention ---
def prevent_sleep_mode():
    components.html(
        """
        <script>
        setInterval(function() {
            window.parent.document.dispatchEvent(new Event('mousemove'));
            console.log("Anti-sleep ping sent.");
        }, 300000);
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

# --- Document Loading with Caching ---
@st.cache_resource
def load_and_split_docs():
    CONSTITUTION_PDF_PATH = "data/The constitution of India.pdf"
    BNS_PDF_PATH = "data/Indian Penal Code.pdf"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    def process(path, category):
        try:
            loader = PyPDFLoader(path)
            docs = loader.load()
            split_docs = splitter.split_documents(docs)
            for d in split_docs:
                d.metadata["source_category"] = category
            return split_docs
        except Exception as e:
            st.error(f"Error loading {path}: {e}")
            return []

    const_docs = process(CONSTITUTION_PDF_PATH, "Constitution")
    bns_docs = process(BNS_PDF_PATH, "BNS")
    return const_docs, bns_docs

const_docs, bns_docs = load_and_split_docs()

# --- Cached Retriever Setup per Domain ---
@st.cache_resource
def get_retrievers_for_domain(domain_name):
    if domain_name == "Constitution":
        selected_docs = const_docs
    elif domain_name == "BNS":
        selected_docs = bns_docs
    else:
        selected_docs = const_docs + bns_docs

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Unique collection name to avoid Chroma mixing up the domains
    vector_store = Chroma.from_documents(
        documents=selected_docs,
        embedding=embeddings,
        collection_name=f"legal_db_{domain_name.lower()}"
    )

    bm25 = BM25Retriever.from_documents(selected_docs)
    bm25.k = 7
    chroma_ret = vector_store.as_retriever(search_kwargs={"k": 7})

    # Returning both retrievers separately to bypass EnsembleRetriever errors
    return bm25, chroma_ret

# --- UI Controls ---
selection = st.selectbox("Select Domain:", ["Both", "Constitution", "BNS"])
bm25_retriever, chroma_retriever = get_retrievers_for_domain(selection)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

template = """
You are an expert AI Legal Assistant.
Current Selected Domain Filter: {selected_domain}

STRICT DOMAIN BOUNDARY RULE:
1. If Current Selected Domain is 'Constitution': Answer ONLY using the Constitution of India. If the question asks about IPC/BNS sections, explicitly refuse and state: "This topic falls under BNS/IPC. Please change the domain filter to BNS or Both."
2. If Current Selected Domain is 'BNS': Answer ONLY using BNS / IPC. If the question asks about Articles or Constitutional topics (such as Article 370, Fundamental Rights, etc.), explicitly refuse and state: "This topic falls under the Constitution of India. Please change the domain filter to Constitution or Both."
3. If Current Selected Domain is 'Both': You can freely answer from either or both sources.
4. TERMINOLOGY RULE: The Constitution contains "Articles", whereas BNS/IPC contain "Sections". If a user asks for a "Section" within the Constitution, or an "Article" within BNS/IPC, immediately point out the incorrect terminology. NEVER invent or hallucinate a Constitutional Section.

Fallback Rule (Index Trap):
If the question is within the allowed domain but the retrieved context only has headings or table-of-contents fragments, answer from internal knowledge and add:
"*(Note: Retrieved PDF context was limited by the Table of Contents trap. This full detail is provided from my internal legal knowledge base.)*"

Context:
{context}

Question: {question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    # Custom logic to remove duplicate contents
    seen_texts = set()
    clean_texts = []
    for d in docs:
        text = d.page_content
        if text not in seen_texts:
            seen_texts.add(text)
            source = d.metadata.get("source_category", "Unknown")
            clean_texts.append(f"[Source: {source}]\n{text}")
    return "\n\n---\n\n".join(clean_texts)

user_question = st.text_input("🧑‍⚖️ Your Question:")

if st.button("Get Answer"):
    if user_question:
        with st.spinner("Searching strictly within selected domain..."):
            try:
                # Manually combining BM25 and ChromaDB results
                bm25_docs = bm25_retriever.invoke(user_question)
                chroma_docs = chroma_retriever.invoke(user_question)
                combined_docs = bm25_docs + chroma_docs
                
                context_text = format_docs(combined_docs)
                
                chain = prompt | llm | StrOutputParser()
                response = chain.invoke({
                    "context": context_text,
                    "question": user_question,
                    "selected_domain": selection
                })
                
                st.markdown("### ✅ Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"Error: {e}")
                
