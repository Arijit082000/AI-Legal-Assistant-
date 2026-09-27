# ⚖️ AI Legal Assistant (Constitution & BNS)

This project is an AI-powered legal assistant designed to provide accurate answers to legal queries regarding the **Constitution of India** and the **Bharatiya Nyaya Sanhita (BNS) / IPC**. It is built using an advanced RAG (Retrieval-Augmented Generation) pipeline and powered by the Google Gemini 2.5 Flash model.

🚀 **Live App Link:** [Insert your Streamlit app link here] (e.g., https://your-app-name.streamlit.app)

---

## ✨ Key Features

* **Domain Filtering:** Users can filter queries to search strictly within the 'Constitution', 'BNS', or 'Both'.
* **Hybrid Search System:** Utilizes a combination of **BM25** (keyword-based search) and **ChromaDB** (semantic vector search) for highly accurate retrieval.
* **Advanced Prompt Engineering:** The LLM is equipped with strict guardrails to respect domain boundaries and correct users on terminology (e.g., distinguishing between Constitutional "Articles" and BNS "Sections").
* **Anti-Sleep Mechanism:** Includes a background JavaScript component to prevent the Streamlit Cloud app from going into sleep mode during inactivity.
* **Intelligent Data Cleaning:** Uses `PyMuPDF` to dynamically strip out page numbers and clean the text during the document loading phase, preventing context pollution.

---

## 🛠️ Tech Stack

* **Frontend:** Streamlit
* **LLM:** Google Gemini 2.5 Flash (`ChatGoogleGenerativeAI`)
* **Framework:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings:** HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`)
* **Document Processing:** PyMuPDF (fitz)

---

## 🚀 Local Setup & Installation

Follow these steps to run the project on your local machine:

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Install required dependencies
```bash
pip install streamlit langchain langchain-community langchain-google-genai langchain-huggingface chromadb pymupdf sentence-transformers rank_bm25
```

### 3. Add the PDF documents
Create a `data/` folder in the root directory and place the following PDF files inside it:
* `The constitution of India.pdf`
* `Indian Penal Code.pdf`

### 4. Configure API Keys
Create a `.streamlit` folder in the root directory and add a `secrets.toml` file inside it. Add your Google API key:
```toml
GOOGLE_API_KEY = "your_google_api_key_here"
```

### 5. Run the application
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```text
📦 AI-Legal-Assistant
 ┣ 📂 .streamlit
 ┃ ┗ 📜 secrets.toml         # API Keys (Do not commit to GitHub)
 ┣ 📂 data
 ┃ ┣ 📜 The constitution of India.pdf
 ┃ ┗ 📜 Indian Penal Code.pdf
 ┣ 📜 app.py                 # Main application script
 ┣ 📜 requirements.txt       # List of dependencies
 ┗ 📜 README.md              # Project documentation
```

---

## 👨‍💻 Developer
Developed by **Arijit Dasgupta**