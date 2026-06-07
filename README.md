# 🟢 NVIDIA 10-K AI Analyst: Advanced Hybrid RAG Chatbot

An interactive, high-performance RAG (Retrieval-Augmented Generation) web application designed to analyze and answer complex queries on **NVIDIA’s 2025 Annual Report (10-K)**. The system retrieves grounded context directly from the official SEC filing and synthesizes precise, professional answers with automatic source-page citations.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

---

## 🚀 Key Features

* **Grounded Question Answering:** Generates responses strictly bound to retrieved context to eliminate hallucinations.
* **Hybrid Search Retrieval:** Combines semantic vector search (for concept matching) and BM25 lexical search (for precise keyword/financial matches).
* **Neural Reranking:** Filters and prioritizes candidate passages using a Cross-Encoder transformer model.
* **Source Citations:** Automatically displays the source SEC section name and page numbers for verified claims.
* **Interactive UI:** Built using Streamlit with a premium "Midnight Ocean" custom dark theme and clickable suggested questions.

---

## 🛠️ Tech Stack & AI Models

* **LLM Generator:** **`Google Gemini 1.5 Flash`** (via `gemini-flash-lite-latest` alias) for fast, accurate conversational synthesis.
* **Text Embeddings:** **`BAAI/bge-m3`** (state-of-the-art multilingual/multi-functional embedding model).
* **Vector Store:** **ChromaDB** for local persistence and low-latency vector indexing.
* **Neural Reranker:** **`cross-encoder/ms-marco-MiniLM-L-6-v2`** for computing precise query-passage relevance scores.
* **Keyword Search:** **BM25** (Okapi implementation) for exact matching on numbers, codes, and names.
* **Frontend:** **Streamlit** (Python web framework).

---

## 🔧 System Architecture

The pipeline executes the following stages on every user query:
1. **Dense Retrieval:** Encodes the query using `bge-m3` and finds the top-20 semantic matches in `ChromaDB`.
2. **Sparse Retrieval:** Tokenizes the query and finds the top-20 keyword matches in the corpus using `BM25`.
3. **Rank Fusion:** Fuses and deduplicates candidates using Reciprocal Rank Fusion (RRF).
4. **Reranking:** Feeds the top candidates to the `cross-encoder` model, scoring and sorting down to the absolute best **7** passages.
5. **Grounded Synthesis:** Constructs a detailed prompt enclosing the passages and sends it to the `Gemini` API to generate the final response.

---

## 💻 Local Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/saisharanya06/nvidia-rag-chatbot.git
   cd nvidia-rag-chatbot
   ```

2. **Set Up Virtual Environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Secrets:**
   Create a `.env` file in the root directory:
   ```env
   GEMINI_API_KEY=your_google_gemini_api_key
   ```

5. **Run the App:**
   ```bash
   streamlit run app.py
   ```

---

## 🌐 Cloud Deployment (Streamlit Community Cloud)

1. Connect your GitHub repository to [share.streamlit.io](https://share.streamlit.io/).
2. In the **Secrets** configuration section of Streamlit settings, enter your API key in TOML format:
   ```toml
   GEMINI_API_KEY = "your_actual_gemini_api_key"
   ```
3. Deploy! The precompiled database files are included in the repository, so the website will be fully ready to query out-of-the-box.
