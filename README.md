# SQL Chatbot with Google Gemini (Few-Shot RAG)

This project implements an intelligent **SQL Chatbot** that allows users to interact with a PostgreSQL database using natural language.

The system automatically:
- Converts user questions into SQL queries
- Retrieves similar examples using embeddings (Few-Shot RAG)
- Executes the generated SQL query
- Returns a clear natural language answer

---

## 🚀 Technologies Used

- **Google Gemini (gemini-2.5-flash)** via LangChain
- **Streamlit** for the web interface
- **PostgreSQL**
- **SQLAlchemy** for database connection
- **Pandas** for query result handling
- **Sentence Transformers** for embeddings
- **FAISS** for vector similarity search
- **Few-Shot Retrieval (RAG-based prompting)**

---

## 🧠 How It Works

1. The user enters a natural language question.
2. The system retrieves the most similar examples from a Few-Shot dataset using:
   - SentenceTransformer embeddings
   - FAISS vector search
3. The selected examples are injected dynamically into the prompt.
4. Gemini generates a PostgreSQL-compatible SQL query.
5. The query is executed using SQLAlchemy.
6. The results are converted into a natural language answer.

This is a **Dynamic Few-Shot + Retrieval-Augmented Generation (RAG)** approach.

---

## ⚙️ Setup

1.Create virtual environment: python -m venv venv

2.Activate: venv\Scripts\activate

3.Install dependencies: pip install -r requirements.txt

4.Create .env file: GOOGLE_API_KEY=your_api_key_here

5.Run: python Streamlit run dynamic.py
