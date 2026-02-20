import re
import os
import json
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_URL = os.getenv("DATABASE_URL")

st.set_page_config(page_title="SQL Chatbot", layout="wide")
st.title("Chat with DB")

# ----------------------------
# 1) Gemini Models
# ----------------------------
sql_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=GOOGLE_API_KEY
)

answer_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    api_key=GOOGLE_API_KEY
)

# ----------------------------
# 2) DB Engine
# ----------------------------
@st.cache_resource
def get_engine():
    return create_engine(DB_URL)

# ----------------------------
# 3) Get Schema
# ----------------------------
@st.cache_data(show_spinner=False)
def get_schema():
    engine = get_engine()
    schema_str = ""
    inspector_query = text("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(inspector_query)
            current_table = None
            for row in result:
                table_name, column_name = row[0], row[1]
                table_fmt = f'"{table_name}"'
                col_fmt = f'"{column_name}"'

                if table_name != current_table:
                    schema_str += f"\nTable: {table_fmt}\nColumns: "
                    current_table = table_name

                schema_str += f"{col_fmt}, "
    except Exception as e:
        return f"ERROR_READING_SCHEMA: {e}"

    return schema_str.strip()

# ----------------------------
# 4) Load Few-Shot RAG with SentenceTransformers
# ----------------------------
@st.cache_resource
def load_fewshot_retriever():
    # تحميل fewshots
    with open("fewshots.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = []
    texts = []
    for item in data:
        question = item.get("naturalQuestion")
        sql = item.get("sqlQuery")
        if question and sql:
            content = f"Question: {question}\nSQL: {sql}"
            documents.append(content)
            texts.append(content)
        else:
            print(f"Skipped invalid item: {item}")

    # استخدام موديل SentenceTransformers للـ embeddings
    embed_model = SentenceTransformer("intfloat/multilingual-e5-small")
    embeddings = embed_model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    # إنشاء FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity
    index.add(embeddings)

    # دالة retriever
    def retriever(query, top_k=3):
        q_emb = embed_model.encode([query], normalize_embeddings=True)
        q_emb = np.array(q_emb).astype("float32")
        scores, ids = index.search(q_emb, top_k)
        valid_ids = [i for i in ids[0] if 0 <= i < len(documents)]
        results = [documents[i] for i in valid_ids]
        return results

    return retriever

# استدعاء retriever
retriever = load_fewshot_retriever()

# ----------------------------
# 5) SQL Generation with RAG
# ----------------------------
def clean_sql(sql: str) -> str:
    sql = sql.strip()
    sql = re.sub(r"```sql?", "", sql, flags=re.IGNORECASE)
    sql = sql.replace("```", "").replace("`", "").strip()
    parts = [p.strip() for p in sql.split(";") if p.strip()]
    return parts[0] if parts else sql

def get_sql_from_llm(question, schema_str):
    # جلب الأمثلة من الملف
    docs = retriever(question, top_k=3)
    fewshot_examples = "\n\n".join(docs)

    full_prompt = f"""
You are a PostgreSQL expert.

I will provide you with some examples of questions and their SQL queries. 
IMPORTANT: Some examples might be in SQLite syntax. You MUST convert them to valid PostgreSQL syntax.

RULES:
1) Use double quotes for all identifiers (e.g., "Invoice", "InvoiceDate", "Total").
2) PostgreSQL does NOT have STRFTIME. 
   - Instead of STRFTIME('%Y', "Date"), use EXTRACT(YEAR FROM "Date"::timestamp).
   - Instead of STRFTIME('%m', "Date"), use EXTRACT(MONTH FROM "Date"::timestamp).
3) Always cast date columns to timestamp using ::timestamp before extracting.
4) Return ONLY the SQL query, no explanations.

Similar Examples (Transform these to PostgreSQL if needed):
{fewshot_examples}

Database Schema:
{schema_str}

User Question:
{question}

PostgreSQL Query:
"""
    result = sql_model.invoke(full_prompt)
    return clean_sql(result.content)

# ----------------------------
# 6) Run SQL
# ----------------------------
def run_sql(sql_query):
    engine = get_engine()
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(sql_query), conn)
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)

# ----------------------------
# 7) Natural Response
# ----------------------------
answer_prompt_template = """
User Question: {question}
SQL Result: {data}
Answer clearly. If empty say: "البيانات لا تعطي إجابة واضحة".
"""

answer_prompt = ChatPromptTemplate.from_template(answer_prompt_template)

def get_natural_answer(question, sql_query, df):
    chain = answer_prompt | answer_model
    data_text = df.to_string(index=False) if not df.empty else "EMPTY"
    result = chain.invoke({"question": question, "data": data_text})
    return result.content.strip()

# ----------------------------
# 8) UI
# ----------------------------
schema_str = get_schema()

question = st.text_input("Ask question for database:")

if st.button("Ask"):
    if question.strip():
        sql_query = get_sql_from_llm(question, schema_str)

        st.subheader("1) Generated SQL")
        st.code(sql_query, language="sql")

        df, err = run_sql(sql_query)

        st.subheader("2) SQL Result")
        if err:
            st.error(f"SQL Error: {err}")
        else:
            st.dataframe(df)
            st.subheader("3) Natural Response")
            st.write(get_natural_answer(question, sql_query, df))
