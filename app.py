import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

# ------------------- CONFIG -------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DB_URL = os.getenv("DB_URL")

st.set_page_config(page_title="SQL Chatbot", page_icon=":bar_chart:", layout="wide")
st.title("Chat with Database :")

# ------------------- DATABASE -------------------
@st.cache_resource
def get_db_engine():
    return create_engine(DB_URL)

def get_schema():
    engine = get_db_engine()
    inspector_query = text("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position;
    """)
    schema_string = ""
    try:
        with engine.connect() as conn:
            result = conn.execute(inspector_query)
            current_table = None
            for row in result:
                table_name, column_name = row
                if table_name != current_table:
                    if current_table is not None:
                        schema_string += "\n"
                    schema_string += f"Table: {table_name}\n"
                    current_table = table_name
                schema_string += f"  - {column_name}\n"
    except Exception as e:
        st.error(f"Error fetching schema: {e}")
        return ""
    return schema_string

# ------------------- LLM INITIALIZATION -------------------
@st.cache_resource
def get_llm():
    return GoogleGenerativeAI(model="models/gemini-2.5-flash", api_key=GOOGLE_API_KEY)

# ------------------- HELPERS -------------------
def clean_sql(sql_text: str) -> str:
    sql_text = re.sub(r"```sql", "", sql_text, flags=re.IGNORECASE)
    sql_text = re.sub(r"```", "", sql_text)
    return sql_text.strip()

# ------------------- SQL GENERATION -------------------
SQL_TEMPLATE = """You are an expert PostgreSQL data analyst with deep expertise in writing efficient, accurate SQL queries.

DATABASE SCHEMA:
{schema}

USER QUESTION:
{user_question}

{conversation_context}

INSTRUCTIONS:
1. Analyze the question carefully to understand what data the user needs
2. Write a valid PostgreSQL query that accurately answers the question
3. Use double quotes around ALL table and column names exactly as shown in the schema
4. Handle edge cases:
   - If the question asks for counts, use COUNT()
   - If asking for aggregations, include appropriate GROUP BY
   - If comparing dates/times, use proper date functions
   - If filtering, use appropriate WHERE clauses
   - For "latest" or "most recent", use ORDER BY with LIMIT
   - For "average", "total", "sum", use appropriate aggregate functions
5. Optimize for performance:
   - Only select columns that are needed
   - Use indexes when available
   - Avoid SELECT * unless specifically needed
6. Handle ambiguity:
   - If the question mentions "test cases", look for tables/columns with "test", "case", "scenario", "execution", etc.
   - If asking about status, look for status/state columns
   - If asking about time periods, infer reasonable date ranges
7. Return ONLY the SQL query - no explanations, no markdown, no extra text

SQL QUERY:"""

def generate_sql_query(user_question, schema, conversation_history=None):
    conversation_context = (
        f"CONVERSATION CONTEXT: {conversation_history}" if conversation_history else ""
    )

    prompt = PromptTemplate.from_template(SQL_TEMPLATE)
    model = get_llm()
    chain = prompt | model | StrOutputParser()

    try:
        response = chain.invoke({
            "schema": schema,
            "user_question": user_question,
            "conversation_context": conversation_context,
        })
        return clean_sql(response)
    except Exception as e:
        st.error(f"Error generating SQL query: {e}")
        return ""

# ------------------- NATURAL LANGUAGE RESPONSE -------------------
NL_TEMPLATE = """You are a helpful data analyst assistant explaining query results in natural, conversational language.

USER'S QUESTION:
{question}

SQL QUERY EXECUTED:
{sql_query}

QUERY RESULTS:
{data}

INSTRUCTIONS:
1. Answer the user's question directly and conversationally
2. Present the data in an easy-to-understand format:
   - Use natural language, not technical jargon
   - Format numbers clearly (use commas for thousands, round decimals appropriately)
   - Present dates in readable format
   - If multiple rows, summarize key insights or patterns
3. Be specific and precise:
   - Include actual numbers and values from the results
   - Reference specific test cases, dates, or identifiers when relevant
   - Don't be vague - if the data shows "5 failed tests", say that, not "some tests failed"
4. Handle different scenarios:
   - If data is empty: "No relevant items were found matching your criteria."
   - If data is incomplete: Mention what's available and what might be missing
   - If results are surprising: Present them objectively
5. Add context when helpful:
   - If showing trends, mention if they're increasing/decreasing
   - If comparing values, highlight the differences
   - If showing percentages, include the underlying counts
6. Keep it concise but complete - answer the question fully without unnecessary elaboration
7. If the data genuinely doesn't answer the question, clearly state what information is missing.

RESPONSE:"""

def get_natural_language_response(question, data, sql_query=""):
    prompt = PromptTemplate.from_template(NL_TEMPLATE)
    model = get_llm()
    chain = prompt | model | StrOutputParser()

    try:
        response = chain.invoke({
            "question": question,
            "sql_query": sql_query,
            "data": data,
        })
        return response.strip()
    except Exception as e:
        st.error(f"Error generating natural language response: {e}")
        return "Error generating response."


# ------------------- STREAMLIT APP -------------------
schema = get_schema()
if not schema:
    st.stop()

user_question = st.text_input("Write a question about DB :")

if st.button("Get") and user_question:
    sql_query = generate_sql_query(user_question, schema)
    st.code(sql_query, language="sql")

    if not sql_query.lower().startswith("select"):
        st.warning("LLM did not generate a SELECT query. Cannot execute.")
        result_df = pd.DataFrame()
    else:
        try:
            engine = get_db_engine()
            with engine.connect() as conn:
                result_df = pd.read_sql(sql_query, conn)
            st.dataframe(result_df)
        except Exception as e:
            st.error(f"Error executing SQL: {e}")
            result_df = pd.DataFrame()

    if not result_df.empty:
        answer = get_natural_language_response(
            question=user_question,
            data=result_df.to_string(),
            sql_query=sql_query,
        )
        st.markdown(f"**Answer:** \n{answer}")