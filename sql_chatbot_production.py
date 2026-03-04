import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.callbacks import StreamlitCallbackHandler
import re
import logging
from typing import Optional, Tuple
from contextlib import contextmanager
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ------------------- LOGGING CONFIG -------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ------------------- CONFIG -------------------
class Config:
    """Configuration management"""
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyD1A3c2ydFBAY2hLNWJV_dQPMYpAv3Io4M")
    DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:XajqWihzsfuATrXRplRmdvtLsjQFMRGx@gondola.proxy.rlwy.net:31883/railway")
    MAX_QUERY_RESULTS = int(os.getenv("MAX_QUERY_RESULTS", "1000"))
    QUERY_TIMEOUT = int(os.getenv("QUERY_TIMEOUT", "30"))
    MODEL_NAME = os.getenv("MODEL_NAME", "models/gemini-2.0-flash-exp")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0"))
    
config = Config()

# ------------------- PAGE CONFIG -------------------
st.set_page_config(
    page_title="SQL Analytics Chatbot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------- CUSTOM CSS -------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

# ------------------- DATABASE CONNECTION -------------------
@st.cache_resource
def get_db_engine():
    """Create and cache database engine with connection pooling"""
    try:
        engine = create_engine(
            config.DB_URL,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before using
            pool_recycle=3600,   # Recycle connections after 1 hour
            echo=False
        )
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database engine created successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to create database engine: {e}")
        st.error(f"Database connection failed: {e}")
        raise

@st.cache_resource
def get_langchain_db():
    """Create LangChain SQLDatabase wrapper"""
    try:
        engine = get_db_engine()
        db = SQLDatabase(engine, sample_rows_in_table_info=3)
        logger.info("LangChain database wrapper created")
        return db
    except Exception as e:
        logger.error(f"Failed to create LangChain database: {e}")
        raise

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    engine = get_db_engine()
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.close()

# ------------------- SCHEMA MANAGEMENT -------------------
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_detailed_schema() -> dict:
    """Get detailed database schema with data types and constraints"""
    engine = get_db_engine()
    schema_info = {}
    
    try:
        inspector = inspect(engine)
        tables = inspector.get_table_names(schema='public')
        
        for table in tables:
            columns = inspector.get_columns(table, schema='public')
            schema_info[table] = {
                'columns': [
                    {
                        'name': col['name'],
                        'type': str(col['type']),
                        'nullable': col['nullable']
                    }
                    for col in columns
                ],
                'primary_keys': inspector.get_pk_constraint(table)['constrained_columns']
            }
        
        logger.info(f"Retrieved schema for {len(tables)} tables")
        return schema_info
    except Exception as e:
        logger.error(f"Error fetching detailed schema: {e}")
        st.error(f"Error fetching schema: {e}")
        return {}

def format_schema_for_llm(schema_info: dict) -> str:
    """Format schema information for LLM prompt"""
    schema_text = "DATABASE SCHEMA:\n\n"
    
    for table_name, table_info in schema_info.items():
        schema_text += f"Table: {table_name}\n"
        if table_info['primary_keys']:
            schema_text += f"  Primary Keys: {', '.join(table_info['primary_keys'])}\n"
        schema_text += "  Columns:\n"
        for col in table_info['columns']:
            nullable = "NULL" if col['nullable'] else "NOT NULL"
            schema_text += f"    - {col['name']} ({col['type']}) {nullable}\n"
        schema_text += "\n"
    
    return schema_text

# ------------------- LLM INITIALIZATION -------------------
@st.cache_resource
def get_llm():
    """Initialize and cache the LLM"""
    try:
        llm = ChatGoogleGenerativeAI(
            model=config.MODEL_NAME,
            google_api_key=config.GOOGLE_API_KEY,
            temperature=config.TEMPERATURE,
            convert_system_message_to_human=True
        )
        logger.info(f"LLM initialized: {config.MODEL_NAME}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        st.error(f"LLM initialization failed: {e}")
        raise

# ------------------- SQL VALIDATION -------------------
class SQLValidator:
    """Validate and sanitize SQL queries"""
    
    FORBIDDEN_KEYWORDS = [
        'DROP', 'DELETE', 'TRUNCATE', 'INSERT', 'UPDATE',
        'ALTER', 'CREATE', 'GRANT', 'REVOKE', 'EXEC'
    ]
    
    @staticmethod
    def clean_sql(sql_text: str) -> str:
        """Clean LLM output"""
        sql_text = re.sub(r"```sql\n?", "", sql_text, flags=re.IGNORECASE)
        sql_text = re.sub(r"```\n?", "", sql_text)
        sql_text = sql_text.strip()
        # Remove trailing semicolons
        sql_text = sql_text.rstrip(';')
        return sql_text
    
    @staticmethod
    def is_safe_query(sql: str) -> Tuple[bool, str]:
        """Check if query is safe to execute"""
        sql_upper = sql.upper()
        
        # Check for forbidden keywords
        for keyword in SQLValidator.FORBIDDEN_KEYWORDS:
            if re.search(rf'\b{keyword}\b', sql_upper):
                return False, f"Query contains forbidden keyword: {keyword}"
        
        # Must start with SELECT
        if not sql_upper.strip().startswith('SELECT'):
            return False, "Only SELECT queries are allowed"
        
        # Check for multiple statements
        if ';' in sql.rstrip(';'):
            return False, "Multiple statements not allowed"
        
        return True, "Query is safe"

# ------------------- LANGCHAIN CHAINS -------------------
class SQLChatbot:
    """Main chatbot class using LangChain"""
    
    def __init__(self):
        self.llm = get_llm()
        self.db = get_langchain_db()
        self.schema_info = get_detailed_schema()
        self.schema_text = format_schema_for_llm(self.schema_info)
        
        # SQL Generation Chain
        self.sql_prompt = PromptTemplate(
            input_variables=["schema", "question"],
            template="""You are an expert PostgreSQL analyst. Generate a SQL query to answer the user's question.

{schema}

IMPORTANT RULES:
1. Return ONLY the SQL query, no explanations
2. Use double quotes for table and column names
3. Use proper PostgreSQL syntax
4. Add LIMIT {max_results} if no limit specified
5. Use appropriate JOINs when needed
6. Handle NULL values appropriately
7. Use aggregate functions when appropriate

User Question: {question}

SQL Query:"""
        )
        
        self.sql_chain = LLMChain(
            llm=self.llm,
            prompt=self.sql_prompt,
            verbose=True
        )
        
        # Response Generation Chain
        self.response_prompt = PromptTemplate(
            input_variables=["question", "query", "result"],
            template="""You are a helpful data analyst. Answer the user's question based on the SQL query results.

User Question: {question}

SQL Query Used: {query}

Query Results:
{result}

Provide a clear, concise answer in natural language. If the results are empty or don't answer the question, say so politely.

Answer:"""
        )
        
        self.response_chain = LLMChain(
            llm=self.llm,
            prompt=self.response_prompt,
            verbose=True
        )
    
    def generate_sql(self, question: str) -> str:
        """Generate SQL query from natural language question"""
        try:
            result = self.sql_chain.run(
                schema=self.schema_text,
                question=question,
                max_results=config.MAX_QUERY_RESULTS
            )
            sql = SQLValidator.clean_sql(result)
            logger.info(f"Generated SQL: {sql[:100]}...")
            return sql
        except Exception as e:
            logger.error(f"Error generating SQL: {e}")
            raise
    
    def execute_query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query safely"""
        # Validate query
        is_safe, message = SQLValidator.is_safe_query(sql)
        if not is_safe:
            raise ValueError(f"Unsafe query: {message}")
        
        try:
            with get_db_connection() as conn:
                # Add timeout
                conn.execute(text(f"SET statement_timeout = {config.QUERY_TIMEOUT * 1000}"))
                df = pd.read_sql(sql, conn)
                logger.info(f"Query executed successfully. Rows returned: {len(df)}")
                return df
        except SQLAlchemyError as e:
            logger.error(f"Database error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def generate_response(self, question: str, query: str, result_df: pd.DataFrame) -> str:
        """Generate natural language response"""
        try:
            # Limit result size for LLM context
            result_text = result_df.head(20).to_string(index=False)
            if len(result_df) > 20:
                result_text += f"\n\n... and {len(result_df) - 20} more rows"
            
            response = self.response_chain.run(
                question=question,
                query=query,
                result=result_text
            )
            return response
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

# ------------------- STREAMLIT UI -------------------
def init_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chatbot' not in st.session_state:
        try:
            st.session_state.chatbot = SQLChatbot()
        except Exception as e:
            st.error(f"Failed to initialize chatbot: {e}")
            st.stop()

def display_sidebar():
    """Display sidebar with schema and settings"""
    with st.sidebar:
        st.header("📋 Database Schema")
        
        schema_info = get_detailed_schema()
        if schema_info:
            for table_name, table_info in schema_info.items():
                with st.expander(f"📊 {table_name}"):
                    st.write("**Columns:**")
                    for col in table_info['columns']:
                        st.text(f"• {col['name']} ({col['type']})")
        
        st.divider()
        st.header("⚙️ Settings")
        st.info(f"Model: {config.MODEL_NAME}")
        st.info(f"Max Results: {config.MAX_QUERY_RESULTS}")
        st.info(f"Query Timeout: {config.QUERY_TIMEOUT}s")
        
        if st.button("🔄 Refresh Schema"):
            st.cache_data.clear()
            st.rerun()
        
        if st.button("🗑️ Clear History"):
            st.session_state.chat_history = []
            st.rerun()

def display_chat_history():
    """Display conversation history"""
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        for i, item in enumerate(st.session_state.chat_history):
            with st.expander(f"Q: {item['question'][:50]}...", expanded=(i == len(st.session_state.chat_history) - 1)):
                st.markdown(f"**Question:** {item['question']}")
                st.code(item['query'], language='sql')
                if not item['result'].empty:
                    st.dataframe(item['result'], use_container_width=True)
                    st.markdown(f"**Answer:** {item['response']}")
                else:
                    st.warning("No results returned")

def main():
    """Main application"""
    init_session_state()
    
    # Header
    st.markdown('<p class="main-header">📊 SQL Analytics Chatbot</p>', unsafe_allow_html=True)
    st.markdown("Ask questions about your database in natural language")
    
    display_sidebar()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_question = st.text_input(
            "Ask a question:",
            placeholder="e.g., What are the top 5 customers by revenue?",
            key="question_input"
        )
    
    with col2:
        st.write("")  # Spacing
        submit_button = st.button("🚀 Get Answer", type="primary", use_container_width=True)
    
    if submit_button and user_question:
        with st.spinner("🤔 Thinking..."):
            try:
                chatbot = st.session_state.chatbot
                
                # Generate SQL
                with st.status("Generating SQL query...", expanded=True) as status:
                    sql_query = chatbot.generate_sql(user_question)
                    st.code(sql_query, language='sql')
                    status.update(label="SQL query generated!", state="complete")
                
                # Execute query
                with st.status("Executing query...", expanded=True) as status:
                    result_df = chatbot.execute_query(sql_query)
                    st.dataframe(result_df, use_container_width=True)
                    status.update(label=f"Query executed! ({len(result_df)} rows)", state="complete")
                
                # Generate response
                if not result_df.empty:
                    with st.status("Generating answer...", expanded=True) as status:
                        answer = chatbot.generate_response(user_question, sql_query, result_df)
                        st.markdown(f"**Answer:** {answer}")
                        status.update(label="Answer generated!", state="complete")
                else:
                    answer = "The query returned no results."
                    st.info(answer)
                
                # Save to history
                st.session_state.chat_history.append({
                    'question': user_question,
                    'query': sql_query,
                    'result': result_df,
                    'response': answer
                })
                
                st.success("✅ Query completed successfully!")
                
            except ValueError as e:
                st.error(f"❌ Validation Error: {e}")
                logger.warning(f"Validation error: {e}")
            except SQLAlchemyError as e:
                st.error(f"❌ Database Error: {e}")
                logger.error(f"Database error: {e}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
                logger.error(f"Unexpected error: {e}", exc_info=True)
    
    # Display history
    if st.session_state.chat_history:
        st.divider()
        display_chat_history()

if __name__ == "__main__":
    main()
