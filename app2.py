import streamlit as st
import os
import warnings
from dotenv import load_dotenv
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.schema import Document
import requests
from bs4 import BeautifulSoup
import time

# Initialize environment variables and suppress warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")
load_dotenv()

# Streamlit app configuration
st.set_page_config(page_title="Stock Price Chatbot")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        color : black;
    }
    .stApp {
        background-color: rgba(0, 0, 0, 0.5);  /* Transparent overlay for contrast */
    }
    .header {
        color: #007BFF;
        font-size: 36px;
        text-align: center;
        margin-top: 20px;
    }
    .response-section {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        color: black;
    }
    .error {
        color: red;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Title with robot emoji
st.markdown('<h1 class="header">Stock Price Chatbot</h1>', unsafe_allow_html=True)

# Fetch stock data
def fetch_stock_data(ticker):
    try:
        url = f'https://www.google.com/finance/quote/{ticker}:NSE'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        class1 = "YMlKec fxKbKc"
        price = float(soup.find(class_=class1).text.strip()[1:].replace(",", ""))
        return f"The current price of {ticker} is ₹{price}."
    except Exception as e:
        return f"Error fetching stock data: {e}"

# Initialize embeddings and vector store
embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
d = len(embeddings.embed_query("test query"))  # Determine embedding dimensionality
index = faiss.IndexFlatL2(d)
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# Define retriever and prompt template
retriever = vector_store
prompt = ChatPromptTemplate.from_template("""
    You are an AI-powered financial assistant specializing in stock market analysis. 
    Your tone should be professional, concise, and easy to understand.

    Question: {question} 
    Context: {context} 
    Answer:
""")

# Define RAG chain
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

model = ChatOllama(model="llama3.2", base_url="http://localhost:11434")
rag_chain = (
    RunnableMap({
        "context": lambda query: retriever.search(query=query, search_type='similarity', k=5),
        "question": RunnablePassthrough()
    })
    | (lambda x: {"context": format_docs(x["context"]), "question": x["question"]})
    | prompt
    | model
    | StrOutputParser()
)

# Input field for stock ticker and questions
ticker = st.text_input("Enter a stock ticker (e.g., M%26M):", key="ticker_input")
question = st.text_input("Ask a question about stocks or the market:", key="question_input")

# Add stock data as a Document object
if ticker:
    # Fetch stock data
    with st.spinner("Fetching stock data..."):
        stock_data = fetch_stock_data(ticker)
        # Create a Document object for the stock data
        stock_document = Document(page_content=stock_data, metadata={"source": f"Stock data for {ticker}"})
        # Add the Document to the vector store
        vector_store.add_documents([stock_document])
        st.write(stock_data)

if question:
    # Process the user's question
    with st.spinner("Processing your query..."):
        try:
            answer = rag_chain.invoke(question)
            st.markdown(f'<div class="response-section">{answer}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="error">Error: {e}</div>', unsafe_allow_html=True)
