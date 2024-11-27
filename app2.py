import streamlit as st
import os
import warnings
import requests
import time
import pandas as pd
import numpy as np
import faiss
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
import yfinance as yf
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.schema import Document

# Load environment variables from .env file
load_dotenv()

# Get Replicate API token from environment variables
replicate_api_token = os.getenv("REPLICATE_API_TOKEN")

# Ensure the API token is available
if not replicate_api_token:
    st.error("Replicate API token not found. Please set the REPLICATE_API_TOKEN in your .env file.")
else:
    headers = {
        "Authorization": f"Bearer {replicate_api_token}"
    }

# Initialize environment variables and suppress warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")

# Streamlit app configuration
st.set_page_config(page_title="Comprehensive Stock Dashboard", layout="wide")

# Custom CSS
st.markdown("""
    <style>
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

# Title
st.markdown('<h1 class="header">Comprehensive Stock Dashboard</h1>', unsafe_allow_html=True)

# Embeddings and vector store setup
embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
d = len(embeddings.embed_query("test query"))
index = faiss.IndexFlatL2(d)
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# RAG Chain Setup
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

prompt = ChatPromptTemplate.from_template("""
    You are an AI-powered financial assistant specializing in stock market analysis. 
    Your tone should be professional, concise, and easy to understand.

    Question: {question} 
    Context: {context} 
    Answer:
""")

model = ChatOllama(model="llama3.2", base_url="http://localhost:11434")
rag_chain = (
    RunnableMap({
        "context": lambda query: vector_store.search(query=query, search_type='similarity', k=5),
        "question": RunnablePassthrough()
    })
    | (lambda x: {"context": format_docs(x["context"]), "question": x["question"]})
    | prompt
    | model
    | StrOutputParser()
)

# Fetch stock price
def fetch_stock_price(ticker, exchange="NSE"):
    try:
        url = f'https://www.google.com/finance/quote/{ticker}:{exchange}'
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        price = float(soup.find(class_="YMlKec fxKbKc").text.strip()[1:].replace(",", ""))
        previous_close = float(soup.find(class_="P6K39c").text.strip()[1:].replace(",", ""))
        revenue = soup.find(class_="QXDnM").text if soup.find(class_="QXDnM") else "N/A"
        news = soup.find(class_="Yfwt5").text if soup.find(class_="Yfwt5") else "No news available"
        about = soup.find(class_="bLLb2d").text if soup.find(class_="bLLb2d") else "No details available"

        return {"Price": price, "Previous Close": previous_close, "Revenue": revenue, "News": news, "About": about}
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Real-time stock tracking (with a limit of 3 updates)
def track_real_time_prices(ticker, exchange="NSE"):
    st.markdown("### Real-Time Stock Prices")
    stock_prices = []
    times = []
    fig, ax = plt.subplots()
    try:
        # Limit the number of iterations to 3 for refreshing the graph
        for timeval in range(3):  # Only 3 updates
            stock_data = fetch_stock_price(ticker, exchange)
            if stock_data:
                stock_prices.append(stock_data["Price"])
                times.append(timeval)

                # Smooth curve
                if len(times) > 3:
                    y_new = np.linspace(min(times), max(times), 300)
                    spl = make_interp_spline(times, stock_prices, k=2)
                    x_smooth = spl(y_new)
                else:
                    y_new = times
                    x_smooth = stock_prices

                ax.clear()
                ax.plot(y_new, x_smooth, label="Stock Price (₹)", color="blue")
                ax.set_title(f"Real-Time Stock Prices for {ticker}")
                ax.set_xlabel("Time Intervals")
                ax.set_ylabel("Price (₹)")
                ax.legend()
                st.pyplot(fig)
            time.sleep(1)  # Delay between each refresh
    except Exception as e:
        st.error(f"Error fetching real-time prices: {e}")


# Historical stock data with moving average
def plot_historical_data(ticker, start='2010-01-01', end='2024-01-01'):
    df = yf.download(ticker, start=start, end=end)
    if not df.empty:
        ma100 = df['Close'].rolling(100).mean()

        plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label="Close Price")
        plt.plot(ma100, 'r', label="100-Day MA")
        plt.title(f"{ticker} Stock Closing Prices")
        plt.xlabel("Days")
        plt.ylabel("Price (USD)")
        plt.legend()
        st.pyplot(plt.gcf())
    else:
        st.error("No historical data found.")

# Main interface
col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("Enter stock ticker (e.g., INFY):")
    exchange = st.text_input("Enter exchange (default: NSE):", value="NSE")
    start_tracking = st.checkbox("Start Real-Time Tracking")
    view_historical = st.checkbox("View Historical Data")

with col2:
    question = st.text_input("Ask a financial question:")

if ticker and start_tracking:
    track_real_time_prices(ticker, exchange)

if ticker and view_historical:
    plot_historical_data(ticker)

if ticker:
    stock_data = fetch_stock_price(ticker, exchange)
    if stock_data:
        st.write(pd.DataFrame(stock_data, index=["Details"]).T)

if question:
    try:
        answer = rag_chain.invoke(question)
        st.markdown(f'<div class="response-section">{answer}</div>', unsafe_allow_html=True)
    except Exception as e:
        st.markdown(f'<div class="error">Error: {e}</div>', unsafe_allow_html=True)
