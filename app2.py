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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import yfinance as yf
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from textblob import TextBlob
import streamlit as st

# Initialize environment variables and suppress warnings
load_dotenv()
warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Streamlit app configuration
st.set_page_config(page_title="Integrated Stock Dashboard", layout="wide")

# Custom CSS for styling
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
st.markdown('<h1 class="header">Integrated Stock Dashboard</h1>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["Dashboard", "Real-Time Tracking", "Historical Data", "Sentiment Analysis", "Prediction"])

# Function to fetch real-time stock price
def fetch_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if not data.empty:
            return {"Price": round(data["Close"].iloc[-1], 2)}
        else:
            raise ValueError("No price data available for the ticker.")
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# Real-time tracking
def track_real_time_prices(ticker):
    st.markdown("### Real-Time Stock Prices")
    stock_prices = []
    times = []

    for timeval in range(3):  # Limit to 3 updates for demo
        stock_data = fetch_stock_price(ticker)
        if stock_data:
            stock_prices.append(stock_data["Price"])
            times.append(timeval)

            plt.figure(figsize=(10, 5))
            plt.plot(times, stock_prices, marker="o", color="red")
            plt.title(f"Real-Time Stock Prices for {ticker}", fontsize=16)
            plt.xlabel("Time Interval", fontsize=12)
            plt.ylabel("Price ($)", fontsize=12)
            st.pyplot()
        time.sleep(1)

# Historical data visualization
def plot_historical_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        if not df.empty:
            ma100 = df['Close'].rolling(100).mean()

            plt.figure(figsize=(12, 6))
            plt.plot(df['Close'], label="Close Price", color='blue')
            plt.plot(ma100, label="100-Day Moving Average", color='red')
            plt.title(f"{ticker} Stock Prices")
            plt.xlabel("Date")
            plt.ylabel("Price ($)")
            plt.legend()
            st.pyplot()
        else:
            st.error("Error: No historical data available.")
    except Exception as e:
        st.error(f"Error fetching historical stock data: {e}")

# Sentiment analysis of recent news
def fetch_news_and_analyze_sentiment(ticker):
    news_api_key = os.getenv("NEWS_API_KEY", "pub_60597507c72081d03d2bd07b8a8410e3e0c5f")
    news_url = f"https://newsdata.io/api/1/news?apikey={news_api_key}&q={ticker}&language=en"

    st.markdown(f"### Recent News for {ticker}")
    try:
        response = requests.get(news_url)
        news_data = response.json()
        articles = news_data.get("results", [])
        
        if articles:
            sentiments = []
            for article in articles[:5]:  # Limit to 5 articles for simplicity
                title = article.get("title", "")
                description = article.get("description", "")
                content = f"{title}. {description}"
                sentiment = TextBlob(content).sentiment.polarity
                sentiments.append(sentiment)
                st.write(f"**{title}**")
                st.write(f"Sentiment Polarity: {sentiment:.2f}")

            avg_sentiment = np.mean(sentiments)
            st.write(f"### Average Sentiment Polarity: **{avg_sentiment:.2f}**")
        else:
            st.warning("No news articles found.")
    except Exception as e:
        st.error(f"Error fetching news data: {e}")

# Prediction using LSTM
def predict_stock_prices(ticker, start_date, end_date, train_ratio):
    st.write("Fetching historical data for prediction...")
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        st.error("No data found. Please check the ticker or date range.")
        return

    df = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    sequence_length = 60
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    train_size = int(len(X) * train_ratio / 100)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    st.write("Training the model...")
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    st.write("Generating predictions...")
    predicted_prices = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)
    actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

    st.write("### Predicted vs Actual Prices")
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, label="Actual Prices", color="blue")
    plt.plot(predicted_prices, label="Predicted Prices", color="red")
    plt.title(f"{ticker} Stock Price Prediction")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot()

# Page logic
if page == "Dashboard":
    st.markdown("### Welcome to the Stock and Financial Dashboard!")
elif page == "Real-Time Tracking":
    ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
    track_real_time_prices(ticker)
elif page == "Historical Data":
    ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2024-01-01"))
    plot_historical_data(ticker, start=start_date, end=end_date)
elif page == "Sentiment Analysis":
    ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
    fetch_news_and_analyze_sentiment(ticker)
elif page == "Prediction":
    ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2015-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-01-01"))
    train_ratio = st.sidebar.slider("Training Data Ratio (%)", 50, 90, 80)
    predict_stock_prices(ticker, start_date, end_date, train_ratio)
