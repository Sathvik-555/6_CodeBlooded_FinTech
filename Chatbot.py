import streamlit as st
from transformers import pipeline
import yfinance as yf
from textblob import TextBlob
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


st.title("AI-Powered Finance Assistant")


st.sidebar.header("User Inputs")
st.sidebar.write("Provide inputs for personalized financial advice.")


st.write("## Finance Query Answering")
st.write("Ask the AI bot any finance-related question, like 'What is asset allocation?' or 'Explain mutual funds.'")


@st.cache_resource
def load_model():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")


qa_model = load_model()

query = st.text_input("Enter your finance-related query:")
if query:
    context = """
    Asset allocation refers to the strategy of distributing investments across various asset classes like stocks, bonds, and cash.
    Mutual funds are investment vehicles that pool money from many investors to invest in stocks, bonds, or other assets.
    Risk tolerance is the degree of variability in investment returns that an individual is willing to withstand.
    Bonds are fixed-income investments where investors lend money to an entity in exchange for periodic interest payments.
    Gold is a precious metal often used as a hedge against economic uncertainty.
    Cryptocurrency is a digital or virtual currency secured by cryptography, often considered highly volatile.
    """
    answer = qa_model(question=query, context=context)
    st.write("**Answer:**", answer['answer'])


st.write("## Asset Allocation Advice")
st.write("Get personalized advice on splitting your money across different asset classes.")

investment_amount = st.number_input("Enter your total investment amount (₹):", min_value=1000, step=1000, value=50000)
risk_tolerance = st.radio(
    "Select your risk tolerance level:",
    ("Low", "Moderate", "High")
)

preferences = st.multiselect(
    "Select preferred asset classes:",
    ["Stocks", "Bonds", "Real Estate", "Gold", "Mutual Funds", "Cryptocurrency"],
    default=["Stocks", "Bonds"]
)

st.write("### Suggested Asset Allocation")
if risk_tolerance == "Low":
    weights = {"Stocks": 10, "Bonds": 50, "Real Estate": 20, "Gold": 15, "Mutual Funds": 5, "Cryptocurrency": 0}
elif risk_tolerance == "Moderate":
    weights = {"Stocks": 30, "Bonds": 30, "Real Estate": 20, "Gold": 10, "Mutual Funds": 10, "Cryptocurrency": 0}
else:
    weights = {"Stocks": 50, "Bonds": 10, "Real Estate": 10, "Gold": 5, "Mutual Funds": 15, "Cryptocurrency": 10}

filtered_weights = {asset: weights[asset] for asset in preferences if asset in weights}

total_weight = sum(filtered_weights.values())
normalized_weights = {asset: (weight / total_weight) for asset, weight in filtered_weights.items()}

allocation = {asset: investment_amount * weight for asset, weight in normalized_weights.items()}

st.write("Based on your inputs, here’s the recommended asset allocation:")
for asset, amount in allocation.items():
    st.write(f"- **{asset}:** ₹{amount:,.2f}")

st.write(
    "**Note:** The above allocation is a general suggestion based on your input. Please consult a financial advisor for personalized advice.")


st.write("## Real-Time Market Sentiment Analysis")
st.write("Analyze real-time sentiment from news headlines for Stocks, Gold, and Real Estate.")

news_api_key = "pub_60597507c72081d03d2bd07b8a8410e3e0c5f"
tickers = ["Stocks", "Gold", "Real Estate"]
sentiments = {}

for ticker in tickers:
    st.write(f"### {ticker} Sentiment Analysis")
    news_url = f"https://newsdata.io/api/1/news?apikey={news_api_key}&q={ticker}&language=en"
    response = requests.get(news_url)
    if response.status_code == 200:
        articles = response.json().get('results', [])
        headlines = [article['title'] for article in articles]
        st.write("Recent Headlines:")
        for headline in headlines[:5]:
            st.write("-", headline)

        sentiment_scores = [TextBlob(headline).sentiment.polarity for headline in headlines]
        avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
        sentiments[ticker] = avg_sentiment
        st.write(
            f"**Average Sentiment:** {'Positive' if avg_sentiment > 0 else 'Negative' if avg_sentiment < 0 else 'Neutral'} ({avg_sentiment:.2f})")
    else:
        st.warning(f"Failed to fetch news for {ticker}.")
        sentiments[ticker] = 0


st.write("### Sentiment-Based Asset Allocation Insights")
sentiment_allocation = {ticker: max(0, 50 + (sentiments[ticker] * 50)) for ticker in tickers}
total_sentiment_weight = sum(sentiment_allocation.values())
sentiment_allocation_normalized = {k: v / total_sentiment_weight for k, v in sentiment_allocation.items()}

fig, ax = plt.subplots()
ax.pie(
    sentiment_allocation_normalized.values(),
    labels=sentiment_allocation_normalized.keys(),
    autopct='%1.1f%%',
    startangle=140
)
ax.set_title("Sentiment-Based Allocation")
st.pyplot(fig)


st.write("## Real-Time Stock Analysis")
stock_ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", value="AAPL")

if stock_ticker:
    st.write(f"Fetching real-time stock value for **{stock_ticker}**...")
    try:
        ticker_data = yf.Ticker(stock_ticker)
        real_time_data = ticker_data.history(period="1d")
        if not real_time_data.empty:
            current_price = real_time_data["Close"].iloc[-1]
            st.write(f"### Current real-time stock value of {stock_ticker}: **${current_price:.2f}**")
        else:
            st.warning("Could not fetch real-time stock data.")
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")

    st.write(f"Fetching historical data for {stock_ticker}...")
    data = yf.download(stock_ticker, start="2020-01-01", end=pd.Timestamp.today())
    if data.empty:
        st.error("No data found. Please check the ticker or date range.")
    else:
        st.write("### Stock Price Trend")
        fig, ax = plt.subplots()
        ax.plot(data.index, data["Close"], label="Close Price", color="blue")
        ax.set_title(f"{stock_ticker} Stock Price Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)
