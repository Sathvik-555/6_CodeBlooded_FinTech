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
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.schema import Document

# Initialize environment variables and suppress warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")
load_dotenv()

# Streamlit app configuration
st.set_page_config(page_title="Comprehensive Stock and Gold Dashboard", layout="wide")

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
st.markdown('<h1 class="header">Comprehensive Stock and Gold Price Dashboard</h1>', unsafe_allow_html=True)

# Embeddings and vector store setup for RAG Chain (Optional, can be used later for more features)
embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
d = len(embeddings.embed_query("test query"))
index = faiss.IndexFlatL2(d)
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# Stock Functions
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

# Real-time stock tracking
def track_real_time_prices(ticker, exchange="NSE"):
    st.markdown("### Real-Time Stock Prices")
    stock_prices = []
    times = []
    fig, ax = plt.subplots()
    
    # Set the background to black
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    # Customize the axes to have white color
    ax.spines['bottom'].set_color('white')  # X-axis
    ax.spines['left'].set_color('white')    # Y-axis
    ax.spines['top'].set_color('none')      # Hide the top spine
    ax.spines['right'].set_color('none')    # Hide the right spine
    
    # Set the tick parameters (axes and labels in white)
    ax.tick_params(axis='x', colors='white')  # X-axis ticks in white
    ax.tick_params(axis='y', colors='white')  # Y-axis ticks in white
    
    # Set the title and labels in white
    ax.set_title(f"Real-Time Stock Prices for {ticker}", color='white')
    ax.set_xlabel("Time Intervals", color='white')
    ax.set_ylabel("Price (₹)", color='white')
    
    # Add legend with white text
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
    
    # Initialize line plot with red color for stock prices
    line, = ax.plot([], [], label="Stock Price (₹)", color='red')

    try:
        for timeval in range(3):  # Limit to 3 updates for demo
            stock_data = fetch_stock_price(ticker, exchange)
            if stock_data:
                stock_prices.append(stock_data["Price"])
                times.append(timeval)

                if len(times) > 3:
                    y_new = np.linspace(min(times), max(times), 300)
                    spl = make_interp_spline(times, stock_prices, k=2)
                    x_smooth = spl(y_new)
                else:
                    y_new = times
                    x_smooth = stock_prices

                # Update plot with smoothed stock price line in red
                line.set_xdata(y_new)
                line.set_ydata(x_smooth)

                # Redraw the plot on Streamlit
                ax.relim()  # Recompute limits
                ax.autoscale_view()  # Rescale view
                st.pyplot(fig)  # Display updated plot

                # Clear figure to avoid overlap in updates
                fig.clf()
                fig, ax = plt.subplots()
                ax.set_facecolor('black')  # Reset background to black
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                ax.set_title(f"Real-Time Stock Prices for {ticker}", color='white')
                ax.set_xlabel("Time Intervals", color='white')
                ax.set_ylabel("Price (₹)", color='white')
                ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
                line, = ax.plot([], [], label="Stock Price (₹)", color='red')

            time.sleep(1)  # Delay between updates
    except Exception as e:
        st.error(f"Error fetching real-time prices: {e}")

# Historical stock data with moving average
def plot_historical_data(ticker, start='2010-01-01', end='2024-01-01'):
    df = yf.download(ticker, start=start, end=end)
    if not df.empty:
        ma100 = df['Close'].rolling(100).mean()

        fig, ax = plt.subplots(figsize=(12, 6))

        # Set the background to black
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')

        # Customize the axes to have white color
        ax.spines['bottom'].set_color('white')  # X-axis
        ax.spines['left'].set_color('white')    # Y-axis
        ax.spines['top'].set_color('none')      # Hide the top spine
        ax.spines['right'].set_color('none')    # Hide the right spine

        # Set the tick parameters (axes and labels in white)
        ax.tick_params(axis='x', colors='white')  # X-axis ticks in white
        ax.tick_params(axis='y', colors='white')  # Y-axis ticks in white

        # Set the title and labels in white
        ax.set_title(f"{ticker} Stock Closing Prices", color='white')
        ax.set_xlabel("Days", color='white')
        ax.set_ylabel("Price (USD)", color='white')

        # Plot the stock prices and 100-day moving average in red
        ax.plot(df['Close'], label="Close Price", color='red')
        ax.plot(ma100, 'r', label="100-Day MA")

        ax.legend(facecolor='black', edgecolor='white', labelcolor='white')

        st.pyplot(fig)
    else:
        st.error("Error fetching historical stock data")

# LSTM stock prediction
def predict_stock_price(ticker):
    st.markdown("### Stock Price Prediction Using LSTM")
    df = yf.download(ticker, period='5y', interval='1d')
    df = df[['Close']]
    df = df.dropna()

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    # Create the dataset
    train_size = int(len(df) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]

    def create_dataset(data, time_step=60):
        x_data, y_data = [], []
        for i in range(len(data) - time_step - 1):
            x_data.append(data[i:i + time_step])
            y_data.append(data[i + time_step])
        return np.array(x_data), np.array(y_data)

    x_train, y_train = create_dataset(train_data)
    x_test, y_test = create_dataset(test_data)

    # Reshape input for LSTM
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32)

    # Predict the stock price
    predicted_price = model.predict(x_test)

    # Rescale predictions
    predicted_price = scaler.inverse_transform(predicted_price)

    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(df['Close'].values[-len(test_data):], color='blue', label='Real Stock Price')
    plt.plot(predicted_price, color='red', label='Predicted Stock Price')
    plt.title(f'{ticker} Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Stock Price (USD)')
    plt.legend()
    st.pyplot()
