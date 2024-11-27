import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st

st.title("Gold Price Trend Predictor (India Region)")

# Path to the dataset
file_path = "/Users/saiabhiram/Desktop/gold_price_dataset.csv"

# Load Data
st.write("Loading data...")
try:
    # Attempt to load the CSV file
    df = pd.read_csv(file_path)
    st.success("Dataset successfully loaded!")

    # Ensure columns are correctly formatted
    if 'Date' not in df.columns or 'Close' not in df.columns:
        st.error("Dataset must contain 'Date' and 'Close' columns.")
        st.stop()

    # Process the data
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Close']]  # Only use the 'Close' column for training
except FileNotFoundError:
    st.error(f"File not found at: {file_path}. Please check the path and try again.")
    st.stop()
except Exception as e:
    st.error(f"Error reading dataset: {e}")
    st.stop()

# Display Data
st.write("Uploaded Data (First 5 Rows):")
st.write(df.head())

# Preprocessing
st.write("Preprocessing data...")
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

sequence_length = 60
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i - sequence_length:i, 0])
    y.append(scaled_data[i, 0])
X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

train_ratio = st.sidebar.slider("Training Data Ratio (%)", 50, 90, 80)
train_size = int(len(X) * train_ratio / 100)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM Model
st.write("Building LSTM model...")
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# Training
st.write("Training the model...")
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test), verbose=1)

# Predictions
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
actual_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot Results
st.write("Plotting results...")
fig, ax = plt.subplots()
ax.plot(actual_prices, label="Actual Prices", color="blue")
ax.plot(predicted_prices, label="Predicted Prices", color="red")
ax.set_title("Gold Price Prediction")
ax.set_xlabel("Days")
ax.set_ylabel("Price (INR)")
ax.legend()
st.pyplot(fig)

# Predict Future Price
last_60_days = scaled_data[-sequence_length:]
last_60_days = np.reshape(last_60_days, (1, sequence_length, 1))
next_day_price = model.predict(last_60_days)
next_day_price = scaler.inverse_transform(next_day_price)
st.write(f"Predicted price for the next day: ₹{next_day_price[0][0]:.2f}")
