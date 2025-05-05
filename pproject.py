import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Function to fetch stock data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Function to preprocess data using all previous data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices = data['Close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(close_prices)
    
    X, y = [], []
    for i in range(1, len(scaled_data)):
        X.append(scaled_data[:i, 0])
        y.append(scaled_data[i, 0])
    
    max_length = len(X[-1])
    X_padded = np.array([np.pad(seq, (max_length - len(seq), 0), 'constant') for seq in X])
    y = np.array(y)
    
    X_padded = X_padded.reshape((X_padded.shape[0], X_padded.shape[1], 1))
    
    train_size = int(len(X_padded) * 0.8)
    X_train, X_test = X_padded[:train_size], X_padded[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    return X_train, X_test, y_train, y_test, scaler

# Function to build and train LSTM model
def train_lstm_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=False, input_shape=(None, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    return model

# Function to extract LSTM features
def extract_lstm_features(model, X):
    lstm_features = model.predict(X, verbose=0)
    return lstm_features

# Function to train SVM on LSTM features
def train_svm_model(lstm_features_train, y_train):
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    svr.fit(lstm_features_train, y_train)
    return svr

# Function to predict future prices
def predict_future_prices(lstm_model, svm_model, last_sequence, scaler, days_to_predict):
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days_to_predict):
        lstm_feature = lstm_model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)
        pred = svm_model.predict(lstm_feature)[0]
        future_predictions.append(pred)
        current_sequence = np.append(current_sequence, pred)
        current_sequence = current_sequence.reshape(-1, 1)
    
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    return future_predictions

# Function to calculate MAPE
def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Function to generate basic suggestions
def generate_basic_suggestions(predictions):
    first_price = float(predictions[0][0])
    last_price = float(predictions[-1][0])
    trend = "Upward" if last_price > first_price else "Downward"
    suggestion = f"The predicted trend is {trend}. "
    if trend == "Upward":
        suggestion += "Consider buying or holding the stock."
    else:
        suggestion += "Consider selling or avoiding the stock."
    return suggestion

# Function for detailed suggestions with error handling
def generate_detailed_suggestions(predictions, historical_data, period):
    if not isinstance(historical_data, pd.DataFrame) or historical_data.empty:
        return "Error: Historical data is not available or invalid"
    
    if 'Close' not in historical_data.columns:
        available_columns = ', '.join(historical_data.columns)
        return f"Error: 'Close' column not found in historical data. Available columns: {available_columns}"
    
    latest_price = float(historical_data['Close'].iloc[-1])
    predicted_max = float(np.max(predictions))
    predicted_min = float(np.min(predictions))
    predicted_end = float(predictions[-1][0])
    
    last_30_days = historical_data['Close'].tail(30)
    volatility = float(np.std(last_30_days) / latest_price * 100 if len(last_30_days) >= 30 else 0)
    
    suggestions = []
    
    price_change = float((predicted_end - latest_price) / latest_price * 100)
    if price_change > 5:
        suggestions.append("Strong bullish trend detected. Consider a long position.")
    elif price_change < -5:
        suggestions.append("Strong bearish trend detected. Consider selling or shorting.")
    else:
        suggestions.append("Neutral trend. Monitor the stock closely.")
    
    if volatility > 10:
        suggestions.append(f"High volatility ({volatility:.1f}%). Higher risk - use tighter stop-losses.")
    elif volatility < 3:
        suggestions.append(f"Low volatility ({volatility:.1f}%). More stable but potentially slower returns.")
    else:
        suggestions.append(f"Moderate volatility ({volatility:.1f}%). Balanced risk-reward potential.")
    
    potential_gain = float((predicted_max - latest_price) / latest_price * 100)
    potential_loss = float((latest_price - predicted_min) / latest_price * 100)
    suggestions.append(f"Potential upside: {potential_gain:.1f}% | Potential downside: {potential_loss:.1f}%")
    
    period_days = {"Tomorrow": 1, "1 Week": 7, "1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365}
    if period in ["Tomorrow", "1 Week"]:
        suggestions.append("Short-term prediction: Consider day trading or swing trading strategies.")
    elif period in ["1 Month", "3 Months"]:
        suggestions.append("Medium-term prediction: Suitable for position trading.")
    else:
        suggestions.append("Long-term prediction: Consider for investment portfolio allocation.")
    
    return "\n".join(suggestions)

# Streamlit app
st.title("Hybrid SVM-LSTM Stock Price Prediction")
st.write("Select a stock and time period to predict future prices using a hybrid SVM and LSTM model!")

stock_options = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS",
    "ITC.NS", "ASIANPAINT.NS", "AXISBANK.NS", "DMART.NS", "BAJFINANCE.NS",
    "SUNPHARMA.NS", "MARUTI.NS", "NTPC.NS", "ULTRACEMCO.NS", "WIPRO.NS"
]
ticker = st.selectbox("Select Stock Ticker", stock_options)

time_periods = {
    "Tomorrow": 1,
    "1 Week": 7,
    "1 Month": 30,
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365
}
selected_period = st.selectbox("Select Prediction Period", list(time_periods.keys()))
days_to_predict = time_periods[selected_period]

if st.button("Predict"):
    end_date = datetime(2025, 4, 9)
    start_date = end_date - timedelta(days=365)
    
    with st.spinner("Fetching and processing data..."):
        data = fetch_stock_data(ticker, start_date, end_date)
        if data.empty:
            st.error("Invalid ticker or no data available!")
        else:
            X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
            
            lstm_model = train_lstm_model(X_train, y_train)
            lstm_features_train = extract_lstm_features(lstm_model, X_train)
            lstm_features_test = extract_lstm_features(lstm_model, X_test)
            svm_model = train_svm_model(lstm_features_train, y_train)
            
            y_pred = svm_model.predict(lstm_features_test)
            y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))
            y_pred_scaled = scaler.inverse_transform(y_pred.reshape(-1, 1))
            
            # Calculate MAPE instead of MSE
            mape = calculate_mape(y_test_scaled, y_pred_scaled)
            
            last_sequence = X_test[-1]
            future_prices = predict_future_prices(lstm_model, svm_model, last_sequence, scaler, days_to_predict)
            
            st.subheader("Model Performance")
            st.write(f"Mean Absolute Percentage Error (MAPE) on Test Data: {mape:.2f}%")
            st.write("Note: Lower MAPE indicates better accuracy (e.g., 5% means predictions are within 5% of actual values)")
            
            st.subheader(f"Price Predictions for {selected_period}")
            future_dates = [end_date + timedelta(days=i+1) for i in range(days_to_predict)]
            future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_prices.flatten()})
            st.dataframe(future_df)
            
            fig, ax = plt.subplots()
            ax.plot(data.index[-50:], data['Close'][-50:], label="Historical Prices")
            ax.plot(future_dates, future_prices, label="Predicted Prices", linestyle="--")
            ax.legend()
            ax.set_title(f"{ticker} Stock Price Prediction (SVM-LSTM)")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            st.subheader("Basic Investment Suggestions")
            basic_suggestion = generate_basic_suggestions(future_prices)
            st.write(basic_suggestion)
            
            st.subheader("Detailed Investment Suggestions")
            detailed_suggestion = generate_detailed_suggestions(future_prices, data, selected_period)
            st.write(detailed_suggestion)

st.write("Note: Predictions are based on historical data using a hybrid SVM-LSTM model. Actual market performance may vary.")