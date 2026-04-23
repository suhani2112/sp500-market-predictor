import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Page Config
st.set_page_config(page_title="S&P 500 Predictor", layout="wide")

st.title("📈 S&P 500 Market Direction Predictor")

# --- 1. DATA LOADING FUNCTION ---
@st.cache_data
def load_data():
    # Yahoo Finance rate limit se bachne ke liye headers
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    session = requests.Session()
    session.headers.update(headers)
    
    sp500 = yf.Ticker("^GSPC", session=session)
    # 10 saal ka data kaafi hai prediction ke liye
    df = sp500.history(period="10y")
    
    # Data cleaning
    if "Dividends" in df.columns: del df["Dividends"]
    if "Stock Splits" in df.columns: del df["Stock Splits"]
    
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
    
    return df.dropna().copy()

# Function call karke 'data' variable banana
data = load_data()

# --- 2. MODEL SETUP & TRAINING ---
predictors = ["Close", "Volume", "Open", "High", "Low"]
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

# Training aur Testing data split
train = data.iloc[:-100]
test = data.iloc[-100:]

model.fit(train[predictors], train["Target"])

# --- 3. LIVE PREDICTION UI ---
st.subheader("Today's Market Analysis")
last_row = data.iloc[-1:]
current_price = last_row["Close"].iloc[0]

prediction = model.predict(last_row[predictors])[0]
proba = model.predict_proba(last_row[predictors])[0]

col1, col2 = st.columns(2)

with col1:
    st.metric("Latest Closing Price", f"${current_price:,.2f}")
    if prediction == 1:
        st.success("PREDICTION: PRICE WILL GO UP TOMORROW ⬆️")
    else:
        st.error("PREDICTION: PRICE WILL GO DOWN TOMORROW ⬇️")

with col2:
    confidence = proba[1] if prediction == 1 else proba[0]
    st.write(f"**Model Confidence:** {confidence:.2%}")
    st.progress(confidence)

# --- 4. HISTORICAL PERFORMANCE ---
st.divider()
st.subheader("Historical Model Accuracy")
test_preds = model.predict(test[predictors])
precision = precision_score(test["Target"], test_preds)

st.write(f"Based on the last 100 trading days, this model has a precision of **{precision:.2%}**.")
st.line_chart(data['Close'].tail(100))
