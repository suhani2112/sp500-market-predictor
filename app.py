import streamlit as st
import yfinance as yf
import pandas as pd
import requests # Naya import

# --- 1. DATA LOADING (Updated for Rate Limit Fix) ---
@st.cache_data
def load_data():
    # Yahoo Finance ko lagna chahiye ki ye ek real browser hai
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Session create karein headers ke sath
    session = requests.Session()
    session.headers.update(headers)
    
    # Ticker ko session pass karein
    sp500 = yf.Ticker("^GSPC", session=session)
    
    # "max" ki jagah "10y" ya "5y" use karein taaki data kam load ho (Rate limit se bachne ke liye)
    data = sp500.history(period="10y") 
    
    if "Dividends" in data.columns: del data["Dividends"]
    if "Stock Splits" in data.columns: del data["Stock Splits"]
    
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    return data.copy()

# --- 2. MODEL SETUP ---
# Predictors established in your notebook
predictors = ["Close", "Volume", "Open", "High", "Low"]
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

# Training on all but the last 100 days for live testing
train = data.iloc[:-100]
test = data.iloc[-100:]
model.fit(train[predictors], train["Target"])

# --- 3. LIVE PREDICTION ---
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

# --- 4. MODEL PERFORMANCE ---
st.divider()
st.subheader("Historical Model Accuracy")
test_preds = model.predict(test[predictors])
precision = precision_score(test["Target"], test_preds)

st.write(f"Based on the last 100 trading days, this model has a precision of **{precision:.2%}**.")
st.line_chart(data['Close'].tail(100))
