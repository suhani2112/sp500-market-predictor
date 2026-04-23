import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Page configuration
st.set_page_config(page_title="S&P 500 Predictor", page_icon="📈")

st.title("S&P 500 Price Movement Predictor")
st.markdown("""
This app predicts whether the **S&P 500 (^GSPC)** will close higher tomorrow than it did today using a Random Forest model.
""")

# --- 1. DATA LOADING ---
@st.cache_data
def load_data():
    sp500 = yf.Ticker("^GSPC")
    data = sp500.history(period="max")
    # Clean-up logic from your notebook
    if "Dividends" in data.columns: del data["Dividends"]
    if "Stock Splits" in data.columns: del data["Stock Splits"]
    
    # Create target (1 if tomorrow's price is higher than today's)
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    return data.loc["1990-01-01":].copy()

data = load_data()

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