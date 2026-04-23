import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Page Config
st.set_page_config(page_title="S&P 500 Predictor", layout="wide")

st.title("📈 S&P 500 Market Direction Predictor")

# --- 1. DATA LOADING (With MultiIndex Fix) ---
@st.cache_data
def load_data():
    # Data download
    df = yf.download("^GSPC", period="10y")
    
    # FIX: Agar columns complex hain (MultiIndex), toh unhe simple karein
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Data cleaning
    if "Dividends" in df.columns: del df["Dividends"]
    if "Stock Splits" in df.columns: del df["Stock Splits"]
    
    # Prediction target banana
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
    
    return df.dropna().copy()

# Function call
data = load_data()

# --- 2. MODEL SETUP & TRAINING ---
# Ensure columns are treated as Series
predictors = ["Close", "Volume", "Open", "High", "Low"]
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)

train = data.iloc[:-100]
test = data.iloc[-100:]

model.fit(train[predictors], train["Target"])

# --- 3. LIVE PREDICTION UI ---
st.subheader("Today's Market Analysis")
last_row = data.iloc[-1:]

# Float conversion to avoid further errors
current_price = float(last_row["Close"].iloc[0])

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
    confidence = float(proba[1] if prediction == 1 else proba[0])
    st.write(f"**Model Confidence:** {confidence:.2%}")
    st.progress(confidence)

# --- 4. HISTORICAL PERFORMANCE ---
st.divider()
st.subheader("Historical Model Accuracy")
test_preds = model.predict(test[predictors])
precision = precision_score(test["Target"], test_preds)

st.write(f"Based on the last 100 trading days, this model has a precision of **{precision:.2%}**.")
st.line_chart(data['Close'].tail(100))
