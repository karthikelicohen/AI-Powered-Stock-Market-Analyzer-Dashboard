import streamlit as st
import yfinance as yf
import pandas as pd

st.title("Portfolio Analyzer")

stocks = st.text_input(
    "Enter stocks separated by comma",
    "INFY.NS,TCS.NS,RELIANCE.NS"
)

run = st.button("Analyze Portfolio")

if run:

    stock_list = stocks.split(",")

    results = []

    for s in stock_list:

        df = yf.download(s.strip(), period="6mo")

        if df.empty:
            continue

        price = df["Close"].iloc[-1]
        start = df["Close"].iloc[0]

        ret = ((price - start) / start) * 100

        results.append({
            "Stock": s.strip(),
            "Price": round(price,2),
            "Return %": round(ret,2)
        })

    table = pd.DataFrame(results)

    st.write("Portfolio Result")
    st.dataframe(table)

