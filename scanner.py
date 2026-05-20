import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide")

st.title("AI STOCK ANALYZER")

tab1, tab2, tab3 = st.tabs(["Stock Analysis", "Stock Scanner", "Portfolio Analyzer"])

with tab1:

    col1, col2 = st.columns(2)

    with col1:
        stock = st.text_input("Stock Symbol", "INFY.NS", key="stock_input")

    with col2:
        period = st.selectbox("Timeframe", ["3mo","6mo","1y","2y","5y"], key="period_select")

    if st.button("Run Analysis", key="analysis_btn"):

        data = yf.download(stock, period=period, progress=False)

        if data.empty:
            st.stop()

        close = data["Close"]

        data["MA20"] = close.rolling(20).mean()
        data["MA50"] = close.rolling(50).mean()

        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()

        rs = avg_gain / avg_loss
        data["RSI"] = 100 - (100/(1+rs))

        data = data.dropna()

        data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

        features = data[["RSI","MA20","MA50","Volume"]]
        target = data["Target"]

        X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.2)

        model = RandomForestClassifier()
        model.fit(X_train,y_train)

        latest = features.iloc[-1:].values

        prob = model.predict_proba(latest)

        confidence = round(max(prob[0])*100,2)

        if prob[0][1] > 0.6:
            signal = "BUY"
        elif prob[0][0] > 0.6:
            signal = "SELL"
        else:
            signal = "HOLD"

        c1, c2 = st.columns(2)

        c1.metric("Signal", signal)
        c2.metric("Confidence", str(confidence)+"%")

        support = round(data["Low"].tail(20).min(),2)
        resistance = round(data["High"].tail(20).max(),2)

        data["Crossover"] = data["MA20"] - data["MA50"]
        data["Signal"] = np.sign(data["Crossover"])
        data["SignalChange"] = data["Signal"].diff()

        buy_points = data[data["SignalChange"] == 2]
        sell_points = data[data["SignalChange"] == -2]

        fig = make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[0.6,0.2,0.2])

        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                increasing_line_color="green",
                decreasing_line_color="red"
            ),
            row=1,col=1
        )

        fig.add_trace(go.Scatter(x=data.index,y=data["MA20"],line=dict(color="orange",width=3)),row=1,col=1)
        fig.add_trace(go.Scatter(x=data.index,y=data["MA50"],line=dict(color="blue",width=3)),row=1,col=1)

        fig.add_trace(go.Scatter(x=buy_points.index,y=buy_points["Close"],mode="markers",
                                 marker=dict(symbol="triangle-up",color="lime",size=18)),row=1,col=1)

        fig.add_trace(go.Scatter(x=sell_points.index,y=sell_points["Close"],mode="markers",
                                 marker=dict(symbol="triangle-down",color="red",size=18)),row=1,col=1)

        fig.add_hline(y=support,line_color="lime",line_dash="dash",row=1,col=1)
        fig.add_hline(y=resistance,line_color="red",line_dash="dash",row=1,col=1)

        colors = np.where(data["Close"]>data["Open"],"green","red")

        fig.add_trace(go.Bar(x=data.index,y=data["Volume"],marker_color=colors),row=2,col=1)
        fig.add_trace(go.Scatter(x=data.index,y=data["RSI"],line=dict(color="purple")),row=3,col=1)

        fig.update_layout(template="plotly_dark",height=850,xaxis_rangeslider_visible=False)

        st.plotly_chart(fig,use_container_width=True)


with tab2:

    st.subheader("Real Time Stock Scanner")

    stocks = ["RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS","SBIN.NS","ITC.NS"]

    if st.button("Run Scanner", key="scanner_btn"):

        results = []

        for s in stocks:

            df = yf.download(s,period="3mo",progress=False)

            if not df.empty:

                close = df["Close"]

                ma20 = close.rolling(20).mean()
                ma50 = close.rolling(50).mean()

                delta = close.diff()

                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)

                avg_gain = gain.rolling(14).mean()
                avg_loss = loss.rolling(14).mean()

                rs = avg_gain/avg_loss
                rsi = 100 - (100/(1+rs))

                trend = "Bullish" if ma20.iloc[-1] > ma50.iloc[-1] else "Bearish"

                results.append({
                    "Stock": s,
                    "Price": round(close.iloc[-1],2),
                    "Trend": trend,
                    "RSI": round(rsi.iloc[-1],2)
                })

        st.dataframe(pd.DataFrame(results))


with tab3:

    st.subheader("Portfolio Analyzer")

    portfolio = st.text_input("Enter stocks separated by comma",
                              "INFY.NS,TCS.NS,RELIANCE.NS",
                              key="portfolio_input")

    if st.button("Analyze Portfolio", key="portfolio_btn"):

        stocks_list = portfolio.split(",")

        results = []

        for s in stocks_list:

            df = yf.download(s.strip(),period="6mo",progress=False)

            if not df.empty:

                price = df["Close"].iloc[-1]
                start_price = df["Close"].iloc[0]

                ret = ((price-start_price)/start_price)*100

                ma20 = df["Close"].rolling(20).mean()
                ma50 = df["Close"].rolling(50).mean()

                trend = "Bullish" if ma20.iloc[-1] > ma50.iloc[-1] else "Bearish"

                volatility = df["Close"].pct_change().std()

                risk = "Low"

                if volatility > 0.02:
                    risk = "High"
                elif volatility > 0.01:
                    risk = "Medium"

                results.append({
                    "Stock": s,
                    "Price": round(price,2),
                    "Return %": round(ret,2),
                    "Trend": trend,
                    "Risk": risk
                })

        st.dataframe(pd.DataFrame(results))

