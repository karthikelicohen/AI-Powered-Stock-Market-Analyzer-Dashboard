import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import feedparser
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from streamlit_autorefresh import st_autorefresh



st.set_page_config(
    page_title="AI Stock Analyzer Pro",
    layout="wide"
)



st_autorefresh(
    interval=60000,
    key="refresh"
)



st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.stMetric {
    background-color: #1E1E1E;
    padding: 15px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)



st.title("📈 AI STOCK ANALYZER PRO")



st.sidebar.title("📊 Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Market",
        "Gainers/Losers",
        "Analyzer",
        "Scanner",
        "Breakouts",
        "Portfolio",
        "Heatmap",
        "Sector AI",
        "News AI",
        "AI Prediction"
    ]
)



stocks = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "AXISBANK": "AXISBANK.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "HCLTECH": "HCLTECH.NS",
    "WIPRO": "WIPRO.NS",
    "TECHM": "TECHM.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "CIPLA": "CIPLA.NS",
    "DRREDDY": "DRREDDY.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "JSWSTEEL": "JSWSTEEL.NS"
}



search_stock = st.sidebar.text_input(
    "🔍 Search Stock"
)

if search_stock:

    filtered = {
        k: v for k, v in stocks.items()
        if search_stock.upper() in k
    }

else:

    filtered = stocks


@st.cache_data(ttl=3600)

def safe_download(symbol, period="1y"):

    try:

        df = yf.download(
            symbol,
            period=period,
            progress=False,
            auto_adjust=False
        )

        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[[
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]]

        df = df.astype(float)

        return df

    except Exception as e:

        st.error(f"{symbol} Error: {e}")

        return None


all_data = {}

with st.spinner("Loading Market Data..."):

    for name, symbol in filtered.items():

        all_data[name] = safe_download(symbol)


def draw_chart(df, name):

    close = df["Close"]

    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()

    support = float(df["Low"].tail(20).min())
    resistance = float(df["High"].tail(20).max())

    df["signal"] = np.where(
        df["MA20"] > df["MA50"],
        1,
        0
    )

    df["pos"] = df["signal"].diff()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2]
    )

    

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name=name
    ), row=1, col=1)

   

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["MA20"],
        name="MA20"
    ), row=1, col=1)

   

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["MA50"],
        name="MA50"
    ), row=1, col=1)

   

    buy = df[df["pos"] == 1]
    sell = df[df["pos"] == -1]

    fig.add_trace(go.Scatter(
        x=buy.index,
        y=buy["Close"],
        mode="markers",
        marker=dict(
            color="green",
            size=10
        ),
        name="BUY"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sell.index,
        y=sell["Close"],
        mode="markers",
        marker=dict(
            color="red",
            size=10
        ),
        name="SELL"
    ), row=1, col=1)

    

    fig.add_hline(
        y=support,
        line_color="green"
    )

    fig.add_hline(
        y=resistance,
        line_color="red"
    )

    

    fig.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"],
        name="Volume"
    ), row=2, col=1)

    

    delta = close.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    rs = gain.rolling(14).mean() / (
        loss.rolling(14).mean() + 1e-10
    )

    rsi = 100 - (100 / (1 + rs))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=rsi,
        name="RSI"
    ), row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=850,
        xaxis_rangeslider_visible=False
    )

    return fig



if page == "Market":

    st.header("📊 Market Overview")

    st.success("Live Market Connected")

    st.write(
        f"Total Stocks Loaded: {len(filtered)}"
    )


elif page == "Gainers/Losers":

    movers = []

    for s in filtered:

        df = all_data[s]

        if df is None or len(df) < 2:
            continue

        c1 = float(df["Close"].iloc[-1])

        c2 = float(df["Close"].iloc[-2])

        change = ((c1 - c2) / c2) * 100

        movers.append({
            "Stock": s,
            "Change %": round(change, 2)
        })

    movers = pd.DataFrame(movers)

    col1, col2 = st.columns(2)

    col1.subheader("🚀 Top Gainers")

    col1.dataframe(
        movers.sort_values(
            "Change %",
            ascending=False
        ).head()
    )

    col2.subheader("📉 Top Losers")

    col2.dataframe(
        movers.sort_values(
            "Change %"
        ).head()
    )



elif page == "Analyzer":

    stock = st.selectbox(
        "Select Stock",
        list(filtered.keys())
    )

    period = st.selectbox(
        "Select Time Period",
        ["1mo", "3mo", "6mo", "1y", "5y"]
    )

    if st.button("Run Analysis"):

        symbol = filtered[stock]

        df = safe_download(symbol, period)

        if df is not None:

            current_price = float(
                df["Close"].iloc[-1]
            )

            prev_price = float(
                df["Close"].iloc[-2]
            )

            change = (
                (current_price - prev_price)
                / prev_price
            ) * 100

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Current Price",
                round(current_price, 2)
            )

            col2.metric(
                "Day Change %",
                round(change, 2)
            )

            col3.metric(
                "Volume",
                int(df["Volume"].iloc[-1])
            )

            st.plotly_chart(
                draw_chart(df.copy(), stock),
                use_container_width=True
            )

            csv = df.to_csv().encode()

            st.download_button(
                "⬇ Download CSV",
                csv,
                f"{stock}.csv",
                "text/csv"
            )



elif page == "Scanner":

    if st.button("Run Scanner"):

        results = []

        for s in filtered:

            df = all_data[s]

            if df is None:
                continue

            ma20 = df["Close"].rolling(20).mean()

            ma50 = df["Close"].rolling(50).mean()

            trend = (
                "Bullish"
                if float(ma20.iloc[-1])
                > float(ma50.iloc[-1])
                else "Bearish"
            )

            results.append({
                "Stock": s,
                "Price": round(
                    float(df["Close"].iloc[-1]),
                    2
                ),
                "Trend": trend
            })

        st.dataframe(
            pd.DataFrame(results)
        )


elif page == "Breakouts":

    if st.button("Find Breakouts"):

        res = []

        for s in filtered:

            df = all_data[s]

            if df is None:
                continue

            price = float(
                df["Close"].iloc[-1]
            )

            high = float(
                df["High"].tail(20).max()
            )

            if price >= high:

                res.append({
                    "Stock": s,
                    "Breakout": round(price, 2)
                })

        st.dataframe(
            pd.DataFrame(res)
        )



elif page == "Portfolio":

    portfolio = st.text_area(
        "Enter Portfolio",
        "INFY,TCS"
    )

    buy_price = st.number_input(
        "Buy Price",
        value=100.0
    )

    quantity = st.number_input(
        "Quantity",
        value=1
    )

    if st.button("Analyze Portfolio"):

        res = []

        for s in portfolio.split(","):

            s = s.strip().upper()

            if s not in all_data:
                continue

            df = all_data[s]

            if df is None:
                continue

            current = float(
                df["Close"].iloc[-1]
            )

            pnl = (
                current - buy_price
            ) * quantity

            res.append({
                "Stock": s,
                "Current Price": round(current, 2),
                "Profit/Loss": round(pnl, 2)
            })

        st.dataframe(
            pd.DataFrame(res)
        )



elif page == "Heatmap":

    heat = []

    for s in filtered:

        df = all_data[s]

        if df is None:
            continue

        start = float(
            df["Close"].iloc[0]
        )

        end = float(
            df["Close"].iloc[-1]
        )

        change = (
            (end - start)
            / start
        ) * 100

        heat.append({
            "Stock": s,
            "Performance %": round(change, 2)
        })

    heat = pd.DataFrame(heat)

    st.dataframe(
        heat.style.background_gradient(
            cmap="RdYlGn"
        )
    )



elif page == "Sector AI":

    sectors = {

        "IT": [
            "TCS",
            "INFY",
            "WIPRO",
            "TECHM"
        ],

        "BANKING": [
            "HDFCBANK",
            "ICICIBANK",
            "SBIN"
        ],

        "PHARMA": [
            "SUNPHARMA",
            "CIPLA",
            "DRREDDY"
        ]
    }

    res = []

    for sec, stk in sectors.items():

        perf = []

        for s in stk:

            if s not in all_data:
                continue

            df = all_data[s]

            if df is None:
                continue

            start = float(
                df["Close"].iloc[0]
            )

            end = float(
                df["Close"].iloc[-1]
            )

            ret = (
                (end - start)
                / start
            ) * 100

            perf.append(ret)

        if len(perf) > 0:

            res.append({
                "Sector": sec,
                "Performance %": round(
                    np.mean(perf),
                    2
                )
            })

    st.dataframe(
        pd.DataFrame(res)
    )



elif page == "News AI":

    stock = st.text_input(
        "Stock For News",
        "INFY"
    )

    if st.button("Fetch News"):

        url = (
            f"https://news.google.com/rss/search?q="
            f"{stock}+stock"
        )

        news = feedparser.parse(url)

        results = []

        for item in news.entries[:10]:

            title = item.title

            sentiment = (
                TextBlob(title)
                .sentiment
                .polarity
            )

            if sentiment > 0:

                label = "Positive"

            elif sentiment < 0:

                label = "Negative"

            else:

                label = "Neutral"

            results.append({
                "Headline": title,
                "Sentiment": label,
                "Link": item.link
            })

        st.dataframe(
            pd.DataFrame(results)
        )



elif page == "AI Prediction":

    stock = st.selectbox(
        "Select Stock For Prediction",
        list(filtered.keys())
    )

    if st.button("Predict Next Price"):

        df = all_data[stock]

        if df is not None:

            data = df[["Close"]].copy()

            data["Prediction"] = (
                data["Close"].shift(-1)
            )

            data.dropna(inplace=True)

            X = np.array(
                data[["Close"]]
            )

            y = np.array(
                data["Prediction"]
            )

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42
            )

            model = LinearRegression()

            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_test
            )

            accuracy = r2_score(
                y_test,
                predictions
            )

            last_price = np.array([
                [float(df["Close"].iloc[-1])]
            ])

            future_price = model.predict(
                last_price
            )[0]

            st.subheader(
                f"📈 Predicted Next Price: ₹{round(future_price, 2)}"
            )

            st.metric(
                "Model Accuracy",
                f"{round(accuracy * 100, 2)}%"
            )

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                y=y_test,
                mode='lines',
                name='Actual'
            ))

            fig.add_trace(go.Scatter(
                y=predictions,
                mode='lines',
                name='Predicted'
            ))

            fig.update_layout(
                template="plotly_dark",
                title="AI Prediction Graph"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )
