import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from textblob import TextBlob
import numpy as np
import time

print("SMART STOCK ANALYZER")

stock = input("Enter Stock Symbol: ")

data = None

for i in range(3):
    try:
        data = yf.download(stock, period="1y", progress=False)
        if data is not None and not data.empty:
            break
    except:
        pass
    time.sleep(2)

if data is None or data.empty:
    print("Failed to fetch stock data")
    exit()

data['MA20'] = data['Close'].rolling(20).mean()
data['MA50'] = data['Close'].rolling(50).mean()

delta = data['Close'].diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()

rs = avg_gain / avg_loss
data['RSI'] = 100 - (100/(1+rs))

data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

data = data.dropna()

features = data[['RSI','MA20','MA50','Volume']]
target = data['Target']

X_train, X_test, y_train, y_test = train_test_split(features,target,test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train,y_train)

latest = features.iloc[-1:].values

prediction = model.predict(latest)

probability = model.predict_proba(latest)

confidence = round(max(probability[0])*100,2)

trend = "Bearish"

if prediction[0] == 1:
    trend = "Bullish"

news_samples = [
"Company reports strong quarterly earnings",
"Company expands new business operations",
"Company faces regulatory investigation"
]

scores = []

for news in news_samples:
    scores.append(TextBlob(news).sentiment.polarity)

sentiment = "Neutral"

if sum(scores)/len(scores) > 0:
    sentiment = "Positive"
elif sum(scores)/len(scores) < 0:
    sentiment = "Negative"

returns = data['Close'].pct_change()

volatility = returns.std().item()

risk = "Medium"

if volatility < 0.01:
    risk = "Low"
elif volatility > 0.02:
    risk = "High"

support = round(data['Low'].tail(20).min(),2)

resistance = round(data['High'].tail(20).max(),2)

score = 50

if trend == "Bullish":
    score += 20
else:
    score -= 10

if sentiment == "Positive":
    score += 15
elif sentiment == "Negative":
    score -= 15

if risk == "Low":
    score += 15
elif risk == "High":
    score -= 15

score = max(0,min(100,score))

outlook = "Neutral"

if score > 75:
    outlook = "Strong Buy"
elif score > 60:
    outlook = "Buy"
elif score > 40:
    outlook = "Hold"
else:
    outlook = "Avoid"

print("\nSMART STOCK ANALYZER RESULT\n")

print("Trend:",trend)

print("AI Prediction:",prediction[0])

print("Confidence:",confidence,"%")

print("News Sentiment:",sentiment)

print("Risk Level:",risk)

print("Support Level:",support)

print("Resistance Level:",resistance)

print("Stock Health Score:",score,"/ 100")

print("FINAL OUTLOOK:",outlook)

plt.figure(figsize=(10,5))

plt.plot(data['Close'],label="Price")

plt.plot(data['MA20'],label="MA20")

plt.plot(data['MA50'],label="MA50")

plt.legend()

plt.title(stock + " Analysis")

plt.show()