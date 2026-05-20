import requests

print("NSE STOCK CHECK")

symbol = input("Enter NSE stock (example: INFY): ").upper()

url = f"https://www.nseindia.com/api/quote-equity?symbol={symbol}"

headers = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br"
}

session = requests.Session()
session.get("https://www.nseindia.com", headers=headers)

response = session.get(url, headers=headers)

data = response.json()

price = data["priceInfo"]["lastPrice"]
open_price = data["priceInfo"]["open"]
high = data["priceInfo"]["intraDayHighLow"]["max"]
low = data["priceInfo"]["intraDayHighLow"]["min"]

print("\nSTOCK DATA\n")
print("Symbol:", symbol)
print("Price:", price)
print("Open:", open_price)
print("High:", high)
print("Low:", low)