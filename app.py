import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
import tensorflow
import time

# def fetch_data(ticker, start, end):
#     time.sleep(2)  # Add a short delay
#     return yf.download(ticker, start=start, end=end)

# data = fetch_data('BTC-USD', '2018-01-01', '2023-01-01')

# List of cryptocurrency abbreviations
cryptos = [
    "AUR",   # Auroracoin
    "BCH",   # Bitcoin Cash
    "BTC",   # Bitcoin
    "DASH",  # Dash
    "DOGE",  # Dogecoin
    "EOS",   # EOS.IO
    "ETC",   # Ethereum Classic
    "ETH",   # Ether (Ethereum)
    "GRC",   # Gridcoin
    "LTC",   # Litecoin
    "MZC",   # Mazacoin
    "NEO",   # Neo
    "NMC",   # Namecoin
    "Nxt",   # NXT
    "POT",   # PotCoin
    "PPC",   # Peercoin
    "TIT",   # Titcoin
    "USDC",  # USD Coin
    "USDT",  # Tether
    "VTC",   # Vertcoin
    "XEM",   # NEM
    "XLM",   # Stellar
    "XMR",   # Monero
    "XPM",   # Primecoin
    "XRP",   # Ripple
    "XVG",   # Verge
    "ZEC"    # Zcash
]

# Add "-USD" to each abbreviation
yfinance_symbols = [f"{crypto}-USD" for crypto in cryptos]

# Print the resulting array
# print(yfinance_symbols)


start_date = "2019-01-01"
end_date = "2024-01-01"

# data = yf.download('BTC-USD', start='2018-01-01', end='2023-01-01')
# print(data.shape)
# try:
#     data = yf.download('BTC-USD', start='2018-01-01', end='2023-01-01')
#     st.write(data)
# except Exception as e:
#     st.error(f"Error: {e}")

st.title('Cryptocurrency Trend Prediction')
# crypto_symbol = st.text_input('Enter Stock Ticker (It can be searched from Yahoo Finance - Enter crypto with -USD)', 'BTC-USD')
crypto_symbol = st.multiselect("Choose one or more options:", yfinance_symbols)

df = yf.download(crypto_symbol, start=start_date, end=end_date)
df.columns = df.columns.get_level_values(0)

st.subheader('Data from 2019 - 2024')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100MA, 200MA & 300MA')
ma300 = df.Close.rolling(300).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(ma300, 'black')
st.pyplot(fig)

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.4)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.4):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

model = load_model('keras_model.h5')

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, pd.DataFrame(data_testing)], ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []
for i in range(100, input_data.shape[0]):
  x_test.append(input_data[i-100:i])
  y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)
scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted *= scale_factor
y_test *= scale_factor

st.subheader('Predicted Vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
