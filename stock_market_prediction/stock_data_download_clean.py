import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
#import pandas_ta as ta
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, log_loss, classification_report, roc_curve

url = 'https://en.wikipedia.org/wiki/Russell_1000_Index'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Find the table containing the Russell 1000 constituents
table = soup.find_all('table', {'class': 'wikitable'})[2]

# Extract tickers from the table
tickers_russel = [row.find_all('td')[1].text.strip() for row in table.find_all('tr')[1:]]

print(len(tickers_russel))


def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Generate Trading Signals
def generate_signals(rsi_values):
    signals = []
    for rsi in rsi_values:
        if rsi > 70:
            signals.append(1)#SELL
        elif rsi < 30:
            signals.append(-1)#BUY
        else:
            signals.append(0)#HOLD
    return signals


def fetch_stock_data(ticker,start_date,end_date):
    #d=yf.download(tickers='AAPL',
                  #start='2023-06-26', end='2024-06-26',
                  #period='1y',interval='1d',rounding=True)
    #rsi_values = calculate_rsi(d['Close'])
    #d['RSI'] = rsi_values
    #d['Signal'] = generate_signals(rsi_values)
    #d.sort_index(ascending=False).reset_index()
    try:
               
            # Fetch historical data for a ticker
        tick = yf.download(tickers=ticker,
                                start=start_date, end=end_date,
                              interval='1d',rounding=True)
        #print('Here')
        rsi_values = calculate_rsi(tick['Close'])
        tick['RSI'] = rsi_values
        tick['Signal'] = generate_signals(rsi_values)
            #data = yf.download(ticker, start='2021-01-01', end='2022-01-01')
        if tick.notnull().any().any():
            tick = tick.sort_index(ascending=False)
                #print(ticker)
            stock_name = ticker
            #print('-----------------------')
                #print(df_dummy.iloc[0,0])
          
            mvgAvg200 = tick.iloc[7:205,4].mean(skipna=True)
                #print(mvgAvg200)
            mvgAvg100 = tick.iloc[7:105,4].mean(skipna=True)
            mvgAvg50 = tick.iloc[7:55,4].mean(skipna=True)
            mvgAvg20 = tick.iloc[7:25,4].mean(skipna=True)
            mvgAvg10 = tick.iloc[7:15,4].mean(skipna=True)
            mvgLabel = tick.iloc[0:6,4].mean(skipna=True)
           
            RSI_signal = tick.iloc[6,7]
            FuturePrice = tick.iloc[0,4]
            percived_today_price = tick.iloc[7,4]  
            Cut_off_date = tick.index[7]
            
            X_Y = [stock_name, mvgAvg200, mvgAvg100, mvgAvg50,mvgAvg20, mvgAvg10,RSI_signal,FuturePrice,percived_today_price,Cut_off_date]
        return X_Y
    except Exception as e:
         print(f"Error fetching data for {ticker}: {e}")



def create_stock_hist_data(st_dt,end_Dt):
    for ticker in tickers_russel :
         list_a = fetch_stock_data(ticker,st_dt,end_Dt)
         master_list.append(list_a)

master_list= []


st_dt = datetime(2023, 4, 30)
end_Dt = datetime(2024, 6, 30)

for i in range(10):
    create_stock_hist_data(st_dt.strftime('%Y-%m-%d'),end_Dt.strftime('%Y-%m-%d'))
    print(st_dt)
    print(end_Dt)
    print(len(master_list))
    st_dt = st_dt-timedelta(days=1)
    end_Dt = end_Dt-timedelta(days=1)
    
none_count = 0
for sublist in master_list:
    
    if sublist is None:
        master_list.remove(sublist)

master_list_nan = master_list

columns = ['tick', 'mvgAvg200', 'mvgAvg100','mvgAvg50','mvgAvg20','mvgAvg10','RSI_signal','FuturePrice','percived_today_price','Cut_off_date']
df = pd.DataFrame(master_list_nan,columns=columns)

nan_counts = df.isna().sum()

# Display the NaN counts
print("NaN counts for each column:")
print(nan_counts)



df['Percentage_Difference_200'] = (df['percived_today_price'] - df['mvgAvg200']) / (df['percived_today_price'])
df['Percentage_Difference_100'] = (df['percived_today_price'] - df['mvgAvg100']) / (df['percived_today_price'])
df['Percentage_Difference_50'] = (df['percived_today_price'] - df['mvgAvg50']) / (df['percived_today_price'])
df['Percentage_Difference_20'] = (df['percived_today_price'] - df['mvgAvg20']) / (df['percived_today_price'])
df['Percentage_Difference_10'] = (df['percived_today_price'] - df['mvgAvg10']) / (df['percived_today_price'])


print(df.head())

df['label_class'] = df.apply(lambda row: 1 if (row['FuturePrice'] - row['percived_today_price'])/ row['percived_today_price']>0.1 else 0, axis=1)

df_cleaned = df.dropna()

df_cleaned.to_csv('/Users/sanchitsuman/Documents/Data/Stock_data.csv', index=False)    