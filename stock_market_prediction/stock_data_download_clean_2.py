import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
#from datetime import datetime, timedelta
#import pandas_ta as ta
import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from polygon import RESTClient
#import datetime as dt
#import pandas as pd
#import numpy as np
from polygon.rest.models import (
    TickerNews,
)
from datetime import datetime,date,timedelta
#from datetime import date
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, log_loss, classification_report, roc_curve
from config import polygonAPIkey

def get_stock_tickers():

    url = 'https://en.wikipedia.org/wiki/Russell_1000_Index'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find the table containing the Russell 1000 constituents
    table = soup.find_all('table', {'class': 'wikitable'})[2]

    # Extract tickers from the table
    tickers_russel = [row.find_all('td')[1].text.strip() for row in table.find_all('tr')[1:]]

    print(len(tickers_russel))
    return tickers_russel



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


def get_vol_change(ticker, date):
    end_date = datetime.strptime(date, "%Y-%m-%d").date()
    start_date_week = end_date - timedelta(days=7)
    start_date_month = end_date - timedelta(days=30)
    client = RESTClient(polygonAPIkey)
    df_month = pd.DataFrame(client.get_aggs(ticker = ticker, 
                             multiplier = 1,
                             timespan = 'day',
                             from_ = start_date_week.strftime("%Y-%m-%d"),
                             to = end_date.strftime("%Y-%m-%d") ))

    df_week = pd.DataFrame(client.get_aggs(ticker = ticker, 
                             multiplier = 1,
                             timespan = 'day',
                             from_ = start_date_month.strftime("%Y-%m-%d"),
                             to = end_date.strftime("%Y-%m-%d") ))

#df['timestamp'] = pd.to_datetime(df['timestamp'],unit='ms')


    df_week_mvg_Avg = df_week['volume'].mean()
    df_month_mvg_Avg = df_month['volume'].mean()
    #print(df_week_mvg_Avg)
    #print(df_month_mvg_Avg)
    diff = (df_week_mvg_Avg-df_month_mvg_Avg)/df_month_mvg_Avg
    return diff

def get_sentiment(ticker, date_str):
    
    client = RESTClient(polygonAPIkey)
    end_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    start_date = end_date - timedelta(days=7)
    current_date = end_date 
    news = []
    news_list = []
    while current_date>=start_date:
        
        for n in client.list_ticker_news(ticker,published_utc=current_date.strftime("%Y-%m-%d") ,order="desc", limit=1000):
            news.append(n)
            #print(n)
        
        for index, item in enumerate(news):
        # verify this is an agg
            if isinstance(item, TickerNews):
                #datetime_obj = datetime.strptime(item.published_utc,  "%Y-%m-%dT%H:%M:%SZ")
                #date_only = datetime_obj.date()
                #print(date_only)
                #print("{:<25}{:<15}".format(item.published_utc, item.title))
                vader = SentimentIntensityAnalyzer()
                polarity_Scores = vader.polarity_scores(item.title)
                #print(polarity_Scores['compound'])
                #print(type(polarity_Scores))
                list_combo = [current_date.strftime("%Y-%m-%d"),item.title,polarity_Scores['neg'],polarity_Scores['pos'],polarity_Scores['neu'],polarity_Scores['compound']]
                news_list.append(list_combo)
            
        current_date = current_date - timedelta(days=1)
    news_array = np.array(news_list)
    if len(news_array) > 0:
        median_sentiment_neg_score = np.median(news_array[:,2].astype(float))
        median_sentiment_pos_score = np.median(news_array[:,3].astype(float))
        median_sentiment_neu_score = np.median(news_array[:,4].astype(float))
        median_sentiment_compound_score = np.median(news_array[:,5].astype(float))
        # median_sentiment_neg_score, median_sentiment_pos_score,median_sentiment_neu_score, 
        return median_sentiment_compound_score
    else :
        return 0

            