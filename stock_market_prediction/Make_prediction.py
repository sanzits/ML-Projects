import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import base64
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
import joblib
from datetime import datetime,date,timedelta
import seaborn as sns
#from datetime import date
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from io import BytesIO
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, log_loss, classification_report, roc_curve
from Model_build_functions import rf_straightfoward, rf_hp_tuning,plot_roc,plot_scatter
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
                             from_ = '2024-08-18',
                             to = '2024-08-24'))

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
        #print(tick)
        rsi_values = calculate_rsi(tick['Close'])
        tick['RSI'] = rsi_values
        tick['Signal'] = generate_signals(rsi_values)
            #data = yf.download(ticker, start='2021-01-01', end='2022-01-01')
        if tick.notnull().any().any():
            tick = tick.sort_index(ascending=False)
                #print(ticker)
            stock_name = ticker
           
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
            Relative_vol_change = get_vol_change(ticker, Cut_off_date.strftime("%Y-%m-%d"))
            

            median_sentiment_compound_score = get_sentiment(ticker, Cut_off_date.strftime("%Y-%m-%d"))
            #print('----------here-------------')
            X_Y = [stock_name, mvgAvg200, mvgAvg100, mvgAvg50,mvgAvg20, mvgAvg10,RSI_signal,FuturePrice,percived_today_price,Relative_vol_change,median_sentiment_compound_score,Cut_off_date]
        return X_Y
    except Exception as e:
         print(f"Error fetching data for {ticker}: {e}")

def make_prediction(df):


    X = df[['Percentage_Difference_200', 'Percentage_Difference_100','Percentage_Difference_50','Percentage_Difference_20','Percentage_Difference_10','RSI_signal','Relative_vol_change','median_sentiment_compound_score']]
    y = df['label_class']
    #df = pd.read_csv('/Users/sanchitsuman/Documents/Data/test1.csv')

    #X = df[['Percentage_Difference_200', 'Percentage_Difference_100','Percentage_Difference_50','Percentage_Difference_20','Percentage_Difference_10','RSI_signal','Relative_vol_change','median_sentiment_compound_score']]
    #y = df['label_class']
    model = joblib.load('/Users/sanchitsuman/vcs/github.com/drcscodes/ML-Projects/stock_market_prediction/random_forest_model_pkl.pkl')
    predictions = model.predict(X)

    probability_scores = model.predict_proba(X)
    prob_df = pd.DataFrame(probability_scores, columns=[f'prob_class_{i}' for i in model.classes_])

    fpr, tpr, thresholds = roc_curve(y, probability_scores[:,1])
    roc_auc = roc_auc_score(y, probability_scores[:,1])
    roc,precision,recall,roc_auc=plot_roc(predictions,y,probability_scores[:,1], 'RF Parameterized')
    df['prediction'] = predictions
    df = pd.concat([df.reset_index(drop=True), prob_df], axis=1)

    df['count'] = 1
    #print(df.columns)
    df_A = df[["prob_class_1" ,"label_class","count"]]
    print(df_A.columns)
    print(df_A)
    df_grouped_A = df_A.groupby(['prob_class_1','label_class']).count().reset_index()
    #label_class
    scatter_plt = plot_scatter(df_grouped_A)
    #investment_10 = (np.sum(df['prediction']==1)*10)
    every_predicted_share_bought = np.sum(df['percived_today_price'][df['prediction']==1])
    every_predicted_share_bought_fp = np.sum(df['FuturePrice'][df['prediction']==1]) 
    
    every_predicted_share_bought_bcs = np.sum(df['percived_today_price'][df['label_class']==1])
    every_predicted_share_bought_bcs_fp = np.sum(df['FuturePrice'][df['label_class']==1])
    
    return  roc,precision,recall,roc_auc,scatter_plt,df


def prediction_download():


    #st_dt = datetime(2023, 6, 30)
    end_Dt = datetime(2024, 9, 5)
    #list_a = []

    #tickers_russel = get_stock_tickers()
    #for ticker in tickers_russel :
    #     list_a.append(fetch_stock_data(ticker,st_dt,end_Dt))
         #master_list.append(list_a)
    #return list_a
    #master_list_nan = [row for row in list_a if row is not None]

    df = pd.read_csv('/Users/sanchitsuman/Documents/Data/test1.csv')


    #columns = ['tick', 'mvgAvg200', 'mvgAvg100','mvgAvg50','mvgAvg20','mvgAvg10','RSI_signal','FuturePrice','percived_today_price','Relative_vol_change','median_sentiment_compound_score','Cut_off_date']
    #df = pd.DataFrame(master_list_nan,columns=columns)
    df['Percentage_Difference_200'] = (df['percived_today_price'] - df['mvgAvg200']) / (df['percived_today_price'])
    df['Percentage_Difference_100'] = (df['percived_today_price'] - df['mvgAvg100']) / (df['percived_today_price'])
    df['Percentage_Difference_50'] = (df['percived_today_price'] - df['mvgAvg50']) / (df['percived_today_price'])
    df['Percentage_Difference_20'] = (df['percived_today_price'] - df['mvgAvg20']) / (df['percived_today_price'])
    df['Percentage_Difference_10'] = (df['percived_today_price'] - df['mvgAvg10']) / (df['percived_today_price'])   
    df['label_class'] = df.apply(lambda row: 1 if (row['FuturePrice'] - row['percived_today_price'])/ row['percived_today_price']>0.05 else 0, axis=1)
    df_cleaned = df.dropna()
    today_date = datetime.today().strftime('%Y-%m-%d')

    # Define the filename with today's date appended
    
    #filename = f'Prediction_{today_date}.csv'
    #df_cleaned.to_csv('/Users/sanchitsuman/Documents/Data/'+filename, index=False)  
    return df_cleaned




    

