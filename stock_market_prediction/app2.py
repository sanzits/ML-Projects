from flask import Flask, render_template
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score,accuracy_score,make_scorer, precision_score
import matplotlib
#matplotlib.use("Qt5Agg")
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from Model_build_functions import rf_straightfoward, rf_hp_tuning,plot_roc
import base64
from io import BytesIO
from Make_prediction import make_prediction,prediction_download
from datetime import datetime,date,timedelta
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model_training_results')
def roc_curve_view():
    today_date = datetime.today().strftime('%Y-%m-%d')
    #df = pd.read_csv('/Users/sanchitsuman/documents/Data/Stock_data_2024-08-31.csv')
    #df_cleaned = df.dropna()
    model_type_rf= 'Random Forest'
    filename_json = f'Training_model_Variables_{model_type_rf}_{today_date}.json'
    file_path_json = '/Users/sanchitsuman/documents/Data/'+filename_json
    print(file_path_json)
    with open(file_path_json, 'r') as json_file:
        data_rf= json.load(json_file)

    model_type_hp= 'Random Forest Hyper Parameterized'

    filename_json = f'Training_model_Variables_{model_type_hp}_{today_date}.json'
    file_path_json = '/Users/sanchitsuman/documents/Data/'+filename_json

    with open(file_path_json, 'r') as json_file:
        data_rf_hp= json.load(json_file)
    image_rf = f'Roc_Curve_{model_type_rf}_{today_date}.png'
    image_rf_hp = f'Roc_Curve_{model_type_hp}_{today_date}.png'

    return render_template('model_training_results.html',
        path = '/Users/sanchitsuman/vcs/github.com/drcscodes/ML-Projects/stock_market_prediction/static/',
        roc_rf=image_rf,
        roc_rf_hp=image_rf_hp,
        precision_rf=data_rf['precision'], 
        recall_rf=data_rf['recall'], 
        roc_auc_rf=data_rf['roc_auc'],
        precision_rf_hp=data_rf_hp['precision'], 
        recall_rf_hp=data_rf_hp['recall'], 
        roc_auc_rf_hp=data_rf_hp['roc_auc'])

@app.route('/Latest_predictions')
def prediction():
	roc,precision,recall,roc_auc,scatter_plt,df = make_prediction(prediction_download())
	return render_template('Latest_predictions.html',
		roc = roc,
		precision = precision,
		recall = recall,
		roc_auc = roc_auc,
		scatter_plt = scatter_plt
		)

if __name__ == '__main__':
    app.run(debug=True)