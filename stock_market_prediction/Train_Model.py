from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score,accuracy_score,make_scorer, precision_score
import matplotlib
#import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64
import joblib
import seaborn as sns
#from sklearn.datasets import make_classification
#from sklearn.model_selection import train_test_split,GridSearchCV
#from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score,accuracy_score,make_scorer, precision_score
#import matplotlib
#matplotlib.use("Qt5Agg")
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
from Model_build_functions import rf_straightfoward, rf_hp_tuning,plot_roc
import json
from datetime import datetime,date,timedelta

def store_data(roc, precision, recall, roc_auc,model_type ):

	# Decode the Base64 string
	today_date = datetime.today().strftime('%Y-%m-%d')
	image_data = base64.b64decode(roc)

	# Specify the file path where you want to save the image
	filename_img = f'Roc_Curve_{model_type}_{today_date}.png'
	file_path_img = '/Users/sanchitsuman/vcs/github.com/drcscodes/ML-Projects/stock_market_prediction/static/images/'+filename_img
	#"/Users/sanchitsuman/vcs/github.com/drcscodes/ML-Projects/static/images/{image_name}.png"
	print(file_path_img)
	# Write the decoded image data to a file
	with open(file_path_img, 'wb') as file: 
		file.write(image_data)

    # Define the filename with today's date appended
    #"static/images/{image_name}.png"
	filename_json = f'Training_model_Variables_{model_type}_{today_date}.json'
	file_path_json = '/Users/sanchitsuman/documents/Data/'+filename_json
	print(file_path_json)

	values_dict = {
		"roc_curve_file" : file_path_img,
		"date_trained" : today_date,
		"model_type": model_type,
	    "precision": precision,
	    "recall": recall,
	    "roc_auc": roc_auc
	}

    # Save the dictionary to a JSON file
	with open(file_path_json, 'w') as json_file:
		json.dump(values_dict, json_file, indent=4)


	print(f"Values have been saved")

def train():
	df = pd.read_csv('/Users/sanchitsuman/documents/Data/Stock_data_2024-08-31.csv')
	X = df[['Percentage_Difference_200', 'Percentage_Difference_100','Percentage_Difference_50','Percentage_Difference_20','Percentage_Difference_10','RSI_signal','Relative_vol_change','median_sentiment_compound_score']]
	y = df['label_class']

	roc_rf, precision_rf, recall_rf, roc_auc_rf = rf_straightfoward(X,y)
	store_data(roc_rf, precision_rf, recall_rf, roc_auc_rf,'Random Forest')

	roc_rf_hp, precision_rf_hp, recall_rf_hp, roc_auc_rf_hp = rf_hp_tuning(X,y)
	store_data(roc_rf_hp, precision_rf_hp, recall_rf_hp, roc_auc_rf_hp,'Random Forest Hyper Parameterized')

def main():
	train()

if __name__ == "__main__":
	main()