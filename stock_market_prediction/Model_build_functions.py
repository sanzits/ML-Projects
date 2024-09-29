from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score,accuracy_score,make_scorer, precision_score
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import base64
import joblib
import seaborn as sns

def get_training_data(file_name_path):
    return pd.read_csv(file_name_path)

def plot_roc(y_pred,y_test,y_test_probability_score, model_type):

    fpr, tpr, thresholds = roc_curve(y_test, y_test_probability_score)
    roc_auc = roc_auc_score(y_test, y_test_probability_score)
     # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve '+model_type)
    plt.legend(loc='lower right')
    #plt.show()
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    #plt.savefig('static/images/roc_curve.png')
    plt.close()
    
     # Compute additional metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    #print('Precision : ',precision)
    #print('Recall : ',recall)
    #img.seek(0)
    # Encode the image to base64
    return base64.b64encode(img.getvalue()).decode('utf-8'),precision,recall,roc_auc
    
    

def rf_straightfoward(X,y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Fit the model
    rf_classifier.fit(X_train, y_train)
    
    y_test_pred = rf_classifier.predict(X_test)
    y_test_probability_score = rf_classifier.predict_proba(X_test)[:,1]
    roc,precision,recall,roc_auc = plot_roc(y_test_pred,y_test,y_test_probability_score, 'simple RF')
    joblib.dump(rf_classifier, 'random_forest_model.pkl')
    return roc,precision,recall,roc_auc
  
def rf_hp_tuning(X,y):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2'],
    'bootstrap': [True, False]
    }
    
    rf_model = RandomForestClassifier(random_state=42)
   

    precision_scorer = make_scorer(precision_score)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                            scoring=precision_scorer, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    y_test_pred = grid_search.predict(X_test)
    y_test_probability_score = grid_search.predict_proba(X_test)[:,1]
    roc,precision,recall,roc_auc = plot_roc(y_test_pred,y_test,y_test_probability_score, 'hyper_parameterized RF')
    joblib.dump(grid_search, 'random_forest_model_pkl.pkl')
    return roc,precision,recall,roc_auc

def plot_scatter(df_grouped_A):
    class_colors = {0: 'blue', 1: 'green'}
    # Plot the area chart with highlighted classes
    plt.figure(figsize=(10, 6))
    sns.set(style="whitegrid")
    
    for class_value, group_data in df_grouped_A.groupby('label_class'):
        sns.scatterplot(x='prob_class_1', y='count', data=group_data, 
                            #drawstyle='steps-post',
                     label=f'Class {class_value}', color=class_colors[class_value])
        #plt.fill_between(group_data['prob_class_1'], group_data['count'], alpha=0.3, color=class_colors[class_value])
    
    # Add a reference line at the default threshold of 0.5
    plt.axvline(x=0.5, color='red', linestyle='--', label='Threshold 0.5')
    
    # Adding labels and title
    plt.title('Area Chart of Probability Scores and Number of Observations by Class')
    plt.xlabel('Probability Score')
    plt.ylabel('Number of Observations')
    plt.legend()
    img = BytesIO()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    #plt.savefig('static/images/roc_curve.png')
    plt.close()
    return base64.b64encode(img.getvalue()).decode('utf-8')