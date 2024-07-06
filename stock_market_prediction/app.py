from flask import Flask, render_template
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/roc_curve')
def roc_curve_view():
    df = pd.read_csv('/Users/sanchitsuman/Documents/Data/Stock_data_2024-07-02.csv')
    df_cleaned = df.dropna()
    X = df_cleaned[['Percentage_Difference_200', 'Percentage_Difference_100','Percentage_Difference_50','Percentage_Difference_20','Percentage_Difference_10','RSI_signal']]
    y = df_cleaned['label_class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Fit the model
    rf_classifier.fit(X_train, y_train)

    y_test_pred = rf_classifier.predict(X_test)
    y_test_probability_score = rf_classifier.predict_proba(X_test)[:,1]

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
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.savefig('static/images/roc_curve.png')
    plt.close()

     # Compute additional metrics
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)

    # Pass metrics to the template
    return render_template('roc_curve.html', roc_auc=roc_auc, precision=precision, recall=recall)

if __name__ == '__main__':
    app.run(debug=True)