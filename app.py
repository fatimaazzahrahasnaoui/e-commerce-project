from flask import Flask, render_template, send_file,send_from_directory,request, redirect, url_for
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.cluster import KMeans
from matplotlib.figure import Figure
import seaborn as sns
import plotly.express as px
import io
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

app = Flask(__name__, static_folder="static")

model = joblib.load("stock_prediction_model.pkl")
scaler = joblib.load("scaler2.pkl")

data = pd.read_csv('DATA/data_commerce.csv')  



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/statistics')
def statistics():
    description = data.describe().to_html(classes="table table-striped")
    return render_template('results.html', description=description)



@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        quantity = float(request.form['quantity'])
        unit_price = float(request.form['unit_price'])
        age = int(request.form['age'])
        stock = int(request.form['stock'])
        rating = float(request.form['rating'])
        days_since_registration = int(request.form['days_since_registration'])
        
        input_data = np.array([[quantity, unit_price, age, stock, rating, days_since_registration]])
        input_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_scaled)[0]
        prediction = round(prediction, 2)
    
    return render_template('predict.html', prediction=prediction)

@app.route('/age_distribution')
def age_distribution():
    return send_from_directory('static/images', 'age_distribution.png')

@app.route('/category_distribution')
def category_distribution():
    return send_from_directory('static/images', 'category_distribution.png')

@app.route('/user_segmentation')
def user_segmentation():
    return send_from_directory('static/images', 'user_segmentation.png')

@app.route('/power_bi_dashboard')
def power_bi_dashboard():
    
    power_bi_url = "https://app.powerbi.com/reportEmbed?reportId=25e967c5-eefd-46aa-9469-3ac526be3075&autoAuth=true&embeddedDemo=true"
    return render_template('power_bi.html', power_bi_url=power_bi_url)

if __name__ == '__main__':
    app.run(debug=True)


