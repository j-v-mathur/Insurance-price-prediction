from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import math
from joblib import load
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model and scaler reference data
model = load('insurance.joblib')
scaler_data = pd.read_csv('Preprocessed_insurance_data.csv').drop(columns=['charges'])
scaler = StandardScaler()
scaler.fit(scaler_data)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = int(request.form['smoker'])
        if bmi < 10 or bmi > 60:
            return render_template("index.html", prediction="BMI must be between 10 and 60.")
        if age < 18 or bmi > 65:
            return render_template("index.html", prediction="Age must be between 18 and 65.")
        bmi_log = math.log(bmi)
        input_df = pd.DataFrame({
            'age': [age],
            'bmi': [bmi_log],
            'children': [children],
            'smoker_yes': [smoker]
        })

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        prediction = round(prediction, 2)

        return render_template('index.html', prediction=f"${prediction}")
    
    except Exception as e:
        return render_template('index.html', prediction="Error")

if __name__ == '__main__':
    app.run(port=5000)
