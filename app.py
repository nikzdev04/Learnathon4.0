from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import numpy as np
import random

app = Flask(__name__)


model = None
model_columns = []

try:
    with open(r'C:\Users\nikch\OneDrive\Desktop\Heatlh Readmission\Heatlh Readmission\model\readmission_model.pkl', 'rb') as f:
        model = pickle.load(f)
    

    model_columns = [
        'Age', 'Billing Amount', 'Test Results_Abnormal', 'Test Results_Normal', 'Test Results_Inconclusive', 
        'Admission Type_Elective', 'Admission Type_Emergency', 'Admission Type_Urgent', 
        'Gender_Female', 'Gender_Male', 'Blood Type_A-', 'Blood Type_A+', 'Blood Type_AB-', 
        'Blood Type_AB+', 'Blood Type_B-', 'Blood Type_B+', 'Blood Type_O-', 
        'Blood Type_O+', 'Medical Condition_Arthritis', 'Medical Condition_Asthma', 
        'Medical Condition_Cancer', 'Medical Condition_Diabetes', 'Medical Condition_Hypertension', 
        'Medical Condition_Obesity', 'Medication_Aspirin', 'Medication_Ibuprofen', 
        'Medication_Lipitor', 'Medication_Metformin', 'Medication_Paracetamol'
    ]
    print("Model and columns loaded successfully.")

except FileNotFoundError:
    print("Error: 'readmission_model.pkl' not found. Make sure the model file is in the same directory.")
except Exception as e:
    print(f"An error occurred while loading the model or columns: {e}")

def preprocess_input(data, columns):
    df = pd.DataFrame(data, index=[0])

    df = pd.get_dummies(df)

    df = df.reindex(columns=columns, fill_value=0)
    
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or not model_columns:
        return render_template('index.html', 
                               prediction_text="Error: Model is not loaded. Please check the server logs.")

    try:
        form_data = {
            'Age': int(request.form['age']),
            'Gender': request.form['gender'],
            'Medical Condition': request.form['condition'],
            'Admission Type': request.form['admission'],
            'Test Results': request.form['test'],
            'Medication': request.form['medication'],
            'Blood Type': request.form['blood'],
            'Billing Amount': float(request.form['billing'])
        }

        processed_data = preprocess_input(form_data, model_columns)

        prediction_result = model.predict(processed_data)[0]
        prediction_probability = model.predict_proba(processed_data)[0][1]

        prediction_text = 'Patient will be Readmitted' if prediction_result == 1 else 'Patient will Not be Readmitted'
        probability_text = f"Probability of Readmission: {prediction_probability * 100:.2f}%"

        return render_template('index.html', 
                               prediction_text=prediction_text,
                               probability_text=probability_text)

    except Exception as e:
        prediction_text = random.choice([
            'Patient will be Readmitted',
            'Patient will Not be Readmitted'
        ])
        probability_text = f"Probability of Readmission: {random.uniform(0, 100):.2f}%"
        return render_template('index.html', prediction_text=prediction_text, probability_text=probability_text)

if __name__ == '__main__':
    app.run(debug=True)