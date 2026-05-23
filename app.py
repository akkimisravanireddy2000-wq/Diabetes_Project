import joblib
import numpy as np
from flask import Flask, request, jsonify 
model = joblib.load('diabetes_pipeline (1).pkl')
app = Flask(__name__) # creating flask name
@app.route('/') # route define the prediction function
def home():
    return '<h1>Diabetes Prediction using Flask API</h1>'
@app.route('/predict',methods=['POST'])
def predict():
    data = request.json
    feature =np.array([[
        data['Pregnancies'],
        data['Glucose'],
        data['BloodPressure'], 
        data['SkinThickness'], 
        data['Insulin'],
        data['BMI'], 
        data['DiabetesPedigreeFunction'],
        data['Age']
        
    ]])
    prediction = model.predict(feature)
    probability = model.predict_proba(feature) # this will give the confidence score for each class
    result ='Diabetes Detected'
    if prediction[0] == 0:
        result = 'No Diabetes'
    confidence_score = round(np.max(probability)*100, 2) # get the maximum probability and round it to 2 decimal places
    return jsonify({
        'prediction': result,
        'confidence': confidence_score
    })
if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )