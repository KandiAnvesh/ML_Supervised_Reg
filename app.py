from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
with open('rf_reg.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')  # Create this HTML file later

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the form
    data = request.form
    features = np.array([[
        data['Age'], data['BusinessTravel'], data['Department'], data['DistanceFromHome'], 
        data['Education'], data['EnvironmentSatisfaction'], data['Gender'], 
        data['JobInvolvement'], data['JobLevel'], data['JobRole'], data['JobSatisfaction'], 
        data['MaritalStatus'], data['NumCompaniesWorked'], data['OverTime'], 
        data['PercentSalaryHike'], data['PerformanceRating'], data['StockOptionLevel'], 
        data['TotalWorkingYears'], data['WorkLifeBalance'], data['YearsAtCompany'], 
        data['YearsInCurrentRole'], data['YearsSinceLastPromotion'], data['YearsWithCurrManager']
    ]], dtype=float)  # Ensure the input matches your model's expected type
    
    prediction = model.predict(features)
    
    return jsonify({
        "predicted_monthly_income": prediction[0]
    })

if __name__ == '__main__':
    app.run(debug=True)
