import requests

# API endpoint
url = "http://127.0.0.1:8000/predict"

# Patient data
data = {
    "Age": 54,
    "Sex": 1,  
    "ChestPainType": 0,  
    "RestingBP": 108,
    "Cholesterol": 267,
    "FastingBS": 0,  
    "RestingECG": 1,  
    "MaxHR": 167,
    "ExerciseAngina": 0,  
    "Oldpeak": 0,
    "ST_Slope": 2  
}

# Make prediction
response = requests.post(url, json=data)
result = response.json()

# Print results
for model, prediction in result.items():
    risk = "Heart Disease" if prediction else "Normal"
    print(f"{model}: {risk}")