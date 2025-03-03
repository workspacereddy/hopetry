from fastapi import FastAPI
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Initialize FastAPI
app = FastAPI()

# Load AI model from Hugging Face
model_name = "bert-base-uncased"  # Replace with your trained model
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# MeldRx API Credentials (Replace with your values)
CLIENT_ID = "515c20bc202145ea8d38af0074422da5"
CLIENT_SECRET = "NQkUIoXL0zkLRQwVWfn8cOy_jEovGR"
FHIR_BASE_URL = "https://app.meldrx.com/api/fhir/42e65a34-6d55-4daf-95ef-3c3b33b8e17c"

# Authenticate with MeldRx API
def get_access_token():
    payload = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(f"{FHIR_BASE_URL}/auth/token", data=payload, headers=headers)
    return response.json().get("access_token")

# Fetch patient data from MeldRx
def get_patient_data(patient_id):
    token = get_access_token()
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/fhir+json"}
    response = requests.get(f"{FHIR_BASE_URL}/Patient/{patient_id}", headers=headers)
    return response.json()

# AI Model Prediction
def predict_health_risk(features):
    inputs = tokenizer(str(features), return_tensors="pt")
    output = model(**inputs)
    prediction = torch.argmax(output.logits, dim=1).item()
    return "High Risk" if prediction == 1 else "Low Risk"

# API Route: Fetch Patient Data & Predict Health Condition
@app.get("/predict/{patient_id}")
def predict(patient_id: str):
    patient_data = get_patient_data(patient_id)
    health_risk = predict_health_risk(patient_data)
    return {"patient_id": patient_id, "health_risk": health_risk}
