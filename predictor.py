import joblib
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel

#fastapi instance
app = FastAPI()

#loading model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

#input class
class HealthData(BaseModel):
    Age: float
    SystolicBP: float
    DiastolicBP: float
    BS: float
    BodyTemp: float
    HeartRate: float

#get request
@app.get("/")
def home():
    return {"message": "Welcome to the Maternal Health Risk Predictor"}


#post request for prediction
@app.post("/predict")
def predict(data: HealthData):
    df = pd.DataFrame([data.model_dump()])
    df_scaled = scaler.transform(df.values)

    prediction = model.predict(df_scaled)

    result = int(prediction[0])
    label={0:"low risk",1:"Mid Risk",2:"high risk"}[result]

    return {"predicted_risk":label}