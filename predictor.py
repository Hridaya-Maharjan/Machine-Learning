import joblib
from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel
app = FastAPI()

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ✅ Define input schema
class HealthData(BaseModel):
    Age: float
    SystolicBP: float
    DiastolicBP: float
    BS: float
    BodyTemp: float
    HeartRate: float

@app.get("/")
def home():
    return {"message": "Welcome to the Maternal Health Risk Predictor"}



@app.post("/predict")
def predict(data: HealthData):
    df = pd.DataFrame([data.model_dump()])
    df_scaled = scaler.transform(df.values)

    prediction = model.predict(df_scaled)

    # force conversion to pure python
    result = int(prediction[0])
    label={0:"low risk",1:"Mid Risk",2:"high risk"}[result]

    return {"predicted_risk":label}