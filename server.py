from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

XGBoostModel = joblib.load("pkl_files/xgb_model.pkl")

model = XGBoostModel

app = FastAPI()

@app.get("/")
def returnHello():
    return {"message": "Hello"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    probs = model.predict_proba(df)[0].tolist()
    return {
        "prediction": prediction,
        "probabilities": probs
    }

class Features(BaseModel):
    data: dict

@app.post("/predict/kepler")
def predict(features: Features):
    # Convert to DataFrame
    df = pd.DataFrame([features.data])
    
    # Make prediction
    prediction = model.predict(df)[0]
    probs = model.predict_proba(df)[0].tolist()
    
    return {
        "prediction": prediction,
        "probabilities": probs
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
