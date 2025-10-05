from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

XGBoostModel = joblib.load("pkl_files/xgb_model.pkl")
keplerGBM = joblib.load("pkl_files/light_gbm_model_kepler.pkl")
tessGBM = joblib.load("pkl_files/TESS_LightGBM.pkl")
K2GBM = joblib.load("pkl_files/lightgbm_tuned_model_K2.pkl")

app = FastAPI()

@app.get("/")
def returnHello():
    return {"message": "Hello"}

class Features(BaseModel):
    data: dict

@app.post("/predict/kepler")
def predict(features: Features):
    # Convert to DataFrame
    df = pd.DataFrame([features.data])
    
    prediction = keplerGBM.predict(df)[0]
    probs = keplerGBM.predict_proba(df)[0].tolist()

    predictionText = ""
    if (prediction == 1):
        predictionText = "Probable"
    elif (prediction == 0):
        predictionText = "Improbable"

    return {
        "prediction": predictionText,
        "probabilities": probs
    }

@app.post("/predict/tess")
def predict(features: Features):
    # Convert to DataFrame
    df = pd.DataFrame([features.data])
    
    prediction = tessGBM.predict(df)[0]
    probs = tessGBM.predict_proba(df)[0].tolist()

    predictionText = ""
    if (prediction == 1):
        predictionText = "Probable"
    elif (prediction == 0):
        predictionText = "Improbable"

    return {
        "prediction": predictionText,
        "probabilities": probs
    }

@app.post("/predict/K2")
def predict(features: Features):
    # Convert to DataFrame
    df = pd.DataFrame([features.data])
    
    prediction = K2GBM.predict(df)[0]
    probs = K2GBM.predict_proba(df)[0].tolist()

    predictionText = ""
    if (prediction == 1):
        predictionText = "Probable"
    elif (prediction == 0):
        predictionText = "Improbable"

    return {
        "prediction": predictionText,
        "probabilities": probs
    }

PORT = int(os.getenv("PORT", "8000"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=PORT)

