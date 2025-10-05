from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from fastapi.middleware.cors import CORSMiddleware
from analytics import router as analytics_router

XGBoostModel = joblib.load("pkl_files/xgb_model.pkl")
keplerGBM = joblib.load("pkl_files/light_gbm_model_kepler.pkl")
tessGBM = joblib.load("pkl_files/TESS_LightGBM.pkl")
K2GBM = joblib.load("pkl_files/K2_FinalV4.pkl")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # or ["*"] to allow all
    allow_credentials=True,
    allow_methods=["*"],         # allow all HTTP methods
    allow_headers=["*"],         # allow all headers
)

# Include analytics router
app.include_router(analytics_router)

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
    try:
        # Convert to DataFrame
        df = pd.DataFrame([features.data])

        # Example model prediction
        prediction = K2GBM.predict(df)[0]
        probs = K2GBM.predict_proba(df)[0].tolist()

        predictionText = "Probable" if prediction == 1 else "Improbable"

        return {
            "prediction": predictionText,
            "probabilities": probs
        }

    except Exception as e:
        # Return the actual error message
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/hot-stars")
def get_hot_stars():
    """
    Get the top 50 hottest stars from Kepler dataset based on koi_teq (equilibrium temperature).
    """
    try:
        # Read the Kepler unmodified dataset
        df = pd.read_csv("datasets/unmodified/Kepler Objects of Interest.csv", comment="#")

        # Check if koi_teq column exists
        if "koi_teq" not in df.columns:
            raise HTTPException(
                status_code=500,
                detail="Column 'koi_teq' not found in dataset"
            )

        # Get top 50 hottest stars, excluding NaN values
        hot_stars = df.nlargest(50, "koi_teq")[["kepid", "kepler_name", "koi_teq"]].copy()

        # Convert to list of dictionaries
        results = []
        for _, row in hot_stars.iterrows():
            results.append({
                "kepid": int(row["kepid"]) if pd.notna(row["kepid"]) else None,
                "kepler_name": row["kepler_name"] if pd.notna(row["kepler_name"]) else None,
                "temperature": float(row["koi_teq"]) if pd.notna(row["koi_teq"]) else None
            })

        return {
            "dataset": "kepler",
            "count": len(results),
            "results": results
        }

    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Dataset file not found: datasets/unmodified/Kepler Objects of Interest.csv"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing dataset: {str(e)}"
        )

PORT = int(os.getenv("PORT", "8000"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=PORT)

