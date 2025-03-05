from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import pickle
import os

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Pydantic model for input data validation
class CreditScoreRequest(BaseModel):
    Age: Optional[float] = None
    Annual_Income: Optional[float] = None
    Delay_from_due_date: Optional[float] = None
    Num_of_Delayed_Payment: Optional[float] = None
    Num_Credit_Inquiries: Optional[float] = None
    Outstanding_Debt: Optional[float] = None
    Total_EMI_per_month: Optional[float] = None
    Credit_Age_years: Optional[float] = None

# Load the model on startup
model_components = None

@app.on_event("startup")
async def load_model():
    global model_components
    try:
        model_path = os.path.join("models", "tgamt_model.pkl")
        with open(model_path, "rb") as f:
            model_components = pickle.load(f)
        print("TGAMT model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        model_components = None

@app.get("/")
async def root():
    return {"message": "Credit Score"}

@app.post("/predict")
async def predict_credit_score(request: CreditScoreRequest):
    global model_components
    
    if model_components is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Extract model components
    tree = model_components['tree']
    leaf_gams = model_components['leaf_gams']
    imputer = model_components['imputer']
    numeric_features = model_components['numeric_features']
    unique_leaves = model_components['unique_leaves']
    
    # Convert request to DataFrame
    input_data = pd.DataFrame([request.dict()])
    
    # Ensure all needed columns exist and process features
    for feature in numeric_features:
        if feature not in input_data.columns:
            input_data[feature] = np.nan
    
    X = input_data[numeric_features]
    X_imputed = imputer.transform(X)
    
    # Get leaf node and GAM model
    leaf_idx = tree.apply(X_imputed)[0]
    if leaf_idx not in leaf_gams:
        raise HTTPException(status_code=400, detail=f"No model available for leaf {leaf_idx}")
    
    
    # Check if we have a GAM for this leaf
    if leaf_idx not in leaf_gams:
        raise HTTPException(status_code=400, detail=f"No model available for leaf {leaf_idx}")
    
    # Predict using GAM
    gam = leaf_gams[leaf_idx]
    prediction = gam.predict(X_imputed)[0]
    
    # Clip and round prediction
    prediction = np.clip(prediction, 350, 850)
    prediction = int(round(prediction))
    
    # Map numeric score to credit level
    credit_level = "Poor"
    if 600 <= prediction <= 700:
        credit_level = "Standard"
    elif prediction > 700:
        credit_level = "Good"
    
    return {
        "credit_score": prediction,
        "credit_level": credit_level
    }





if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)