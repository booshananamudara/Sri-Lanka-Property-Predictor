"""
Sri Lanka Property Price Predictor — FastAPI Backend
=====================================================
Prediction API endpoint with model loading and inference.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import numpy as np
import os
import pandas as pd

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "catboost_model.pkl")
PREPROCESSOR_PATH = os.path.join(BASE_DIR, "models", "preprocessor.pkl")

# Load model and preprocessor
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(PREPROCESSOR_PATH, "rb") as f:
    preprocessor = pickle.load(f)

label_encoders = preprocessor["label_encoders"]
feature_columns = preprocessor["feature_columns"]
categorical_cols = preprocessor["categorical_cols"]

# Extract valid values for dropdowns
city_options = sorted(label_encoders["City"].classes_.tolist())
district_options = sorted(label_encoders["District"].classes_.tolist())

# Create FastAPI app
app = FastAPI(
    title="Sri Lanka Property Price Predictor",
    description="Predict house prices in Sri Lanka using CatBoost ML model",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request model
class PredictionRequest(BaseModel):
    bedrooms: int
    bathrooms: int
    land_size_perches: float
    house_size_sqft: float
    city: str
    district: str


# Response model
class PredictionResponse(BaseModel):
    predicted_price: float
    predicted_price_formatted: str
    predicted_price_millions: float


@app.get("/")
def root():
    return {
        "message": "Sri Lanka Property Price Predictor API",
        "endpoints": {
            "/predict": "POST - Predict property price",
            "/options": "GET - Get valid city/district options",
            "/docs": "GET - API documentation",
        },
    }


@app.get("/options")
def get_options():
    """Return valid city and district options for the frontend."""
    return {
        "cities": city_options,
        "districts": district_options,
        "bedrooms_range": list(range(1, 11)),
        "bathrooms_range": list(range(1, 11)),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Predict property price based on input features."""

    # Encode city
    if request.city in label_encoders["City"].classes_:
        city_encoded = label_encoders["City"].transform([request.city])[0]
    else:
        # Default to most common city
        city_encoded = 0

    # Encode district
    if request.district in label_encoders["District"].classes_:
        district_encoded = label_encoders["District"].transform([request.district])[0]
    else:
        district_encoded = 0

    # Create feature array
    features = pd.DataFrame(
        [[
            request.bedrooms,
            request.bathrooms,
            request.land_size_perches,
            request.house_size_sqft,
            city_encoded,
            district_encoded,
        ]],
        columns=feature_columns,
    )

    # Predict
    predicted_price = float(model.predict(features)[0])
    predicted_price = max(0, predicted_price)  # Ensure non-negative

    # Format
    predicted_millions = round(predicted_price / 1_000_000, 2)
    formatted = f"Rs. {predicted_price:,.0f}"

    return PredictionResponse(
        predicted_price=predicted_price,
        predicted_price_formatted=formatted,
        predicted_price_millions=predicted_millions,
    )
