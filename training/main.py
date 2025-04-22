from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf
import joblib

# Load model and scaler once
model = tf.keras.models.load_model('model/saved_model.keras')
scaler = joblib.load('model/scaler.save')

app = FastAPI()

# Define input schema
class HouseFeatures(BaseModel):
    CRIM: float
    ZN: float
    INDUS: float
    CHAS: float
    NOX: float
    RM: float
    AGE: float
    DIS: float
    RAD: float
    TAX: float
    PTRATIO: float
    B: float
    LSTAT: float

@app.post("/predict")
def predict(features: HouseFeatures):
    # Convert input to array
    input_data = np.array([[features.CRIM, features.ZN, features.INDUS, features.CHAS,
                            features.NOX, features.RM, features.AGE, features.DIS,
                            features.RAD, features.TAX, features.PTRATIO, features.B, features.LSTAT]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    pred = model.predict(input_scaled)[0][0]
    price = round(float(pred) * 1000, 2)  # Convert to dollars

    return {"predicted_price_usd": price}
