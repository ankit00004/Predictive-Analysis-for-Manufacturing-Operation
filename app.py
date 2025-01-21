from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Initialize the FastAPI app
app = FastAPI()

# Global variables
data = None
model = None
scaler = None

# Upload Endpoint


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global data
    if file.filename.endswith(".csv"):
        data = pd.read_csv(file.file)
        return {"message": "File uploaded successfully"}
    return {"error": "Invalid file format. Please upload a CSV file."}

# Train Endpoint


@app.post("/train")
def train_model():
    global data, model, scaler
    if data is None:
        return {"error": "No data uploaded. Use /upload endpoint first."}

    # Features and target
    X = data[["Temperature", "Run_Time"]]
    y = data["Downtime_Flag"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_scaled, y)

    # Return a success message
    return {"message": "Model trained successfully"}

# Predict Endpoint


class PredictionInput(BaseModel):
    Temperature: float
    Run_Time: float


@app.post("/predict")
def predict(input_data: PredictionInput):
    global model, scaler
    if model is None:
        return {"error": "Model not trained. Use /train endpoint first."}

    # Prepare input data
    input_df = pd.DataFrame([input_data.dict()])
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    confidence = model.predict_proba(input_scaled).max()

    return {
        "Downtime": "Yes" if prediction == 1 else "No",
        "Confidence": round(confidence, 2)
    }
