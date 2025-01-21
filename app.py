from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os

# FastAPI application
app = FastAPI()

# Directory to save uploaded files and models
UPLOAD_DIR = "uploads"
MODEL_DIR = "models"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Global variables
model = None
X_columns = []

# Dataset schema for predictions
class PredictionInput(BaseModel):
    Temperature: float
    Run_Time: float

# Helper function to train the model
def train_model(data):
    global model, X_columns

    # Convert Downtime_Flag to 1 and 0 (mapping 'Yes' to 1, 'No' to 0)
    data['Downtime_Flag'] = data['Downtime_Flag'].map({'Yes': 1, 'No': 0})

    # Extract features and target
    X = data[["Temperature", "Run_Time"]]
    y = data["Downtime_Flag"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Save the model and columns
    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
    X_columns = X.columns.tolist()
    joblib.dump(X_columns, os.path.join(MODEL_DIR, "columns.pkl"))

    return {"accuracy": accuracy, "f1_score": f1}

# Endpoint to upload data
@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    try:
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())

        # Load the data
        data = pd.read_csv(file_path)

        # Validate required columns
        required_columns = ["Machine_ID", "Temperature", "Run_Time", "Downtime_Flag"]
        if not all(col in data.columns for col in required_columns):
            raise HTTPException(status_code=400, detail=f"Missing required columns: {required_columns}")

        return {"message": "File uploaded successfully", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to train the model
@app.post("/train")
async def train():
    try:
        # Check for uploaded files
        files = os.listdir(UPLOAD_DIR)
        if not files:
            raise HTTPException(status_code=400, detail="No uploaded files found")

        # Load the most recent file
        file_path = os.path.join(UPLOAD_DIR, files[-1])
        data = pd.read_csv(file_path)

        # Train the model
        metrics = train_model(data)
        return {"message": "Model trained successfully", "metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to make predictions
@app.post("/predict")
async def predict(input_data: PredictionInput):
    try:
        global model, X_columns

        # Load the model and columns if not already loaded
        if model is None:
            model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
            X_columns = joblib.load(os.path.join(MODEL_DIR, "columns.pkl"))

        # Prepare the input data for prediction
        input_df = pd.DataFrame([input_data.dict()])

        # Ensure columns match the training data
        input_df = input_df[X_columns]

        # Make prediction
        prediction = model.predict(input_df)[0]
        downtime = 'Yes' if prediction == 1 else 'No'  # Map numeric prediction to 'Yes' or 'No'
        confidence = max(model.predict_proba(input_df)[0])

        return {"Downtime": downtime, "Confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Instructions for testing
@app.get("/")
def read_root():
    return {"message": "API is running. Use /upload to upload data, /train to train the model, and /predict to make predictions."}
