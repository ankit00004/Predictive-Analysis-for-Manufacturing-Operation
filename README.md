# Predictive-Analysis-for-Manufacturing-Operation

Predictive Analysis for Manufacturing Operations
Project Overview
This project aims to predict machine downtime or production defects in a manufacturing setting using machine learning. The model is trained on synthetic manufacturing data and exposed via a RESTful API built with FastAPI.

Features
Upload Data: Upload CSV files containing manufacturing data (e.g., Temperature, Run_Time).
Train Model: Train a predictive model (Logistic Regression) on the uploaded data.
Make Predictions: Predict machine downtime based on input features (e.g., Temperature, Run_Time).
API Endpoints
POST /upload: Upload a CSV file containing manufacturing data.
POST /train: Train the model on the uploaded data.
POST /predict: Make predictions on new data (e.g., Temperature, Run_Time).
Requirements
Python 3.7+
FastAPI
Uvicorn
scikit-learn
pandas
Installation
Clone the repository:
bash
Copy
Edit
git clone https://github.com/yourusername/repository-name.git
Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Run the FastAPI server:
bash
Copy
Edit
uvicorn app:app --reload
Example Usage
Upload Data:
URL: POST http://127.0.0.1:8000/upload
Body: Form-data with file key and dataset as value.
Train Model:
URL: POST http://127.0.0.1:8000/train
Make Prediction:
URL: POST http://127.0.0.1:8000/predict
Body (JSON):
json
Copy
Edit
{
  "Temperature": 88.5,
  "Run_Time": 155
}
