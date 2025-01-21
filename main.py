from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the synthetic dataset
file_path = "synthetic_manufacturing_data.csv"  # Replace with the actual path
data = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(data.head())

# EDA
print(data.isnull().sum())

print(data.describe())

print(data["Downtime_Flag"].value_counts())

# Preprocessing
X = data[["Temperature", "Run_Time"]]  # Features
y = data["Downtime_Flag"]             # Target variable


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Predictive Model

# Initialize and train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the Model

# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

# Make Prediction
# Example input
new_data = pd.DataFrame({"Temperature": [88.5], "Run_Time": [155]})

# Scale the input data
new_data_scaled = scaler.transform(new_data)

# Predict downtime
downtime_prediction = model.predict(new_data_scaled)
confidence = model.predict_proba(new_data_scaled).max()

print(f"Downtime Prediction: {'Yes' if downtime_prediction[0] == 1 else 'No'}")
print(f"Confidence: {confidence:.2f}")
