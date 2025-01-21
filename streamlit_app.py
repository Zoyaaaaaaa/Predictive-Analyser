import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000"  # Adjust if your FastAPI is running elsewhere

# Upload data
st.title("Upload Data for Model Training")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    files = {'file': uploaded_file.getvalue()}
    response = requests.post(f"{API_URL}/upload", files=files)
    if response.status_code == 200:
        st.success("File uploaded successfully")
    else:
        st.error(f"Error: {response.text}")

# Train model
if st.button("Train Model"):
    response = requests.post(f"{API_URL}/train")
    if response.status_code == 200:
        st.success("Model trained successfully")
        metrics = response.json()
        st.write("Accuracy:", metrics['metrics']['accuracy'])
        st.write("F1 Score:", metrics['metrics']['f1_score'])
    else:
        st.error(f"Error: {response.text}")

# Make predictions
st.title("Make Predictions")
temperature = st.number_input("Temperature")
run_time = st.number_input("Run Time")
if st.button("Predict"):
    input_data = {"Temperature": temperature, "Run_Time": run_time}
    response = requests.post(f"{API_URL}/predict", json=input_data)
    if response.status_code == 200:
        prediction = response.json()
        st.write("Prediction:", prediction['Downtime'])
        st.write("Confidence:", prediction['Confidence'])
    else:
        st.error(f"Error: {response.text}")
