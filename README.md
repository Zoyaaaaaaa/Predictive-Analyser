# Predictive Analyser

This repository contains a FastAPI-based web application that allows you to upload a dataset, train a logistic regression model, and make predictions about machine downtime based on temperature and runtime.

---
![image](https://github.com/user-attachments/assets/f8f23147-4f17-4a4c-bc13-7ce70bbb1043)

## Features

1. **Upload Dataset**: Upload a CSV file containing machine data.
2. **Train Model**: Train a logistic regression model on the uploaded data.
3. **Predict Downtime**: Predict whether a machine will experience downtime based on input parameters.

---

## Installation

Follow these steps to set up and run the application:

### Prerequisites

- Python 3.8 or later
- `pip` package manager

### Clone the Repository

```bash
$ git clone <repository-url>
$ cd <repository-folder>
```

### Install Dependencies

Install the required Python packages using the following command:

```bash
$ pip install -r requirements.txt
```

### File Structure

- `main.py`: The FastAPI backend script.
- `app_ui.py`: The Streamlit front-end script.
- `dataset.csv`: Sample dataset for testing.
- `uploads/`: Directory to store uploaded CSV files.
- `models/`: Directory to store the trained model and metadata.

---

## Usage

### Step 1: Start the FastAPI Server

Run the backend FastAPI server using:

```bash
$ uvicorn app:app --reload
```

By default, the server runs at `http://127.0.0.1:8000`.

### Step 2: Start the Streamlit Front-End

In another terminal, start the Streamlit UI:

```bash
$ streamlit run streamlit_app.py
```

The front-end will open in your default web browser.

### Step 3: Upload the Dataset

- Use the `dataset.csv` file provided for testing.
- Navigate to the "Upload Data" section of the Streamlit app.
- Upload the `dataset.csv` file.

### Step 4: Train the Model

- Click the "Train Model" button in the Streamlit app.
- Once training is complete, the accuracy and F1 score will be displayed.

### Step 5: Make Predictions

- Navigate to the "Make Predictions" section.
- Enter `Temperature` and `Run Time` values.
- Click "Predict" to see the downtime prediction and confidence score.

---

## Example Dataset

The `dataset.csv` file provided has the following structure:

```csv
Machine_ID,Temperature,Run_Time,Downtime_Flag
1,75.3,120,Yes
2,80.5,150,No
3,78.1,100,Yes
4,85.2,200,No
5,70.8,130,Yes
```

- `Machine_ID`: Unique identifier for the machine.
- `Temperature`: Operating temperature of the machine.
- `Run_Time`: Duration the machine has been running.
- `Downtime_Flag`: Indicates if the machine experienced downtime (`Yes`/`No`).

---

## API Endpoints

### Root Endpoint

- **URL**: `/`
- **Method**: `GET`
- **Description**: Returns a welcome message and instructions for using the API.

### Upload Data Endpoint

- **URL**: `/upload`
- **Method**: `POST`
- **Description**: Upload a CSV file for model training.

### Train Model Endpoint

- **URL**: `/train`
- **Method**: `POST`
- **Description**: Train the logistic regression model using the uploaded data.

### Predict Endpoint

- **URL**: `/predict`
- **Method**: `POST`
- **Request Body**:

```json
{
  "Temperature": 75.0,
  "Run_Time": 120.0
}
```

- **Response**:

```json
{
  "Downtime": "Yes",
  "Confidence": 0.85
}
```


