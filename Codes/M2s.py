import streamlit as st
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boto3
import os
from CNN2 import CNN  # Import your CNN model

# Load the model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model = CNN()
cnn_model.load_state_dict(torch.load('Models/cnn_model.pth', map_location=DEVICE))
cnn_model.to(DEVICE)
cnn_model.eval()

# AWS S3 Configuration
S3_BUCKET_NAME = 'waferdataset2'
S3_FOLDER_PATH = 'User_data'

# Initialize AWS S3 client (Ensure credentials are properly set in AWS CLI)
s3_client = boto3.client(
    's3',
    aws_access_key_id='',
    aws_secret_access_key='',
    region_name='us-east-1'
)

# Class Labels
CLASS_LABELS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "None"]

# Function to Predict & Display Results
def make_predict(file_path, model):
    try:
        data = pd.read_csv(file_path).values
        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            ps = model(tensor).detach().cpu().numpy()[0]
            probabilities = np.exp(ps)
            predicted_class = np.argmax(probabilities)
            label_desc = CLASS_LABELS[predicted_class] if 0 <= predicted_class < len(CLASS_LABELS) else "Unknown"

        return probabilities, predicted_class, label_desc

    except Exception as e:
        st.error(f"Error processing the file: {e}")
        return None, None, None

# Function to Upload to S3
def upload_to_s3(file_path, file_name):
    try:
        s3_target_path = f"{S3_FOLDER_PATH}/{file_name}"
        s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_target_path)
        st.success(f"File uploaded to S3 at '{S3_BUCKET_NAME}/{s3_target_path}'.")
    except Exception as e:
        st.error(f"Failed to upload file to S3: {e}")

# Streamlit UI
st.title("Wafer Defect Prediction Tool")

uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

if uploaded_file:
    # Save uploaded file locally
    temp_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Make Prediction
    probabilities, predicted_class, label_desc = make_predict(temp_path, cnn_model)

    if probabilities is not None:
        st.write(f"### Predicted Class: {predicted_class}")
        st.write(f"**Description:** {label_desc}")

        # Display Probability Bar Chart
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(CLASS_LABELS, probabilities, color='blue')
        ax.set_xlabel("Probability")
        ax.set_title("Class Probability Distribution")
        ax.invert_yaxis()
        st.pyplot(fig)

        # Upload to S3 Button
        if st.button("Upload to S3"):
            upload_to_s3(temp_path, uploaded_file.name)
