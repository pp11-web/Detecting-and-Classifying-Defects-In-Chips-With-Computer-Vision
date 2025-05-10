# Import matplotlib configuration first
import matplotlib_config

from flask import Blueprint, request, render_template, jsonify, send_from_directory
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boto3
from CNN2 import CNN
from datetime import datetime
import uuid

# Create blueprint
m2f_bp = Blueprint('m2f', __name__, template_folder='templates')

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load CNN model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model = CNN()
cnn_model.load_state_dict(torch.load('Models/cnn_model.pth', map_location=DEVICE))
cnn_model.to(DEVICE)
cnn_model.eval()

CLASS_LABELS = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "None"]

# AWS S3 Config
S3_BUCKET_NAME = 'waferdataset2'
S3_FOLDER_PATH = 'User_data'
s3_client = boto3.client(
    's3',
    aws_access_key_id='',
    aws_secret_access_key='',
    region_name='us-east-1'
)

@m2f_bp.route('/')
def wafer_index():
    return render_template('index3.html')

@m2f_bp.route('/process_wafer', methods=['POST'])
def process_wafer():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    filename = f"{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    try:
        data = pd.read_csv(file_path).values
        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            ps = cnn_model(tensor).detach().cpu().numpy()[0]
            probabilities = np.exp(ps)
            predicted_class = np.argmax(probabilities)
            label_desc = CLASS_LABELS[predicted_class]
            confidence = probabilities[predicted_class] / np.sum(probabilities)

        # Always upload to S3
        s3_key = f"{S3_FOLDER_PATH}/{filename}"
        s3_client.upload_file(file_path, S3_BUCKET_NAME, s3_key)

        # Create and save plots - using a safer approach
        result_plot_path = os.path.join(RESULT_FOLDER, f"{filename}.png")
        wafer_plot_path = os.path.join(RESULT_FOLDER, f"{filename}_wafer.png")

        # Create probability distribution plot
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.barh(CLASS_LABELS, probabilities, color='green')
        ax1.set_xlabel("Probability")
        ax1.set_title("Class Prediction Distribution")
        ax1.invert_yaxis()
        fig1.tight_layout()

        # Save and close the figure
        fig1.savefig(result_plot_path)
        plt.close(fig1)

        # Create wafer visualization with custom colormap and legend
        fig2, ax2 = plt.subplots(figsize=(8, 7))  # Increased figure size to accommodate legend

        # data is already a NumPy array, no need to use .values
        wafer_data = data  # data is already a NumPy array from pd.read_csv().values

        # Create a custom colormap for better interpretation
        from matplotlib.colors import LinearSegmentedColormap

        # Define custom colormap: purple (background) -> green (silicon) -> yellow (defect)
        colors = [(0.5, 0, 0.5),    # purple (background)
                 (0, 0.8, 0),      # green (silicon)
                 (1, 1, 0)]        # yellow (defect)

        custom_cmap = LinearSegmentedColormap.from_list('wafer_cmap', colors, N=256)

        # Create a heatmap of the wafer data
        im = ax2.imshow(wafer_data, cmap=custom_cmap)
        ax2.set_title(f"Wafer Visualization - {label_desc}", fontsize=14)

        # Add colorbar
        cbar = fig2.colorbar(im, ax=ax2)

        # Add legend as text annotations
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc=(0.5, 0, 0.5), label='Background'),
            plt.Rectangle((0, 0), 1, 1, fc=(0, 0.8, 0), label='Silicon'),
            plt.Rectangle((0, 0), 1, 1, fc=(1, 1, 0), label='Defect')
        ]

        # Add the legend below the plot
        ax2.legend(handles=legend_elements, loc='upper center',
                  bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=12)

        # Remove axis ticks for cleaner look
        ax2.set_xticks([])
        ax2.set_yticks([])

        # Add grid for better visualization
        ax2.grid(False)

        # Add a note about the visualization
        ax2.text(0.5, -0.15,
                "Note: Colors represent intensity values from wafer scan data",
                horizontalalignment='center', fontsize=10,
                transform=ax2.transAxes, style='italic')

        fig2.tight_layout()

        # Save and close the wafer figure
        fig2.savefig(wafer_plot_path)
        plt.close(fig2)

        # Upload the plots to S3
        plot_s3_key = f"{S3_FOLDER_PATH}/plots/{filename}.png"
        wafer_s3_key = f"{S3_FOLDER_PATH}/plots/{filename}_wafer.png"
        try:
            s3_client.upload_file(result_plot_path, S3_BUCKET_NAME, plot_s3_key)
            s3_client.upload_file(wafer_plot_path, S3_BUCKET_NAME, wafer_s3_key)
            print(f"Plots uploaded to S3: {plot_s3_key}, {wafer_s3_key}")
        except Exception as e:
            print(f"Error uploading plots to S3: {e}")

        result_image_url = f"/static/results/{filename}.png"
        wafer_image_url = f"/static/results/{filename}_wafer.png"
        return jsonify({
            'result': f"Predicted Class: {predicted_class} - {label_desc}",
            'confidence': round(float(confidence), 4),
            'defect_type': label_desc,
            'chart_image_url': result_image_url,
            'wafer_image_url': wafer_image_url,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'ask_feedback': True
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@m2f_bp.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    feedback = request.form.get('feedback', 'unknown')
    original_image = request.form.get('original_image', 'N/A')
    defect_type = request.form.get('defect_type', 'N/A')

    # Log the feedback
    print(f"Received feedback: {feedback}, for {original_image} - {defect_type}")

    # Import the feedback counter module
    from feedback_counter import update_feedback_count, generate_feedback_report

    # Update feedback count in S3
    success = update_feedback_count(feedback, defect_type, original_image)

    # Generate a new report
    if success:
        generate_feedback_report()
        message = 'Thank you for your feedback! Your response has been recorded.'
    else:
        message = 'Thank you for your feedback! (Note: There was an issue saving to S3)'

    return jsonify({'success': True, 'message': message})
