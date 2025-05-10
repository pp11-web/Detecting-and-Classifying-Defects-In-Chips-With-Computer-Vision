import tkinter as tk
from tkinter import filedialog, messagebox
import torch
import numpy as np
import pandas as pd
from torch import nn
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import boto3
import os
from CNN2 import CNN 

# Load the model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model = CNN()
cnn_model.load_state_dict(torch.load('Models/cnn_model.pth', map_location=DEVICE))
cnn_model.to(DEVICE)

# Initialize AWS S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id='',
    aws_secret_access_key='',
    region_name='us-east-1'
)
S3_BUCKET_NAME = 'waferdataset2'
S3_FOLDER_PATH = 'User_data'

# Label descriptions function
def label_description(i):
    descriptions = ["Center", "Donut", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "None"]
    return descriptions[i] if 0 <= i < len(descriptions) else "Unknown label"

# Function to display image and graph
def view_classify(img, ps):
    ps = ps.squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(10,20), ncols=2)
    ax1.imshow(img.resize_(1, 26, 26).numpy().squeeze(), cmap="gray")
    ax1.axis('off')
    ax2.barh(np.arange(9), ps)
    ax2.set_aspect(0.5)
    ax2.set_yticks(np.arange(9))
    ax2.set_yticklabels(['Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Near-full', 'Random', 'Scratch', 'None'])
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 15)
    plt.tight_layout()
    return fig

# Function to load and predict from CSV
def make_predict(file_path, model):
    global current_file_path
    current_file_path = file_path
    try:
        data = pd.read_csv(file_path).values
        tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            ps = model(tensor).detach().cpu().numpy()[0]
            probabilities = np.exp(ps)
            predicted_class = np.argmax(probabilities)
            label_desc = label_description(predicted_class)
        
        result_text.set(f"Predicted class: {predicted_class}\nDescription: {label_desc}")
        
        fig = view_classify(tensor.squeeze().cpu(), probabilities)
        
        for widget in plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to process the file:\n{e}")

# Function to open file dialog
def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        make_predict(file_path, cnn_model)

# Function to clear the saved_files directory
def clear_saved_folder():
    folder = "saved_files"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

# Function to upload the file to S3
def upload_to_s3():
    try:
        if not current_file_path:
            messagebox.showwarning("Warning", "No file selected to save and upload.")
            return

        output_dir = "saved_files"
        os.makedirs(output_dir, exist_ok=True)

        file_name = os.path.basename(current_file_path)
        save_path = os.path.join(output_dir, file_name)
        pd.read_csv(current_file_path).to_csv(save_path, index=False)

        s3_target_path = f"{S3_FOLDER_PATH}/{file_name}"
        s3_client.upload_file(save_path, S3_BUCKET_NAME, s3_target_path)

        messagebox.showinfo("Success", f"File uploaded to S3 at '{S3_BUCKET_NAME}/{s3_target_path}'.")
        clear_saved_folder()

    except Exception as e:
        messagebox.showerror("Error", f"Failed to save or upload the file:\n{e}")

# Function to exit the application
def exit_application():
    root.quit()

# GUI setup
root = tk.Tk()
root.title("Wafer Defect Prediction")
root.geometry("1000x1000")

# Frame for the title, button, and prediction label
top_frame = tk.Frame(root)
top_frame.pack(pady=10)

result_text = tk.StringVar()
result_text.set("Please select a CSV file to predict.")

# Prediction display frame
result_frame = tk.Frame(root)
result_frame.pack(pady=5)

# Plot frame
plot_frame = tk.Frame(root)
plot_frame.pack(pady=10, expand=True, fill="both")

# Bottom frame for the exit and upload buttons
bottom_frame = tk.Frame(root)
bottom_frame.pack(pady=10)

# GUI Elements
tk.Label(top_frame, text="Wafer Defect Prediction Tool", font=("Arial", 16)).pack()
tk.Button(top_frame, text="Select CSV File", command=open_file, font=("Arial", 12)).pack(pady=5)
tk.Label(result_frame, textvariable=result_text, wraplength=400, justify="left", font=("Arial", 12)).pack()

# Add an Upload to S3 button
upload_button = tk.Button(top_frame, text="Upload to S3", command=upload_to_s3, font=("Arial", 12))
upload_button.pack(side="left", padx=10)

# Add an Exit button
exit_button = tk.Button(top_frame, text="Exit", command=exit_application, font=("Arial", 12))
exit_button.pack(side="right", padx=10)

# Run the application
root.mainloop()
