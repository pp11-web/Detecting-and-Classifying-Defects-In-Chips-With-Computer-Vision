import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps, ImageDraw
import os
import boto3
from ultralytics import YOLO

# Load your pre-trained model
best_model_path = os.path.join('Models', 'best.pt')
model = YOLO(best_model_path)

# Define transformation directory and S3 bucket details
transformation_dir = 'transformation'
s3_bucket_name = 'pcbdataset1'  # Replace with your S3 bucket name

# Initialize boto3 S3 client with hardcoded credentials (for testing purposes only)
s3_client = boto3.client(
    's3',
    aws_access_key_id='',
    aws_secret_access_key='',
    region_name='us-east-1'
)

# Initialize the main window
window = tk.Tk()
window.title("YOLO Image Detection & Transformation")
window.geometry("1000x900")

# Create transformation directory if it doesn't exist
if not os.path.exists(transformation_dir):
    os.makedirs(transformation_dir)

# # Function to upload the transformation folder to S3
# def upload_to_s3():
#     try:
#         for root, dirs, files in os.walk(transformation_dir):
#             for file in files:
#                 file_path = os.path.join(root, file)
#                 s3_key = os.path.relpath(file_path, transformation_dir)  # Key for S3 bucket
#                 s3_client.upload_file(file_path, s3_bucket_name, s3_key)
#         messagebox.showinfo("Success", "Transformation folder uploaded to S3 successfully.")
#     except Exception as e:
#         messagebox.showerror("Error", f"An error occurred during upload: {e}")

def upload_to_s3():
    try:
        for root, dirs, files in os.walk(transformation_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Add "Transform_Data" as a prefix in the S3 key
                s3_key = os.path.join("User_Transform_Data", os.path.relpath(file_path, transformation_dir))
                
                # Upload file to S3
                s3_client.upload_file(file_path, s3_bucket_name, s3_key)
        
        messagebox.showinfo("Success", "Transformation folder uploaded to the 'Transform_Data' folder in S3 successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred during upload: {e}")

# Display the selected image in the GUI
def display_image(image_path, label_widget):
    img = Image.open(image_path)
    img = img.resize((200, 200))  # Resize for display purposes
    img_tk = ImageTk.PhotoImage(img)
    label_widget.config(image=img_tk)
    label_widget.image = img_tk

# Display the classification results
def display_classification(image_path):
    label_file = os.path.join('/Users/prayagpurani/Desktop/298/Codes/runs/detect/predict/labels', 
                              os.path.splitext(os.path.basename(image_path))[0] + '.txt')
    
    if os.path.exists(label_file):
        with open(label_file, 'r') as file:
            labels = file.readlines()
        label_text = "\n".join([f"Class: {line.split()[0]}, Confidence: {line.split()[-1]}" for line in labels])
        labels_text.config(text=label_text)
    else:
        labels_text.config(text="No labels found.")

# Run YOLO model on the uploaded image
def run_detection(image_path):
    try:
        metrics = model(source=image_path, imgsz=640, conf=0.25, save=True, save_txt=True, save_conf=True)
        messagebox.showinfo("Success", "Detection completed and results saved.")
        
        result_image_path = os.path.join('/Users/prayagpurani/Desktop/298/Codes/runs/detect/predict', os.path.basename(image_path))
        display_image(result_image_path, img_label)
        display_classification(image_path)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# Apply transformations, including saving the original image and labels
def apply_transformations(image_path):
    original_img = Image.open(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # Save the original image to the 'transformation/original' folder
    original_folder = os.path.join(transformation_dir, "original")
    os.makedirs(original_folder, exist_ok=True)
    original_img_path = os.path.join(original_folder, f"{image_name}_original.jpg")
    original_img.save(original_img_path)
    
    # Copy the original labels to 'transformation/original/labels'
    label_src = os.path.join('/Users/prayagpurani/Desktop/298/Codes/runs/detect/predict/labels', f"{image_name}.txt")
    original_label_folder = os.path.join(original_folder, "labels")
    os.makedirs(original_label_folder, exist_ok=True)
    original_label_path = os.path.join(original_label_folder, f"{image_name}_original.txt")
    
    if os.path.exists(label_src):
        with open(label_src, 'r') as src_file:
            with open(original_label_path, 'w') as dst_file:
                dst_file.write(src_file.read())
    else:
        with open(original_label_path, 'w') as label_file:
            label_file.write("No labels generated.")

    # Define transformations
    transformations = {
        "rotate": original_img.rotate(45),
        "crop": original_img.crop((50, 50, 300, 300)),
        "cutout": create_internal_cutout(original_img),  # Use the custom internal cutout function
        "color_space": original_img.convert("L")  # Convert to grayscale
    }
    
    for transform_name, transformed_img in transformations.items():
        transform_folder = os.path.join(transformation_dir, transform_name)
        os.makedirs(transform_folder, exist_ok=True)
        
        transformed_img_path = os.path.join(transform_folder, f"{image_name}_{transform_name}.jpg")
        transformed_img.save(transformed_img_path)
        
        label_dst_folder = os.path.join(transform_folder, "labels")
        os.makedirs(label_dst_folder, exist_ok=True)
        label_dst = os.path.join(label_dst_folder, f"{image_name}_{transform_name}.txt")
        
        if os.path.exists(label_src):
            with open(label_src, 'r') as src_file:
                with open(label_dst, 'w') as dst_file:
                    dst_file.write(src_file.read())
        else:
            with open(label_dst, 'w') as label_file:
                label_file.write("No labels generated.")

        if transform_name == "rotate":
            display_image(transformed_img_path, rotate_label)
        elif transform_name == "crop":
            display_image(transformed_img_path, crop_label)
        elif transform_name == "cutout":
            display_image(transformed_img_path, cutout_label)
        elif transform_name == "color_space":
            display_image(transformed_img_path, color_label)

    messagebox.showinfo("Transformation", "Transformations completed and displayed.")

# Function to create an internal cutout by drawing a black rectangle inside the image
def create_internal_cutout(image):
    cutout_img = image.copy()
    draw = ImageDraw.Draw(cutout_img)
    width, height = image.size
    cutout_area = (width // 4, height // 4, 3 * width // 4, 3 * height // 4)
    draw.rectangle(cutout_area, fill="black")
    return cutout_img

# Function to upload an image file
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        display_image(file_path, img_label)
        run_detection(file_path)
        apply_transformations(file_path)

# Function to exit the application
def exit_application():
    window.destroy()  # Close the Tkinter window
    os._exit(0)       # Force exit the program (works on macOS)

# Upload button
upload_button = tk.Button(window, text="Upload Image", command=upload_image, font=("Arial", 14))
upload_button.pack(pady=20)

# Upload to S3 button
upload_s3_button = tk.Button(window, text="Upload Transformation Folder to S3", command=upload_to_s3, font=("Arial", 14))
upload_s3_button.pack(pady=10)

# Label to display the main detected image
img_label = tk.Label(window)
img_label.pack()

# Label to display classifications and confidence
labels_text = tk.Label(window, text="", font=("Arial", 12), justify="left")
labels_text.pack(pady=10)

# Frame to display transformed images horizontally
transformation_frame = tk.Frame(window)
transformation_frame.pack(pady=20)

# Labels to display each transformation in a row
rotate_label = tk.Label(transformation_frame)
rotate_label.grid(row=0, column=0, padx=10)
rotate_text = tk.Label(transformation_frame, text="Rotated Image")
rotate_text.grid(row=1, column=0)

crop_label = tk.Label(transformation_frame)
crop_label.grid(row=0, column=1, padx=10)
crop_text = tk.Label(transformation_frame, text="Cropped Image")
crop_text.grid(row=1, column=1)

cutout_label = tk.Label(transformation_frame)
cutout_label.grid(row=0, column=2, padx=10)
cutout_text = tk.Label(transformation_frame, text="Internal Cutout Image")
cutout_text.grid(row=1, column=2)

color_label = tk.Label(transformation_frame)
color_label.grid(row=0, column=3, padx=10)
color_text = tk.Label(transformation_frame, text="Grayscale Image")
color_text.grid(row=1, column=3)

# Exit button
exit_button = tk.Button(window, text="Exit", command=exit_application, font=("Arial", 14))
exit_button.pack(pady=20)

# Start the Tkinter event loop
window.mainloop()