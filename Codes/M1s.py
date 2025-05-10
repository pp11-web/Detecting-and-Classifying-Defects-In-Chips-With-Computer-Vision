import streamlit as st
import os
import boto3
from PIL import Image, ImageDraw
from ultralytics import YOLO
import shutil

# Load YOLO Model
best_model_path = os.path.join('Models', 'best.pt')
model = YOLO(best_model_path)

# Transformation Directory and S3 details
transformation_dir = 'transformation'
output_dir = "runs/detect/predict"  # YOLO default output directory
s3_bucket_name = 'pcbdataset1'  # Replace with your S3 bucket name

# Initialize boto3 S3 client (Ensure credentials are properly set up in AWS CLI)
s3_client = boto3.client(
    's3',
    aws_access_key_id='',
    aws_secret_access_key='',
    region_name='us-east-1'
)

# Ensure transformation directory exists
os.makedirs(transformation_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

st.title("YOLO Image Detection & Transformation")

# Upload image
uploaded_file = st.file_uploader("Upload an image (JPG, PNG)", type=["jpg", "jpeg", "png"])

def run_detection(image_path):
    """ Run YOLO detection on an image and return the output path """
    model(source=image_path, imgsz=640, conf=0.25, save=True, save_txt=True, save_conf=True)

    # Find the output image
    detected_images = [f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.png'))]
    
    if detected_images:
        detected_image_path = os.path.join(output_dir, detected_images[0])
        return detected_image_path
    return None

def create_internal_cutout(image):
    """ Create an internal cutout transformation """
    cutout_img = image.copy()
    draw = ImageDraw.Draw(cutout_img)
    width, height = image.size
    cutout_area = (width // 4, height // 4, 3 * width // 4, 3 * height // 4)
    draw.rectangle(cutout_area, fill="black")
    return cutout_img

def apply_transformations(image_path):
    """ Apply multiple transformations and save them """
    original_img = Image.open(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    transform_types = {
        "rotate": original_img.rotate(45),
        "crop": original_img.crop((50, 50, 300, 300)),
        "cutout": create_internal_cutout(original_img),
        "color_space": original_img.convert("L")
    }

    transformed_images = {}
    
    for transform_name, transformed_img in transform_types.items():
        transform_folder = os.path.join(transformation_dir, transform_name)
        os.makedirs(transform_folder, exist_ok=True)
        
        transformed_img_path = os.path.join(transform_folder, f"{image_name}_{transform_name}.jpg")
        transformed_img.save(transformed_img_path)
        transformed_images[transform_name] = transformed_img
    
    return transformed_images

def upload_to_s3():
    """ Upload transformed images to S3 """
    try:
        for root, dirs, files in os.walk(transformation_dir):
            for file in files:
                file_path = os.path.join(root, file)
                s3_key = os.path.join("User_Transform_Data", os.path.relpath(file_path, transformation_dir))
                s3_client.upload_file(file_path, s3_bucket_name, s3_key)
        st.success("Transformation folder uploaded to the 'Transform_Data' folder in S3 successfully.")
    except Exception as e:
        st.error(f"An error occurred during upload: {e}")

if uploaded_file:
    # Save uploaded image temporarily
    temp_image_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    
    if st.button("Run Detection"):
        detected_image_path = run_detection(temp_image_path)
        
        if detected_image_path:
            st.image(detected_image_path, caption="Detection Output", use_container_width=True)
            st.success("Detection completed. Results saved.")
        else:
            st.error("Detection failed or no output image found.")

    if st.button("Apply Transformations"):
        transformed_images = apply_transformations(temp_image_path)
        
        st.write("### Transformed Images:")
        cols = st.columns(4)
        for idx, (name, img) in enumerate(transformed_images.items()):
            with cols[idx]:
                st.image(img, caption=name, use_container_width=True)

    if st.button("Upload Transformations to S3"):
        upload_to_s3()
