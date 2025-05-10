# Import matplotlib configuration first
import matplotlib_config

from flask import Blueprint, render_template, request, send_file, redirect, url_for, current_app, flash
import os
from PIL import Image, ImageDraw
from ultralytics import YOLO
import boto3
import shutil

# Create blueprint
m1f_bp = Blueprint('m1f', __name__, template_folder='templates')

# Configuration
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs("transformation", exist_ok=True)

# Load YOLO model
model = YOLO(os.path.join('Models', 'best.pt'))

# S3 Config
s3_client = boto3.client(
    's3',
    aws_access_key_id='',
    aws_secret_access_key='',
    region_name='us-east-1'
)
s3_bucket_name = 'pcbdataset1'

def run_detection(image_path):
    model(source=image_path, imgsz=640, conf=0.25, save=True, save_txt=True, save_conf=True)
    output_dir = "runs/detect/predict"
    detected_images = [f for f in os.listdir(output_dir) if f.endswith(('.jpg', '.png'))]
    if detected_images:
        output_path = os.path.join(output_dir, detected_images[0])
        new_path = os.path.join(OUTPUT_FOLDER, detected_images[0])
        shutil.copy(output_path, new_path)
        return new_path
    return None

def create_internal_cutout(image):
    cutout_img = image.copy()
    draw = ImageDraw.Draw(cutout_img)
    width, height = image.size
    cutout_area = (width // 4, height // 4, 3 * width // 4, 3 * height // 4)
    draw.rectangle(cutout_area, fill="black")
    return cutout_img

def apply_transformations(image_path):
    original_img = Image.open(image_path)
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    transform_types = {
        "rotate": original_img.rotate(45),
        "crop": original_img.crop((50, 50, 300, 300)),
        "cutout": create_internal_cutout(original_img),
        "color_space": original_img.convert("L")
    }

    saved_paths = {}

    # Also save transformed images to static folder for easier serving
    for name, img in transform_types.items():
        # Save to transformation folder (for S3 upload)
        folder = os.path.join("transformation", name)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{image_name}_{name}.jpg")
        img.save(path)

        # Save to static folder (for web display)
        static_folder = os.path.join("static", "transforms")
        os.makedirs(static_folder, exist_ok=True)
        static_path = os.path.join(static_folder, f"{image_name}_{name}.jpg")
        img.save(static_path)

        # Use static path for web display
        saved_paths[name] = f"/static/transforms/{image_name}_{name}.jpg"

    return saved_paths

def upload_to_s3():
    for root, _, files in os.walk("transformation"):
        for file in files:
            file_path = os.path.join(root, file)
            s3_key = os.path.join("User_Transform_Data", os.path.relpath(file_path, "transformation"))
            s3_client.upload_file(file_path, s3_bucket_name, s3_key)

@m1f_bp.route("/", methods=["GET", "POST"])
def pcb_index():
    uploaded_img = None
    detection_output = None
    transformed_images = {}

    if request.method == "POST":
        file = request.files.get("image")
        if file:
            image_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(image_path)
            # Use URL paths that will work with Flask's static file serving
            uploaded_img = "/" + image_path

            if 'detect' in request.form:
                detection_result = run_detection(image_path)
                if detection_result:
                    detection_output = "/" + detection_result
                    # Automatically upload to S3 after detection
                    upload_to_s3()

            elif 'transform' in request.form:
                transform_results = apply_transformations(image_path)
                # The paths are already in URL format, no need to add another "/"
                transformed_images = transform_results
                # Automatically upload to S3 after transformation
                upload_to_s3()

            # We no longer need this since we're automatically uploading to S3

    return render_template("index2.html", uploaded_img=uploaded_img,
                           detection_output=detection_output,
                           transformed_images=transformed_images)