import streamlit as st
import boto3
import os
import random
from collections import defaultdict
from io import BytesIO
from PIL import Image

# AWS S3 Configuration
BUCKET_NAME = 'pcbdataset1'
RAW_PREFIX = 'PCB_DATASET/images/'
TRANSFORMED_PREFIX = 'PCB_DATASET/images/transformed/'

# Initialize AWS S3 client (Ensure credentials are properly set in AWS CLI)
s3_client = boto3.client(
    's3',
    aws_access_key_id='',
    aws_secret_access_key='',
    region_name='us-east-1'
)

# Function to Count Images in S3
def count_images():
    counts = defaultdict(lambda: {'raw': 0, 'transformed': 0})

    # Count raw images
    raw_response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=RAW_PREFIX)
    if 'Contents' in raw_response:
        for obj in raw_response['Contents']:
            file_key = obj['Key']
            if file_key.endswith(('.jpg', '.jpeg', '.png')):
                class_name = file_key[len(RAW_PREFIX):].split('/')[0]
                counts[class_name]['raw'] += 1

    # Count transformed images
    transformed_response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=TRANSFORMED_PREFIX)
    if 'Contents' in transformed_response:
        for obj in transformed_response['Contents']:
            file_key = obj['Key']
            if file_key.endswith(('.jpg', '.jpeg', '.png')):
                class_name = file_key[len(TRANSFORMED_PREFIX):].split('/')[0]
                counts[class_name]['transformed'] += 1

    return dict(counts)

# Function to Apply Transformations
def apply_transformations(image):
    width, height = image.size

    # Rotation
    rotated_img = image.rotate(random.choice([90, 270]))

    # Resizing
    resized_img = image.resize((256, 256))

    # Cropping
    crop_size = int(0.8 * min(width, height))
    left = random.randint(0, width - crop_size)
    top = random.randint(0, height - crop_size)
    cropped_img = image.crop((left, top, left + crop_size, top + crop_size))

    # Cutout
    cutout_img = image.copy()
    cutout_size = int(0.2 * min(width, height))
    cutout_x = random.randint(0, width - cutout_size)
    cutout_y = random.randint(0, height - cutout_size)
    cutout_img.paste((0, 0, 0), (cutout_x, cutout_y, cutout_x + cutout_size, cutout_y + cutout_size))

    return {
        "rotated": rotated_img,
        "resized": resized_img,
        "cropped": cropped_img,
        "cutout": cutout_img
    }

# Function to Process & Upload Images
def process_images(num_images):
    raw_response = s3_client.list_objects_v2(Bucket=BUCKET_NAME, Prefix=RAW_PREFIX)
    if 'Contents' not in raw_response:
        st.error("No raw images found in the specified path.")
        return

    class_files = defaultdict(list)
    for obj in raw_response['Contents']:
        file_key = obj['Key']
        if file_key.endswith(('.jpg', '.jpeg', '.png')):
            class_name = file_key[len(RAW_PREFIX):].split('/')[0]
            class_files[class_name].append(file_key)

    progress_bar = st.progress(0)

    total_classes = len(class_files)
    processed = 0

    for class_name, files in class_files.items():
        selected_files = random.sample(files, min(num_images, len(files)))
        for file_key in selected_files:
            file_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=file_key)
            image = Image.open(BytesIO(file_obj['Body'].read()))

            # Apply transformations
            transformations = apply_transformations(image)

            # Upload transformed images
            for transform_name, transform_img in transformations.items():
                img_buffer = BytesIO()
                transform_img.save(img_buffer, format="JPEG")
                img_buffer.seek(0)
                transformed_key = f"{TRANSFORMED_PREFIX}{class_name}/{transform_name}_{os.path.basename(file_key)}"
                s3_client.upload_fileobj(img_buffer, BUCKET_NAME, transformed_key)

        processed += 1
        progress_bar.progress(processed / total_classes)

    st.success("Images processed and uploaded successfully!")

# Streamlit UI
st.title("S3 Image Transformer")

# Image Count Button
if st.button("Count Images in S3"):
    with st.spinner("Counting images..."):
        counts = count_images()
    
    if counts:
        st.subheader("Class-wise Image Counts:")
        for class_name, data in counts.items():
            st.write(f"**{class_name}**")
            st.write(f"📌 Raw: {data['raw']}")
            st.write(f"📌 Transformed: {data['transformed']}")
    else:
        st.warning("No images found in S3.")

# Number of Images Input
num_images = st.number_input("Enter the number of images to process per class:", min_value=1, step=1)

# Process Images Button
if st.button("Process Images"):
    if num_images > 0:
        with st.spinner("Processing images..."):
            process_images(num_images)
    else:
        st.warning("Please enter a valid number of images.")
