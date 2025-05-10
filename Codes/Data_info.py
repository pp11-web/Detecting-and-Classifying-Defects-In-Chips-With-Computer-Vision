import boto3
import os
import random
from collections import defaultdict
from io import BytesIO
from PIL import Image
from tkinter import Tk, Label, Button, Entry, messagebox, ttk


def create_s3_client(aws_access_key, aws_secret_key, aws_region):
    """Create and return an S3 client."""
    return boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=aws_region
    )


def count_images(s3, bucket_name, raw_prefix, transformed_prefix):
    """
    Counts raw and transformed images in S3.

    Args:
        s3: Boto3 S3 client.
        bucket_name (str): Name of the S3 bucket.
        raw_prefix (str): Prefix for raw images.
        transformed_prefix (str): Prefix for transformed images.

    Returns:
        dict: Dictionary with class-wise raw and transformed counts.
    """
    counts = defaultdict(lambda: {'raw': 0, 'transformed': 0})

    # Count raw images
    raw_response = s3.list_objects_v2(Bucket=bucket_name, Prefix=raw_prefix)
    if 'Contents' in raw_response:
        for obj in raw_response['Contents']:
            file_key = obj['Key']
            if file_key.endswith(('.jpg', '.jpeg', '.png')):
                class_name = file_key[len(raw_prefix):].split('/')[0]
                counts[class_name]['raw'] += 1

    # Count transformed images
    transformed_response = s3.list_objects_v2(Bucket=bucket_name, Prefix=transformed_prefix)
    if 'Contents' in transformed_response:
        for obj in transformed_response['Contents']:
            file_key = obj['Key']
            if file_key.endswith(('.jpg', '.jpeg', '.png')):
                class_name = file_key[len(transformed_prefix):].split('/')[0]
                counts[class_name]['transformed'] += 1

    return dict(counts)


def apply_transformations(image):
    """
    Applies transformations to the image and returns them as a dictionary of in-memory images.

    Args:
        image (PIL.Image.Image): Original image.

    Returns:
        dict: Transformed images.
    """
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


def process_images(s3, bucket_name, raw_prefix, transformed_prefix, num_images):
    """
    Processes raw images, applies transformations, and uploads transformed images to S3.

    Args:
        s3: Boto3 S3 client.
        bucket_name (str): Name of the S3 bucket.
        raw_prefix (str): Prefix for raw images.
        transformed_prefix (str): Prefix for transformed images.
        num_images (int): Number of raw images to process per class.
    """
    raw_response = s3.list_objects_v2(Bucket=bucket_name, Prefix=raw_prefix)
    if 'Contents' not in raw_response:
        raise Exception("No raw images found in the specified path.")

    class_files = defaultdict(list)
    for obj in raw_response['Contents']:
        file_key = obj['Key']
        if file_key.endswith(('.jpg', '.jpeg', '.png')):
            class_name = file_key[len(raw_prefix):].split('/')[0]
            class_files[class_name].append(file_key)

    for class_name, files in class_files.items():
        selected_files = random.sample(files, min(num_images, len(files)))
        for file_key in selected_files:
            file_obj = s3.get_object(Bucket=bucket_name, Key=file_key)
            image = Image.open(BytesIO(file_obj['Body'].read()))

            # Apply transformations
            transformations = apply_transformations(image)

            # Upload transformed images
            for transform_name, transform_img in transformations.items():
                img_buffer = BytesIO()
                transform_img.save(img_buffer, format="JPEG")
                img_buffer.seek(0)
                transformed_key = f"{transformed_prefix}{class_name}/{transform_name}_{os.path.basename(file_key)}"
                s3.upload_fileobj(img_buffer, bucket_name, transformed_key)


# GUI Application
class S3ImageTransformerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("S3 Image Transformer")
        self.root.geometry("400x400")

        # Hardcoded Parameters
        self.bucket_name = 'pcbdataset1'
        self.raw_prefix = 'PCB_DATASET/images/'
        self.transformed_prefix = 'PCB_DATASET/images/transformed/'

        self.aws_access_key = ''
        self.aws_secret_key = ''
        self.aws_region = ''

        # UI Components
        self.label = Label(root, text="Enter the number of images to process per class:", wraplength=300, justify="center")
        self.label.pack(pady=10)

        self.num_images_entry = Entry(root)
        self.num_images_entry.pack(pady=5)

        self.progress_bar = ttk.Progressbar(root, orient="horizontal", mode="determinate", length=300)
        self.progress_bar.pack(pady=10)

        self.process_button = Button(root, text="Process Images", command=self.process_images)
        self.process_button.pack(pady=10)

        self.count_button = Button(root, text="Count Images", command=self.count_images)
        self.count_button.pack(pady=10)

    def process_images(self):
        try:
            num_images = int(self.num_images_entry.get())
            if num_images <= 0:
                raise ValueError("Number of images must be a positive integer.")

            s3 = create_s3_client(self.aws_access_key, self.aws_secret_key, self.aws_region)
            process_images(s3, self.bucket_name, self.raw_prefix, self.transformed_prefix, num_images)
            messagebox.showinfo("Success", "Images processed and uploaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def count_images(self):
        try:
            s3 = create_s3_client(self.aws_access_key, self.aws_secret_key, self.aws_region)
            counts = count_images(s3, self.bucket_name, self.raw_prefix, self.transformed_prefix)
            if counts:
                count_message = "Class-Wise Image Counts:\n\n"
                for class_name, data in counts.items():
                    count_message += f"Class: {class_name}\n"
                    count_message += f"  Raw: {data['raw']}\n"
                    count_message += f"  Transformed: {data['transformed']}\n\n"
                messagebox.showinfo("Image Counts", count_message)
            else:
                messagebox.showinfo("Image Counts", "No images found in the specified S3 paths.")
        except Exception as e:
            messagebox.showerror("Error", str(e))


# Run GUI
if __name__ == "__main__":
    root = Tk()
    app = S3ImageTransformerApp(root)
    root.mainloop()
