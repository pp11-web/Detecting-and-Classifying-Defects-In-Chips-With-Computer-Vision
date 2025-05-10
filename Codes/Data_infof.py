from flask import Blueprint, render_template, request, jsonify, flash
import boto3
import os
import json

# Create blueprint
data_info_bp = Blueprint('data_info', __name__, template_folder='templates')

# AWS S3 Configuration
AWS_ACCESS_KEY = ''
AWS_SECRET_KEY = ''
AWS_REGION = ''

# Define all buckets to monitor
S3_BUCKETS = [
    'pcbdataset1',  # PCB images bucket
    'waferdataset2'  # Wafer data bucket
]

# Initialize S3 client
def create_s3_client(access_key, secret_key, region):
    return boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region
    )

# Function to count all files in all buckets and subfolders in S3
def count_all_files(s3_client, buckets):
    try:
        # Initialize hierarchical counts dictionary
        hierarchy = {}
        grand_total = 0

        # Process each bucket
        for bucket_name in buckets:
            # Initialize bucket in hierarchy
            hierarchy[bucket_name] = {
                'total': 0,
                'subfolders': {}
            }

            # Get all objects in the bucket
            paginator = s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=bucket_name)

            # Process each page of results
            for page in page_iterator:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']

                        # Skip empty keys
                        if not key:
                            continue

                        # Increment bucket total
                        hierarchy[bucket_name]['total'] += 1
                        grand_total += 1

                        # Process folder structure
                        current_level = hierarchy[bucket_name]['subfolders']
                        path_so_far = ""

                        # Build folder hierarchy
                        parts = key.split('/')
                        for i, part in enumerate(parts[:-1]):  # Skip the file name
                            if not part:  # Skip empty parts
                                continue

                            path_so_far = path_so_far + "/" + part if path_so_far else part

                            # Initialize folder if not exists
                            if part not in current_level:
                                current_level[part] = {
                                    'path': path_so_far,
                                    'total': 0,
                                    'subfolders': {}
                                }

                            # Increment folder count
                            current_level[part]['total'] += 1

                            # Move to next level
                            current_level = current_level[part]['subfolders']

        # Add grand total
        hierarchy['Grand Total'] = grand_total

        # Flatten hierarchy for display
        flat_counts = flatten_hierarchy(hierarchy)

        return flat_counts, hierarchy
    except Exception as e:
        print(f"Error counting files: {e}")
        return None, None

# Helper function to flatten hierarchy for display
def flatten_hierarchy(hierarchy):
    flat_counts = {}

    # Add grand total
    if 'Grand Total' in hierarchy:
        flat_counts['Grand Total'] = hierarchy['Grand Total']

    # Process each bucket
    for bucket_name, bucket_data in hierarchy.items():
        if bucket_name == 'Grand Total':
            continue

        # Add bucket total
        flat_counts[f"{bucket_name}"] = bucket_data['total']

        # Process subfolders
        process_subfolders(flat_counts, bucket_name, bucket_data['subfolders'], 1)

    return flat_counts

# Helper function to process subfolders recursively
def process_subfolders(flat_counts, parent_path, subfolders, level):
    for folder_name, folder_data in sorted(subfolders.items()):
        # Create indented folder name
        indented_name = f"{parent_path}/{folder_name}"

        # Add folder count
        flat_counts[indented_name] = folder_data['total']

        # Process subfolders recursively
        process_subfolders(flat_counts, indented_name, folder_data['subfolders'], level + 1)

# Function to list all files in S3 bucket
def list_s3_files(s3_client, bucket_name, prefix=''):
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                files.append({
                    'key': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified']
                })
        return files
    except Exception as e:
        print(f"Error listing S3 files: {e}")
        return []

def get_feedback_stats():
    """Get feedback statistics from S3"""
    try:
        # Import the feedback counter module
        from feedback_counter import get_feedback_counts

        # Get the feedback data
        feedback_data = get_feedback_counts()
        return feedback_data
    except Exception as e:
        print(f"Error getting feedback stats: {e}")
        return {
            'yes': 0,
            'no': 0,
            'defect_types': {},
            'history': []
        }

@data_info_bp.route('/', methods=['GET', 'POST'])
def data_info_index():
    # Always count files on page load
    s3 = create_s3_client(AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_REGION)
    folder_counts, hierarchy = count_all_files(s3, S3_BUCKETS)

    # Get feedback statistics
    feedback_stats = get_feedback_stats()

    # Get file list if requested
    files = None
    selected_bucket = None
    if request.method == 'POST' and 'list_files' in request.form:
        selected_bucket = request.form.get('bucket', S3_BUCKETS[0])
        prefix = request.form.get('prefix', '')
        files = list_s3_files(s3, selected_bucket, prefix)

    return render_template('data_info.html',
                          folder_counts=folder_counts,
                          files=files,
                          buckets=S3_BUCKETS,
                          selected_bucket=selected_bucket,
                          feedback_stats=feedback_stats)
