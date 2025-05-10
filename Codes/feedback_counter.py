import boto3
import json
import os
from datetime import datetime

# AWS S3 Configuration
AWS_ACCESS_KEY = 'AKIA3GW7WQUV5KKCUHMJ'
AWS_SECRET_KEY = '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'
AWS_REGION = 'us-east-1'
S3_BUCKET_NAME = 'waferdataset2'
FEEDBACK_FOLDER = 'Feedback'
FEEDBACK_FILE = 'feedback_counts.json'

# Initialize S3 client
def create_s3_client():
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION
    )

# Get current feedback counts from S3
def get_feedback_counts():
    s3 = create_s3_client()
    feedback_key = f"{FEEDBACK_FOLDER}/{FEEDBACK_FILE}"
    
    try:
        # Try to get existing feedback file
        response = s3.get_object(Bucket=S3_BUCKET_NAME, Key=feedback_key)
        feedback_data = json.loads(response['Body'].read().decode('utf-8'))
        return feedback_data
    except Exception as e:
        # If file doesn't exist or there's an error, create a new feedback data structure
        print(f"Creating new feedback counts file: {e}")
        feedback_data = {
            'yes': 0,
            'no': 0,
            'defect_types': {},
            'history': []
        }
        return feedback_data

# Update feedback counts and save to S3
def update_feedback_count(feedback_value, defect_type, image_name):
    # Get current counts
    feedback_data = get_feedback_counts()
    
    # Update counts
    if feedback_value.lower() == 'yes':
        feedback_data['yes'] += 1
    elif feedback_value.lower() == 'no':
        feedback_data['no'] += 1
    
    # Update defect type counts
    if defect_type not in feedback_data['defect_types']:
        feedback_data['defect_types'][defect_type] = {'yes': 0, 'no': 0}
    
    feedback_data['defect_types'][defect_type][feedback_value.lower()] += 1
    
    # Add to history
    feedback_data['history'].append({
        'timestamp': datetime.now().isoformat(),
        'feedback': feedback_value.lower(),
        'defect_type': defect_type,
        'image_name': image_name
    })
    
    # Save updated counts to S3
    s3 = create_s3_client()
    feedback_key = f"{FEEDBACK_FOLDER}/{FEEDBACK_FILE}"
    
    try:
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=feedback_key,
            Body=json.dumps(feedback_data, indent=2),
            ContentType='application/json'
        )
        print(f"Feedback counts updated in S3: {feedback_key}")
        return True
    except Exception as e:
        print(f"Error updating feedback counts in S3: {e}")
        return False

# Generate a summary report and save to S3
def generate_feedback_report():
    feedback_data = get_feedback_counts()
    
    # Create a formatted report
    report = []
    report.append("# Wafer Defect Detection Feedback Report")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    report.append("## Overall Feedback Counts")
    report.append(f"- Yes (Helpful): {feedback_data['yes']}")
    report.append(f"- No (Not Helpful): {feedback_data['no']}")
    total = feedback_data['yes'] + feedback_data['no']
    if total > 0:
        yes_percent = (feedback_data['yes'] / total) * 100
        report.append(f"- Satisfaction Rate: {yes_percent:.1f}%\n")
    
    report.append("## Feedback by Defect Type")
    for defect_type, counts in feedback_data['defect_types'].items():
        report.append(f"### {defect_type}")
        report.append(f"- Yes (Helpful): {counts['yes']}")
        report.append(f"- No (Not Helpful): {counts['no']}")
        type_total = counts['yes'] + counts['no']
        if type_total > 0:
            type_yes_percent = (counts['yes'] / type_total) * 100
            report.append(f"- Satisfaction Rate: {type_yes_percent:.1f}%\n")
    
    # Save report to S3
    report_text = "\n".join(report)
    report_key = f"{FEEDBACK_FOLDER}/feedback_report_{datetime.now().strftime('%Y%m%d')}.md"
    
    s3 = create_s3_client()
    try:
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=report_key,
            Body=report_text,
            ContentType='text/markdown'
        )
        print(f"Feedback report saved to S3: {report_key}")
        return True
    except Exception as e:
        print(f"Error saving feedback report to S3: {e}")
        return False
