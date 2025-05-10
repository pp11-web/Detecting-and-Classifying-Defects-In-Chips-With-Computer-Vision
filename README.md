# Wafer & PCB Defect Detection System

A comprehensive web application for detecting and analyzing defects in semiconductor wafers and printed circuit boards (PCBs) using machine learning and computer vision techniques.

![Main Page GUI](Images/Main%20Page%20GUI.png)

## 🌟 Features

- **PCB Defect Detection**: Identify and visualize defects in PCB images using YOLO object detection
- **Wafer Defect Analysis**: Process wafer data to detect and classify defects
- **S3 Integration**: Automatic storage and retrieval of images and data from AWS S3
- **Data Visualization**: Interactive visualizations of defect patterns and statistics
- **User Feedback System**: Collect and analyze user feedback on detection accuracy
- **Multi-bucket Support**: Manage data across multiple S3 buckets

## 🏗️ System Architecture

The application follows a modular architecture with separate components for different functionalities:

![System Architecture](Images/System%20Architectute.png)

## 🔄 User Data Flow

The diagram below illustrates how data flows through the system:

![User Data Flow](Images/User%20Data%20Flow.png)

## 📊 Key Components

### Data Information Module

Provides comprehensive information about data stored in S3 buckets, including file counts, folder structure, and feedback statistics.

![Data Info GUI](Images/Data%20Info%20GUI.png)

### PCB Defect Detection

Analyzes PCB images to detect defects such as shorts, opens, and other manufacturing issues using YOLO object detection.

![PCB Defect GUI](Images/PCB%20Defect%20GUI.png)

### Wafer Defect Analysis

Processes wafer data to identify defects and visualize patterns with custom color mapping:
- Purple: Background
- Green: Silicon
- Yellow: Defects

![Wafer Defect GUI](Images/Wafer%20Defect%20GUI.png)

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- AWS account with S3 access
- Required Python packages (see requirements.txt)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/wafer-pcb-defect-detection.git
   cd wafer-pcb-defect-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure AWS credentials:
   ```bash
   # Set your AWS credentials in the appropriate files or use environment variables
   export AWS_ACCESS_KEY_ID="your-access-key"
   export AWS_SECRET_ACCESS_KEY="your-secret-key"
   export AWS_REGION="your-region"
   ```

4. Run the application:
   ```bash
   python app.py
   ```

## 🧩 Core Modules

### `app.py`
Main Flask application that integrates all components and handles routing.

### `Data_infof.py`
Manages S3 bucket interactions, file listing, and folder structure analysis.

### `M1f.py`
Handles PCB image processing, defect detection using YOLO, and image transformations.

### `M2f.py`
Processes wafer data, generates visualizations, and provides defect analysis.

### `feedback_counter.py`
Tracks and analyzes user feedback on detection accuracy, storing results in S3.

## 🌐 Cloud Deployment

The application can be deployed to AWS EC2 via S3:

1. Upload essential files to an S3 bucket
2. On your EC2 instance:
   - Install Python and dependencies
   - Download application files from S3
   - Configure AWS credentials
   - Run the Flask application (with Gunicorn/uWSGI for production)

## 📁 Project Structure

```
wafer-pcb-defect-detection/
├── app.py                  # Main Flask application
├── Data_infof.py           # S3 data information module
├── M1f.py                  # PCB image processing module
├── M2f.py                  # Wafer data processing module
├── feedback_counter.py     # Feedback tracking module
├── matplotlib_config.py    # Matplotlib configuration
├── Models/                 # ML model files
├── static/                 # Static assets
│   ├── outputs/            # PCB detection outputs
│   ├── results/            # Wafer analysis results
│   ├── transforms/         # PCB transformations
│   └── uploads/            # User uploads
└── templates/              # HTML templates
    ├── index.html          # Main landing page
    ├── data_info.html      # Data information page
    ├── index2.html         # PCB processing page
    └── index3.html         # Wafer processing page
```

## 🔧 Configuration

### S3 Buckets
The application uses multiple S3 buckets:
- `pcbdataset1`: For PCB images and detection results
- `waferdataset2`: For wafer data and feedback statistics

### Feedback System
User feedback is stored in the `waferdataset2` bucket in the `Feedback` folder, with:
- Counts of "Yes" and "No" responses
- Breakdown by defect type
- Historical feedback data
- Automatically generated reports

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or support, please contact [prayagpurani11@gmail.com](mailto:prayagpuarani11@gmail.com.com).
