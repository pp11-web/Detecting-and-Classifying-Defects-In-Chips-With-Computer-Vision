# Import matplotlib configuration first
import matplotlib_config

from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
import subprocess

# Import module blueprints
from M1f import m1f_bp
from M2f import m2f_bp
from Data_infof import data_info_bp

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = 'your_secret_key'

# Register blueprints
app.register_blueprint(m1f_bp, url_prefix='/pcb')
app.register_blueprint(m2f_bp, url_prefix='/wafer')
app.register_blueprint(data_info_bp, url_prefix='/data_info')

# Ensure necessary directories exist
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/outputs', exist_ok=True)
os.makedirs('static/transforms', exist_ok=True)
os.makedirs('static/results', exist_ok=True)
os.makedirs('transformation', exist_ok=True)
os.makedirs('temp', exist_ok=True)
os.makedirs('saved_files', exist_ok=True)
os.makedirs('runs/detect/predict', exist_ok=True)

# Routes to serve files from non-static directories
@app.route('/static/uploads/<path:filename>')
def serve_uploads(filename):
    return send_from_directory('static/uploads', filename)

@app.route('/static/outputs/<path:filename>')
def serve_outputs(filename):
    return send_from_directory('static/outputs', filename)

@app.route('/static/transforms/<path:filename>')
def serve_transforms(filename):
    return send_from_directory('static/transforms', filename)

@app.route('/transformation/<path:filename>')
def serve_transformations(filename):
    # Extract the transformation type from the path
    parts = filename.split('/')
    if len(parts) > 1:
        transform_type = parts[0]
        file_name = '/'.join(parts[1:])
        return send_from_directory(os.path.join('transformation', transform_type), file_name)
    return send_from_directory('transformation', filename)

@app.route('/runs/detect/predict/<path:filename>')
def serve_detections(filename):
    return send_from_directory('runs/detect/predict', filename)

@app.route("/", methods=["GET", "POST"])
def index():
    """Main index page with buttons to access different modules"""
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5001)