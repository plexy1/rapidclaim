from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import base64
from io import BytesIO
import os
import sys
import pandas as pd

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model.load_images import get_random_image
from model.predict import predict

app = Flask(__name__, 
    static_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'node-app', 'public'),
    static_url_path=''
)
CORS(app)  # Enable CORS for all routes

# Load dataset info
DATASET_PATH = os.path.join("c:", "Users", "prana", "OneDrive", "Desktop", "Car Crash Model")
dataset_csv = pd.read_csv(os.path.join(DATASET_PATH, "dataset_database.csv"))

# Map CSV collision values to predictions
def map_collision(val):
    return "crash" if val == 'y' else "no_crash"

@app.route("/", methods=["GET"])
def home():
    return send_file('static/index.html')

@app.route("/predict", methods=["POST"])
def predict_image():
    try:
        if not request.json or 'image' not in request.json:
            return jsonify({"error": "No image provided"}), 400
            
        image_file = request.json['image']
        image_path = os.path.join(DATASET_PATH, "dataset", image_file)
        
        if not os.path.exists(image_path):
            return jsonify({"error": "Image not found"}), 404
            
        # Get actual label from CSV
        actual = dataset_csv[dataset_csv['subject'] == image_file]['collision'].iloc[0]
        actual_label = map_collision(actual)
        
        # Get model prediction
        prediction = predict(image_path)
        
        return jsonify({
            "prediction": "crash" if prediction['crash_detected'] else "no_crash",
            "confidence": prediction['confidence'],
            "actual": actual_label,
            "status": prediction['status']
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/random-image", methods=["GET"])
def get_random():
    try:
        # Get a random image from dataset
        random_row = dataset_csv.sample(n=1).iloc[0]
        image_file = random_row['subject']
        actual = random_row['collision']
        
        img_path = os.path.join(DATASET_PATH, "dataset", image_file)
        if not os.path.exists(img_path):
            return jsonify({"error": "Image file not found"}), 404
            
        # Load and convert image to base64
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Get model prediction
        prediction = predict(img_path)
        
        return jsonify({
            "image": img_str,
            "filename": image_file,
            "prediction": "crash" if prediction['crash_detected'] else "no_crash",
            "confidence": prediction['confidence'],
            "actual": map_collision(actual)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Import PIL here to avoid circular imports
    from PIL import Image
    # Ensure the server can be accessed from other devices on the network
    app.run(host='0.0.0.0', port=5000, debug=True)
