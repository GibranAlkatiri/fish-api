from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import cv2
import numpy as np
from PIL import Image
import logging
from fish_feature import extract_enhanced_features

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# Load model dengan error handling
try:
    model = joblib.load('fish_freshness_svm_model.pkl')
    scaler = joblib.load('fish_freshness_scaler.pkl')
    logging.info("Model and scaler loaded successfully")
except Exception as e:
    logging.error("Error loading model:", exc_info=True)
    model = None
    scaler = None

def extract_roi(img_array):
    """Ekstrak ROI dengan deteksi warna insang atau fallback ke tengah."""
    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return img_array[y:y+h, x:x+w]
    # Fallback: ambil tengah jika deteksi gagal
    h, w = img_array.shape[:2]
    size = min(w, h) // 3
    return img_array[h//3:h//3+size, w//3:w//3+size]

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
        
    try:
        img = Image.open(file).convert('RGB')
        img = np.array(img)
        roi = extract_roi(img)
        features = extract_enhanced_features(roi)
        scaled = scaler.transform([features])
        proba = model.predict_proba(scaled)[0]
        pred = np.argmax(proba)
        return jsonify({
            'label': int(pred),
            'confidence': float(proba[pred])
        })
    except Exception as e:
        logging.error("Prediction error:", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)