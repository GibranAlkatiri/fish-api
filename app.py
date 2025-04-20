from flask import Flask, request, jsonify
import joblib
import cv2
import numpy as np
from PIL import Image
from fish_feature import extract_enhanced_features

app = Flask(__name__)
model = joblib.load('fish_freshness_svm_model.pkl')
scaler = joblib.load('fish_freshness_scaler.pkl')

@app.route('/')
def home():
    return "API Klasifikasi Ikan Siap Pakai"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    try:
        img = Image.open(file.stream).convert('RGB')
        img = np.array(img)
        roi = cv2.resize(img, (128, 128))
        features = extract_enhanced_features(roi)
        scaled = scaler.transform([features])
        pred = model.predict(scaled)[0]
        return jsonify({'label': int(pred)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)