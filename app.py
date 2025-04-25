from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import cv2
import numpy as np
from PIL import Image
from fish_feature import extract_enhanced_features

app = Flask(__name__)
CORS(app)
model = joblib.load('fish_freshness_svm_model.pkl')
scaler = joblib.load('fish_freshness_scaler.pkl')

@app.route('/')
def home():
    return "API Klasifikasi Ikan Siap Pakai"

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file).convert('RGB')
    img = np.array(img)

    roi = extract_roi_from_image(img)
    if roi is None:
        return jsonify({'error': 'ROI tidak valid'}), 400

    features = extract_enhanced_features(roi)
    scaled = scaler.transform([features])

    pred = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0][pred]  # confidence dari prediksi

    return jsonify({
        'label': int(pred),
        'confidence': round(float(proba), 2)
    })

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)