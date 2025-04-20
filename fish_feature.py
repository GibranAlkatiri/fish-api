import cv2
import numpy as np
import pywt

def extract_enhanced_features(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    coeffs = pywt.wavedec2(gray, 'haar', level=2)
    cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    features = [
        np.mean(cA2), np.std(cA2), np.mean(cH1), np.std(cH1),
        np.mean(hsv[:,:,0]), np.std(hsv[:,:,1]),
        np.mean(lab[:,:,1]), np.mean(lab[:,:,2])
    ]
    return features