import cv2
import numpy as np
import pywt

def calculate_entropy(coeff):
    hist, _ = np.histogram(coeff.flatten(), bins=256, density=True)
    return -np.sum(hist * np.log2(hist + 1e-7))

def extract_enhanced_features(roi):
    if roi is None or roi.size == 0:
        raise ValueError("ROI is empty or invalid")
    
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB)
    
    coeffs = pywt.wavedec2(gray, 'haar', level=2)
    cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
    
    wavelet_features = [
        np.mean(cA2), np.std(cA2), 
        np.mean(cH1), np.std(cH1),
        np.sum(cH2**2) / cH2.size,
        calculate_entropy(cV1)
    ]
    
    color_features = [
        np.mean(hsv[:,:,0]), np.std(hsv[:,:,1]),
        np.mean(lab[:,:,1]), np.mean(lab[:,:,2]),
        np.max(hsv[:,:,2]) - np.min(hsv[:,:,2])
    ]
    
    return wavelet_features + color_features