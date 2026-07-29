import cv2
import os
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path):
        """
        License Plate Bounding Box Localization Class.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[CRITICAL ERROR] License Plate Model file not found at: {model_path}")
            
        print(f"[INFO] Loading License Plate Detector Model from: {model_path}")
        self.model = YOLO(model_path)
        
    def detect_plates(self, vehicle_crop, conf=0.15):
        """
        Vehicle crop image lekar number plate coordinates segment karega.
        """
        results = self.model(vehicle_crop, conf=conf, verbose=False)[0]
        plate_boxes = results.boxes.data.cpu().numpy()
        return plate_boxes