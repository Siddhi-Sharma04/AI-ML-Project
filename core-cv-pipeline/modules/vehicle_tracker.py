import cv2
import os
from ultralytics import YOLO

class VehicleTracker:
    def __init__(self, model_path):
        """
        Vehicle Detection aur Tracking class.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[CRITICAL ERROR] Vehicle Tracker file not found at: {model_path}")
            
        print(f"[INFO] Loading Vehicle Tracker Model from: {model_path}")
        self.model = YOLO(model_path)
        
    def track_vehicles(self, frame):
        """
        Frame accept karega aur tracked vehicles return karega.
        """
        results = self.model(frame, verbose=False)[0]
        boxes = results.boxes.data.cpu().numpy()
        return boxes