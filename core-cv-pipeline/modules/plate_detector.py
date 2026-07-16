import cv2
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path="weights/license_plate_detector.pt"):
        """
        License Plate Bounding Box Localization Class.
        license_plate_detector.pt model ko load karti hai.
        """
        print(f"[INFO] Loading License Plate Detector Model from: {model_path}")
        self.model = YOLO(model_path)
        
    def detect_plates(self, vehicle_crop):
        """
        Vehicle crop image lekar number plate coordinates segment karega.
        """
        results = self.model(vehicle_crop, verbose=False)[0]
        plate_boxes = results.boxes.data.cpu().numpy()
        return plate_boxes