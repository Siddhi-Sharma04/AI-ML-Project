from ultralytics import YOLO

class VehicleDetector:

    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect(self, frame):
        results = self.model.predict(
            frame,
            verbose=False
        )
        return results