from ultralytics import YOLO

class VehicleDetector:

    def __init__(self):
        self.model = YOLO("yolov8n.pt")

    def detect(self, frame, conf=0.15, iou=0.45):
        results = self.model.predict(
            frame,
            conf=conf,
            iou=iou,
            verbose=False
        )
        return results