import cv2
import os
import numpy as np
from ultralytics import YOLO


class VehicleTracker:
    def __init__(self, model_path):
        """
        Vehicle Detection and Tracking class using YOLOv8 built-in tracking.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[CRITICAL ERROR] Vehicle Tracker file not found at: {model_path}")

        print(f"[INFO] Loading Vehicle Tracker Model from: {model_path}")
        self.model = YOLO(model_path)

    def track_vehicles(self, frame, conf_threshold=0.20, iou_threshold=0.5):
        """
        Frame accept karega aur tracked vehicles return karega with stable track IDs.
        Format of returned list elements: [x1, y1, x2, y2, track_id, class_id, confidence]
        """
        results = self.model.track(
            frame, 
            persist=True, 
            conf=conf_threshold, 
            iou=iou_threshold, 
            tracker="bytetrack.yaml", 
            verbose=False
        )[0]
        
        boxes = []
        if results.boxes.id is not None:
            xyxy = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)
            clss = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()
            
            for i in range(len(xyxy)):
                boxes.append([
                    xyxy[i][0], xyxy[i][1], xyxy[i][2], xyxy[i][3],
                    ids[i], clss[i], confs[i]
                ])
        else:
            # Fallback if tracker is warming up or no objects tracked:
            # use detections with dummy track ID (idx + 1)
            raw_boxes = results.boxes.data.cpu().numpy()
            for idx, box in enumerate(raw_boxes):
                if len(box) >= 6:
                    boxes.append([
                        box[0], box[1], box[2], box[3],
                        idx + 1, int(box[5]), box[4]
                    ])
        return boxes
