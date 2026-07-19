import cv2
import os
import numpy as np
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

    def track_vehicles(self, frame, conf_threshold=0.4):
        """
        Frame accept karega aur STABLE tracked vehicle IDs return karega.

        IMPORTANT FIX: Pehle self.model(frame) sirf plain detection kar raha tha,
        jisme track_id wala column exist hi nahi karta (data = [x1,y1,x2,y2,conf,cls]).
        main.py isme box[4] ko track_id samajh raha tha, jo actually confidence
        score tha — isliye track IDs frame-to-frame unstable the aur plate-voting
        system kabhi kabhi galat vehicle se match ho raha tha.

        Ab self.model.track(persist=True) use kar rahe hain, jo ByteTrack se
        real consistent track_id deta hai — data format ban jaata hai:
        [x1, y1, x2, y2, track_id, class_id]  (exactly wahi jo main.py expect karta hai)
        """
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            conf=conf_threshold,
            tracker="bytetrack.yaml"
        )[0]

        # Agar is frame mein koi bhi confirmed track nahi hai (naye/lost objects)
        if results.boxes is None or results.boxes.id is None:
            return np.empty((0, 6))

        boxes_xyxy = results.boxes.xyxy.cpu().numpy()
        track_ids = results.boxes.id.cpu().numpy().reshape(-1, 1)
        classes = results.boxes.cls.cpu().numpy().reshape(-1, 1)

        # main.py ka expected format: x1, y1, x2, y2, track_id, class_id
        combined = np.hstack([boxes_xyxy, track_ids, classes])
        return combined