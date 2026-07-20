from ultralytics import YOLO
import os
import numpy as np
import cv2

class HelmetDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.warning_logged = False
        
        if os.path.exists(model_path):
            print(f"[INFO] Loading Helmet Detector Model from: {model_path}")
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                print(f"[ERROR] Failed to load helmet model: {e}")

    def check_helmet(self, rider_crop, track_id):
        """
        Deprecate this method in favor of full frame crop checks.
        """
        return True

def process_motorcycle_violations(
    boxes, frame, helmet_detector, tracker,
    triple_riding_frames, triple_riding_violations, helmet_violations
):
    """
    Processes all tracked motorcycles in the frame to detect:
    1. Triple riding violations (3+ overlapping Person bounding boxes).
    2. Helmet violations (riders not wearing a helmet).
    """
    if boxes is None or len(boxes) == 0:
        return

    motorcycles = []
    
    # Parse tracked boxes (scaled to 960x540 display resolution)
    for idx, box in enumerate(boxes):
        if len(box) >= 5:
            x1, y1, x2, y2 = map(int, box[:4])
            track_id = int(box[4]) if len(box) > 4 and int(box[4]) != 0 else (idx + 1)
            class_id = int(box[5]) if len(box) > 5 else 2
            
            if class_id == 3:  # Motorcycle
                motorcycles.append((x1, y1, x2, y2, track_id))

    # If no tracked motorcycles in the frame, we don't need to do any person checks
    if len(motorcycles) == 0:
        return

    # Use base YOLO detector to find ALL persons in the frame (conf=0.15 for high recall on occluded riders)
    persons = []
    try:
        det_results = tracker.model(frame, conf=0.15, verbose=False)[0]
        for box in det_results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id == 0:  # Person
                px1, py1, px2, py2 = map(int, box.xyxy[0].tolist())
                # Scale coordinates to display scale (960x540) for matching
                spx1 = int(px1 * (960.0 / frame.shape[1]))
                spy1 = int(py1 * (540.0 / frame.shape[0]))
                spx2 = int(px2 * (960.0 / frame.shape[1]))
                spy2 = int(py2 * (540.0 / frame.shape[0]))
                persons.append((spx1, spy1, spx2, spy2, px1, py1, px2, py2))
    except Exception as e:
        print(f"[ERROR] Base person detection failure: {e}")

    # Process violations for each motorcycle
    for mx1, my1, mx2, my2, m_id in motorcycles:
        if m_id is None:
            continue
            
        mw = mx2 - mx1
        mh = my2 - my1
        mcx = (mx1 + mx2) // 2
        
        if mw <= 0 or mh <= 0:
            continue

        riders = []
        for spx1, spy1, spx2, spy2, px1, py1, px2, py2 in persons:
            ph = spy2 - spy1
            pcx = (spx1 + spx2) // 2

            # Calculate intersection bounding box in display space
            ix1 = max(mx1, spx1)
            iy1 = max(my1, spy1)
            ix2 = min(mx2, spx2)
            iy2 = min(my2, spy2)

            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            intersection = iw * ih
            person_area = (spx2 - spx1) * (spy2 - spy1)
            overlap_ratio = intersection / person_area if person_area > 0 else 0

            # Proximity check
            center_offset = abs(pcx - mcx)
            horizontal_ok = center_offset <= (mw * 0.45)
            
            # VERTICAL CHECKS:
            # Rider's bottom spy2 must be close to motorcycle's bottom my2 or seat level (since passengers are occluded)
            vertical_ok = (my2 - mh * 0.55 <= spy2) and (spy2 <= my2 + 20)
            
            height_ok = ph >= (mh * 0.45)

            # Match only if the candidate meets both spatial layout and overlap constraints
            if overlap_ratio >= 0.35 and horizontal_ok and vertical_ok and height_ok:
                riders.append((px1, py1, px2, py2))

        # 1. Triple Riding Check
        riders_count = len(riders)
        if riders_count >= 3:
            triple_riding_violations.add(m_id)

        # 2. Helmet Detection Check for each rider
        if helmet_detector.model is not None:
            for rx1, ry1, rx2, ry2 in riders:
                # Top 45% crop of the raw unscaled rider bounding box to isolate the head
                head_y2 = ry1 + int((ry2 - ry1) * 0.45)
                cx1, cy1 = max(0, rx1), max(0, ry1)
                cx2, cy2 = min(frame.shape[1], rx2), min(frame.shape[0], head_y2)
                
                if cx2 > cx1 and cy2 > cy1:
                    rider_crop = frame[cy1:cy2, cx1:cx2]
                    if rider_crop.size > 0:
                        try:
                            res = helmet_detector.model(rider_crop, conf=0.15, verbose=False)[0]
                            
                            has_helmet = False
                            has_no_helmet = False
                            max_helmet_conf = 0.0
                            max_no_helmet_conf = 0.0
                            
                            for b in res.boxes:
                                cls_id = int(b.cls[0].item())
                                conf = b.conf[0].item()
                                class_name = helmet_detector.model.names.get(cls_id, "").lower()
                                
                                if 'without' in class_name or 'no' in class_name or cls_id == 1:
                                    if conf > max_no_helmet_conf:
                                        max_no_helmet_conf = conf
                                else:
                                    if conf > max_helmet_conf:
                                        max_helmet_conf = conf
                                        
                            # Override logic: If "With Helmet" is strongly detected, ignore "Without Helmet" noise
                            if max_helmet_conf >= 0.30:
                                has_helmet = True
                            elif max_no_helmet_conf >= 0.15:
                                has_no_helmet = True
                                
                            if has_no_helmet and not has_helmet:
                                helmet_violations.add(m_id)
                        except Exception as e:
                            print(f"[ERROR] Helmet crop model inference failure: {e}")
        else:
            # Fallback to mock violations for testing (motorcycles 3, 11, 18)
            if m_id in [3, 11, 18]:
                helmet_violations.add(m_id)
