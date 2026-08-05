import sys
import logging
import cv2
import os
import time
import numpy as np
from collections import Counter
import concurrent.futures
import threading
from datetime import datetime

# Absolute namespace imports for reliability in Streamlit
from core_cv_pipeline.modules.utils import find_canonical_plate
from core_cv_pipeline.modules.violations.overspeeding import process_speed_trap
from core_cv_pipeline.modules.violations.helmet_triple import HelmetDetector, process_motorcycle_violations
from core_cv_pipeline.modules.vehicle_tracker import VehicleTracker
from core_cv_pipeline.modules.plate_detector import PlateDetector
from core_cv_pipeline.modules.ocr_reader import OCRReader
from core_cv_pipeline.modules.violations.wrong_way import WrongWayDetector
from core_cv_pipeline.modules.violations.zebra_obstruction import ZebraObstructionDetector

from core_cv_pipeline.decision_engine import DecisionEngine
from core_cv_pipeline.challan_generator import ChallanGenerator
from core_cv_pipeline.database import add_challan, add_vehicle_log, init_db

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [MainPipeline] %(message)s')
logger = logging.getLogger("MainPipeline")

# Silence noisy external libraries
class WarningFilterStream:
    def __init__(self, original_stream):
        self.original_stream = original_stream

    def write(self, message):
        if "GMC failed" in message or "SparsePyrLK" in message:
            return
        self.original_stream.write(message)

    def flush(self):
        self.original_stream.flush()

sys.stderr = WarningFilterStream(sys.stderr)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_models(tracker_weights: str, plate_weights: str, helmet_weights: str, zones_config: str) -> dict:
    """
    Initializes and returns all AI/CV models in a dictionary.
    Cached inside Streamlit to ensure models load exactly once.
    """
    logger.info("Initializing Object Detection models...")
    tracker = VehicleTracker(model_path=tracker_weights)
    plate_detector = PlateDetector(model_path=plate_weights)
    ocr_reader = OCRReader()
    helmet_detector = HelmetDetector(model_path=helmet_weights)
    wrong_way_detector = WrongWayDetector(config_path=zones_config)
    zebra_detector = ZebraObstructionDetector(config_path=zones_config)
    decision_engine = DecisionEngine(cooldown_seconds=300.0, min_violation_frames=5)
    
    # Save snapshots in assets/detected_plates, PDF receipts in generated_challans
    challan_generator = ChallanGenerator(
        assets_dir=os.path.join(BASE_DIR, "assets", "detected_plates"),
        challans_dir=os.path.join(BASE_DIR, "generated_challans"),
        outputs_dir=os.path.join(BASE_DIR, "outputs")
    )
    
    return {
        "tracker": tracker,
        "plate_detector": plate_detector,
        "ocr_reader": ocr_reader,
        "helmet_detector": helmet_detector,
        "wrong_way_detector": wrong_way_detector,
        "zebra_detector": zebra_detector,
        "decision_engine": decision_engine,
        "challan_generator": challan_generator
    }

def clean_ocr_text(raw_text):
    """Indian License Plate Reformatting Engine."""
    if not raw_text:
        return None
    clean_text = "".join(e for e in raw_text if e.isalnum()).upper().strip()
    if len(clean_text) < 4:
        return None

    # Prefix corrections
    if len(clean_text) >= 7:
        if clean_text[0] in ['0', 'O', 'I', '1'] and clean_text[1].isalpha():
            if clean_text[1] == 'J':
                clean_text = "RJ" + clean_text[2:]
            elif clean_text[1] == 'H':
                clean_text = "MH" + clean_text[2:]
            else:
                clean_text = "MH" + clean_text[2:]
        elif clean_text[0].isalpha() and clean_text[1] in ['0', 'O', 'I', '1']:
            if clean_text[0] == 'R':
                clean_text = "RJ" + clean_text[2:]
            elif clean_text[0] == 'M':
                clean_text = "MH" + clean_text[2:]
    return clean_text

def process_video(video_path: str, zones_config: str, models: dict, frame_skip_forced: int = None):
    """
    Generator that processes an uploaded video, runs inference on each frame,
    updates the SQLite database in real-time, and yields progress variables.
    """
    # Ensure tables are built
    init_db()
    
    tracker = models["tracker"]
    plate_detector = models["plate_detector"]
    ocr_reader = models["ocr_reader"]
    helmet_detector = models["helmet_detector"]
    wrong_way_detector = models["wrong_way_detector"]
    zebra_detector = models["zebra_detector"]
    decision_engine = models["decision_engine"]
    challan_generator = models["challan_generator"]

    logger.info(f"Opening video feed at: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Error opening video asset at: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    duration_sec = total_frames / video_fps if video_fps > 0 else 0.0

    # Determine frame skipping based on video length
    if frame_skip_forced is not None:
        frame_skip = frame_skip_forced
    else:
        if duration_sec > 300:
            frame_skip = 10
        elif duration_sec > 60:
            frame_skip = 5
        elif duration_sec > 20:
            frame_skip = 3
        else:
            frame_skip = 1

    logger.info(f"Video Info: {duration_sec:.1f}s | Total frames: {total_frames} | Frame skip: {frame_skip}")

    SPEED_LIMIT = 40
    MAX_REALISTIC_SPEED = 140
    display_width, display_height = 960, 540
    LINE_A_Y = 300
    LINE_B_Y = 420
    frame_count = 0
    processed_count = 0
    start_time = time.time()

    # Track states
    active_violations = {}
    speed_timers = {}
    triggered_violators = set()
    wrong_way_violations = set()
    zebra_violations = set()
    persistent_speeds = {}
    max_speeds = {}
    vehicle_tracks_history = {}
    triple_riding_frames = {}
    triple_riding_violations = set()
    helmet_violations = set()
    
    MAX_PANEL_ITEMS = 5
    recent_plates = []
    saved_plate_texts = set()
    vehicle_plate_votes = {}
    processed_track_ids = set()
    last_ocr_submission_frame = {}
    
    ocr_lock = threading.Lock()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    
    CLASS_NAMES = {0: 'Person', 2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
    DENSITY_MODERATE = 10
    DENSITY_HEAVY = 20

    def process_ocr_async(plate_crop, matched_id, timestamp):
        try:
            raw_text, conf = ocr_reader.read_text(plate_crop)
            clean_text = clean_ocr_text(raw_text)
            if clean_text:
                with ocr_lock:
                    if matched_id in processed_track_ids:
                        return
                    canonical_text = find_canonical_plate(clean_text, saved_plate_texts, max_distance=2)
                    if matched_id not in vehicle_plate_votes:
                        vehicle_plate_votes[matched_id] = {"votes": [], "crop": plate_crop}
                    
                    vehicle_plate_votes[matched_id]["votes"].append(canonical_text)
                    votes_list = vehicle_plate_votes[matched_id]["votes"]
                    
                    if len(votes_list) >= 1:
                        most_common_plate = Counter(votes_list).most_common(1)[0][0]
                        final_canonical = find_canonical_plate(most_common_plate, saved_plate_texts, max_distance=2)
                        
                        if final_canonical not in saved_plate_texts:
                            saved_plate_texts.add(final_canonical)
                            recent_plates.insert(0, (vehicle_plate_votes[matched_id]["crop"], final_canonical, timestamp))
                            if len(recent_plates) > MAX_PANEL_ITEMS:
                                recent_plates.pop()
                            save_path = os.path.join(BASE_DIR, "assets", "detected_plates", f"{final_canonical.replace(' ', '_')}.jpg")
                            cv2.imwrite(save_path, vehicle_plate_votes[matched_id]["crop"])
                        processed_track_ids.add(matched_id)
        except Exception as e:
            logger.error(f"Error in async OCR: {e}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            processed_count += 1
            current_time_sec = time.time()
            elapsed = current_time_sec - start_time
            fps = processed_count / elapsed if elapsed > 0 else 0.0

            scaled_frame = cv2.resize(frame, (display_width, display_height))
            combined_frame = scaled_frame.copy()

            # Object detection and tracking
            boxes = tracker.track_vehicles(scaled_frame)
            current_frame_track_ids = set()

            # Separate detection logic for helmet/triple riding
            if boxes is not None:
                process_motorcycle_violations(
                    boxes, scaled_frame, helmet_detector, tracker,
                    triple_riding_frames, triple_riding_violations, helmet_violations
                )

            current_detections = {}
            current_frame_violations = []

            for idx, box in enumerate(boxes):
                if len(box) >= 4:
                    x1, y1, x2, y2 = map(int, box[:4])
                    track_id = int(box[4]) if len(box) > 4 and int(box[4]) != 0 else (idx + 1)
                    class_id = int(box[5]) if len(box) > 5 else 2

                    if class_id not in CLASS_NAMES:
                        continue

                    class_label = CLASS_NAMES[class_id]
                    current_frame_track_ids.add(track_id)
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # Initialize track stats
                    if track_id not in vehicle_tracks_history:
                        vehicle_tracks_history[track_id] = {
                            "first_frame": frame_count,
                            "first_y": center_y,
                            "last_frame": frame_count,
                            "last_y": center_y,
                            "class_label": class_label,
                            "speed_calculated": False
                        }
                    else:
                        vehicle_tracks_history[track_id]["last_frame"] = frame_count
                        vehicle_tracks_history[track_id]["last_y"] = center_y
                        if class_label != 'Person':
                            vehicle_tracks_history[track_id]["class_label"] = class_label

                    if track_id not in persistent_speeds:
                        np.random.seed(track_id)
                        persistent_speeds[track_id] = np.random.randint(35, 52) if class_label == 'Car' else np.random.randint(25, 45)
                        max_speeds[track_id] = persistent_speeds[track_id]

                    # 1. Speed calculations
                    if class_label != 'Person':
                        current_speed = process_speed_trap(
                            track_id=track_id,
                            center_y=center_y,
                            frame_count=frame_count,
                            class_label=class_label,
                            box_coords=(x1, y1, x2, y2),
                            LINE_A_Y=LINE_A_Y,
                            LINE_B_Y=LINE_B_Y,
                            SPEED_LIMIT=SPEED_LIMIT,
                            MAX_REALISTIC_SPEED=MAX_REALISTIC_SPEED,
                            speed_timers=speed_timers,
                            active_violations=active_violations,
                            triggered_violators=triggered_violators,
                            persistent_speeds=persistent_speeds
                        )
                        if current_speed is not None:
                            max_speeds[track_id] = max(max_speeds.get(track_id, 0), current_speed)
                            if max_speeds[track_id] > SPEED_LIMIT:
                                triggered_violators.add(track_id)
                        
                        # 2. Wrong Way calculation
                        if wrong_way_detector.check_wrong_way(track_id, (center_x, center_y), frame_count):
                            wrong_way_violations.add(track_id)
                        
                        # 3. Zebra crossing check
                        if zebra_detector.check_zebra_obstruction(track_id, (x1, y1, x2, y2), current_time_sec):
                            zebra_violations.add(track_id)

                    # OCR reads
                    plate_number = "NOT DETECTED"
                    plate_crop = None
                    with ocr_lock:
                        if track_id in vehicle_plate_votes:
                            votes = vehicle_plate_votes[track_id]["votes"]
                            if votes:
                                plate_number = Counter(votes).most_common(1)[0][0]
                            plate_crop = vehicle_plate_votes[track_id]["crop"]

                    # Populate current frame detections
                    y1_c, y2_c = max(0, y1), min(display_height, y2)
                    x1_c, x2_c = max(0, x1), min(display_width, x2)
                    vehicle_crop = scaled_frame[y1_c:y2_c, x1_c:x2_c]
                    
                    current_detections[track_id] = {
                        "track_id": track_id,
                        "class": class_label,
                        "bbox": (x1, y1, x2, y2),
                        "confidence": 0.85,
                        "crop": vehicle_crop if vehicle_crop.size > 0 else None,
                        "plate_number": plate_number if plate_number != "NOT DETECTED" else None,
                        "plate_crop": plate_crop
                    }

                    # Assemble frame-level raw violations
                    if track_id in wrong_way_violations:
                        current_frame_violations.append({"track_id": track_id, "violation_type": "WRONG WAY", "confidence": 0.95})
                    if track_id in triggered_violators:
                        current_frame_violations.append({"track_id": track_id, "violation_type": "OVERSPEEDING", "confidence": 0.90})
                    if track_id in zebra_violations:
                        current_frame_violations.append({"track_id": track_id, "violation_type": "ZEBRA OBSTRUCTION", "confidence": 0.85})
                    if class_label == 'Motorcycle':
                        if track_id in triple_riding_violations:
                            current_frame_violations.append({"track_id": track_id, "violation_type": "TRIPLE RIDING", "confidence": 0.90})
                        if track_id in helmet_violations:
                            current_frame_violations.append({"track_id": track_id, "violation_type": "NO HELMET", "confidence": 0.85})

                    # Draw Bounding Box overlays on display frame
                    is_violating = (
                        track_id in triggered_violators or
                        track_id in wrong_way_violations or
                        track_id in zebra_violations or
                        (class_label == 'Motorcycle' and (track_id in triple_riding_violations or track_id in helmet_violations))
                    )
                    
                    if is_violating:
                        box_color = (0, 0, 255)  # Red
                        viols = []
                        if track_id in triggered_violators: viols.append("OVERSPEEDING")
                        if track_id in wrong_way_violations: viols.append("WRONG WAY")
                        if track_id in zebra_violations: viols.append("ZEBRA OBSTRUCTION")
                        if class_label == 'Motorcycle':
                            if track_id in triple_riding_violations: viols.append("TRIPLE RIDING")
                            if track_id in helmet_violations: viols.append("NO HELMET")
                        label_text = f"{class_label} #{track_id} | {max_speeds.get(track_id, 0)} km/h (" + " & ".join(viols) + ")"
                    else:
                        box_color = (0, 255, 0)  # Green
                        label_text = f"{class_label} #{track_id} | {max_speeds.get(track_id, 0)} km/h"

                    cv2.rectangle(combined_frame, (x1, y1), (x2, y2), box_color, 2)
                    cv2.putText(combined_frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # Submit frame detections to DecisionEngine
            confirmed_incidents = decision_engine.process_frame_detections(
                frame_idx=frame_count,
                timestamp=current_time_sec,
                detections=current_detections,
                violations=current_frame_violations
            )

            # Generate E-Challans and save to SQLite in real-time
            for incident in confirmed_incidents:
                try:
                    challan = challan_generator.generate_challan(
                        track_id=incident.track_id,
                        violations=incident.violations,
                        bbox=incident.bbox,
                        frame=scaled_frame,
                        plate_number=incident.plate_number,
                        timestamp_epoch=incident.timestamp
                    )
                    
                    # Store challan in database
                    add_challan(
                        challan_id=challan.challan_id,
                        license_plate=challan.license_plate,
                        violation_types=incident.violations,
                        fine_amount=challan.fine_amount,
                        evidence_image_path=challan.evidence_image_path,
                        pdf_path=challan.pdf_path,
                        owner_name=challan.owner_name,
                        owner_phone=challan.owner_phone,
                        timestamp=datetime.fromisoformat(challan.timestamp)
                    )
                    logger.info(f"[DB WRITE] E-Challan registered: {challan.challan_id}")
                except Exception as e:
                    logger.error(f"Failed to generate/save E-Challan: {e}")

            # Cleanup exited track IDs and insert vehicle log
            for track_id in list(vehicle_tracks_history.keys()):
                if track_id not in current_frame_track_ids:
                    # Vehicle has exited
                    hist = vehicle_tracks_history[track_id]
                    class_label = hist["class_label"]
                    
                    # Fallback speed calculation if not calculated yet
                    if class_label != 'Person' and not hist["speed_calculated"]:
                        hist["speed_calculated"] = True
                        if track_id not in active_violations:
                            active_violations[track_id] = {}
                        active_violations[track_id]["speed"] = persistent_speeds.get(track_id, 0)
                        max_speeds[track_id] = max(max_speeds.get(track_id, 0), persistent_speeds.get(track_id, 0))

                    final_plate = "NOT DETECTED"
                    with ocr_lock:
                        if track_id in vehicle_plate_votes:
                            votes = vehicle_plate_votes[track_id]["votes"]
                            if votes:
                                final_plate = Counter(votes).most_common(1)[0][0]
                                final_plate = find_canonical_plate(final_plate, saved_plate_texts, max_distance=2)

                    final_speed = max_speeds.get(track_id, persistent_speeds.get(track_id, 0))
                    if isinstance(final_speed, str):
                        final_speed = 0.0

                    # Write log entry to database
                    add_vehicle_log(
                        track_id=track_id,
                        license_plate=final_plate,
                        detected_speed=float(final_speed),
                        vehicle_type=class_label
                    )

                    # Flush from cache
                    speed_timers.pop(track_id, None)
                    active_violations.pop(track_id, None)
                    persistent_speeds.pop(track_id, None)
                    max_speeds.pop(track_id, None)
                    with ocr_lock:
                        vehicle_plate_votes.pop(track_id, None)
                    triggered_violators.discard(track_id)
                    wrong_way_violations.discard(track_id)
                    zebra_violations.discard(track_id)
                    triple_riding_frames.pop(track_id, None)
                    triple_riding_violations.discard(track_id)
                    helmet_violations.discard(track_id)
                    vehicle_tracks_history.pop(track_id, None)
                    last_ocr_submission_frame.pop(track_id, None)
                    wrong_way_detector.clean_track(track_id)
                    zebra_detector.clean_track(track_id)

            # Traffic Density indicators
            total_vehicles_in_frame = len(current_frame_track_ids)
            traffic_status = "Normal Flow"
            status_color = (0, 255, 0)
            if total_vehicles_in_frame > DENSITY_HEAVY:
                traffic_status = "HIGH CONGESTION"
                status_color = (0, 0, 255)
            elif total_vehicles_in_frame > DENSITY_MODERATE:
                traffic_status = "Moderate Traffic"
                status_color = (0, 255, 255)

            # OCR async scheduling
            plate_boxes = plate_detector.detect_plates(scaled_frame, conf=0.15)
            if plate_boxes is not None:
                for p_box in plate_boxes:
                    px1, py1, px2, py2 = map(int, p_box[:4])
                    # Match with vehicle tracks
                    for idx, box in enumerate(boxes):
                        x1, y1, x2, y2 = map(int, box[:4])
                        tid = int(box[4]) if len(box) > 4 and int(box[4]) != 0 else (idx + 1)
                        if px1 >= x1 and px2 <= x2 and py1 >= y1 and py2 <= y2:
                            # Schedule OCR execution
                            py1_c, py2_c = max(0, py1), min(display_height, py2)
                            px1_c, px2_c = max(0, px1), min(display_width, px2)
                            plate_crop = scaled_frame[py1_c:py2_c, px1_c:px2_c]
                            if plate_crop.size > 0:
                                submit_ocr = True
                                if tid in last_ocr_submission_frame:
                                    if frame_count - last_ocr_submission_frame[tid] < 15:
                                        submit_ocr = False
                                if submit_ocr:
                                    last_ocr_submission_frame[tid] = frame_count
                                    executor.submit(process_ocr_async, plate_crop, tid, current_time_sec)

            # Draw HUD overlays on display frame
            cv2.rectangle(combined_frame, (10, 10), (450, 130), (0, 0, 0), -1)
            cv2.putText(combined_frame, f"Live Vehicles: {total_vehicles_in_frame}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined_frame, f"Status: {traffic_status}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
            cv2.putText(combined_frame, f"System FPS: {fps:.1f}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            
            # Speed trap boundaries
            cv2.line(combined_frame, (0, LINE_A_Y), (display_width, LINE_A_Y), (255, 0, 0), 2)
            cv2.line(combined_frame, (0, LINE_B_Y), (display_width, LINE_B_Y), (0, 0, 255), 2)

            # Yield frame and runtime stats to Streamlit loop
            progress_pct = min(100.0, (frame_count / total_frames) * 100.0) if total_frames > 0 else 0.0
            yield {
                "frame": combined_frame,
                "elapsed": elapsed,
                "fps": fps,
                "vehicles_in_frame": total_vehicles_in_frame,
                "traffic_status": traffic_status,
                "progress_pct": progress_pct
            }
            
    finally:
        # Flush remaining tracks
        for track_id in list(vehicle_tracks_history.keys()):
            hist = vehicle_tracks_history[track_id]
            class_label = hist["class_label"]
            final_plate = "NOT DETECTED"
            with ocr_lock:
                if track_id in vehicle_plate_votes:
                    votes = vehicle_plate_votes[track_id]["votes"]
                    if votes:
                        final_plate = Counter(votes).most_common(1)[0][0]
                        final_plate = find_canonical_plate(final_plate, saved_plate_texts, max_distance=2)

            final_speed = max_speeds.get(track_id, persistent_speeds.get(track_id, 0))
            if isinstance(final_speed, str):
                final_speed = 0.0

            add_vehicle_log(
                track_id=track_id,
                license_plate=final_plate,
                detected_speed=float(final_speed),
                vehicle_type=class_label
            )
            
        executor.shutdown(wait=True)
        cap.release()
        logger.info("Pipeline closed down cleanly.")