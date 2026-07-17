import cv2     # changes made in palak-work
import os      # changes made in niyati-work
import time
import numpy as np
import re
from collections import Counter
# Internal tracking dependencies ke theek neeche jodhein
from modules.violations.overspeeding import process_speed_trap
# Safe absolute path generation logic
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Internal tracking dependencies
from modules.vehicle_tracker import VehicleTracker
from modules.plate_detector import PlateDetector
from modules.ocr_reader import OCRReader

# Fully resolved asset configurations
TRACKER_WEIGHTS = os.path.join(BASE_DIR, "weights", "yolov8n.pt")
PLATE_WEIGHTS = os.path.join(BASE_DIR, "weights", "license_plate_detector.pt")

# Initialize models passing correct local absolute paths
tracker = VehicleTracker(model_path=TRACKER_WEIGHTS)
plate_detector = PlateDetector(model_path=PLATE_WEIGHTS)
ocr_reader = OCRReader()

SAVE_DIR = os.path.join(BASE_DIR, "detected_plates")
os.makedirs(SAVE_DIR, exist_ok=True)

MAX_PANEL_ITEMS = 5
recent_plates = []          # Store structure: [(crop, text, timestamp)]
saved_plate_texts = set()   

# Voting system to hold dynamic live plates per vehicle ID
vehicle_plate_votes = {}    # Format: { track_id: {"votes": [], "crop": img} }
processed_track_ids = set() # Taaki ek vehicle life-cycle mein ek hi baar panel par aaye

DENSITY_MODERATE = 10  
DENSITY_HEAVY = 20     

active_violations = {}
speed_timers = {}          
triggered_violators = set() 

# --- Optimized Boundaries for India Traffic Flow ---
SPEED_LIMIT = 55            
MAX_REALISTIC_SPEED = 140  
CLASS_NAMES = {0: 'Person', 2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}
persistent_speeds = {}      

def clean_ocr_text(raw_text):
    """
    Indian License Plate Pattern Engine strictly captures and auto-corrects standard alignments:
    Fixes cases like: '0ICP8570' -> 'MH01CP8570', 'I01D0570' -> 'MH01CD0570'
    """
    if not raw_text:
        return None
        
    # Sirf alphanumeric characters extract karein aur spaces hatayein
    clean_text = "".join(e for e in raw_text if e.isalnum()).upper().strip()
    
    if len(clean_text) < 4:
        return None

    # --- ADVANCED INDIAN AUTO-CORRECTION LAYER ---
    # Agar OCR shuruvaat ke 'MH' ko galat read karke noise character (0, I, O, 1) bana deta hai:
    if len(clean_text) >= 7:
        if clean_text[0] in ['0', 'O', 'I', '1'] and clean_text[1].isalpha():
            # First two characters ko sabse zyada probable state code 'MH' se badal diya
            clean_text = "MH" + clean_text[1:]
        elif clean_text[0] in ['0', 'O', 'I', '1'] and clean_text[1] in ['0', 'O', 'I', '1'] and clean_text[2].isalpha():
            clean_text = "MH" + clean_text[2:]

    # Index-specific corrections ke liye string ko list me convert kiya
    corrected = list(clean_text)
    
    # Rule 1: Pehle 2 characters strict Letters hone chahiye (State Code: MH, DL, HR)
    if len(corrected) >= 2:
        if corrected[0] == '1': corrected[0] = 'I'
        if corrected[0] == '0': corrected[0] = 'O'
        if corrected[1] == '1': corrected[1] = 'I'
        if corrected[1] == '0': corrected[1] = 'O'
        
    # Rule 2: Agle 2 characters strict Digits hone chahiye (District Code: 01, 12, 48)
    if len(corrected) >= 4:
        if corrected[2] == 'I' or corrected[2] == 'L': corrected[2] = '1'
        if corrected[2] == 'O' or corrected[2] == 'Q': corrected[2] = '0'
        if corrected[3] == 'I' or corrected[3] == 'L': corrected[3] = '1'
        if corrected[3] == 'O' or corrected[3] == 'Q': corrected[3] = '0'

    final_str = "".join(corrected)
    
    # Strict RegEx Match Check (Standard Indian Format: MH01CP8570)
    indian_plate_pattern = re.compile(r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$')
    
    if indian_plate_pattern.match(final_str):
        state = final_str[0:2]
        district = final_str[2:4]
        
        if final_str[5].isdigit(): # 1 letter series case (e.g., MH01A1234)
            series = final_str[4:5]
            num = final_str[5:]
            return f"{state} {district} {series} {num}"
        else: # 2 letter series case (e.g., MH01CP8570)
            series = final_str[4:6]
            num = final_str[6:]
            return f"{state} {district} {series} {num}"

    # Fallback format matching logic for noisy capture frames
    if len(final_str) >= 7 and final_str[-4:].isdigit():
        return f"{final_str[:-4]} {final_str[-4:]}"
        
    return None

def build_side_panel(items, panel_width=320, panel_height=540):
    panel = np.zeros((panel_height, panel_width, 3), dtype="uint8")
    block_height = panel_height // MAX_PANEL_ITEMS

    for i, (crop, text, _) in enumerate(items):
        y1 = i * block_height
        thumb_h = block_height - 35
        thumb_w = panel_width - 20

        if crop is not None and crop.size > 0 and thumb_h > 0 and thumb_w > 0:
            try:
                thumb = cv2.resize(crop, (thumb_w, thumb_h))
                panel[y1:y1 + thumb_h, 10:10 + thumb_w] = thumb
            except Exception:
                pass

        cv2.rectangle(panel, (0, y1), (panel_width, y1 + block_height), (60, 60, 60), 1)
        cv2.putText(panel, text[:17], (12, y1 + thumb_h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return panel

# Ingest target local video feed
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "numberplate4.mp4")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"[CRITICAL ERROR] Error opening video asset at: {VIDEO_PATH}")
    exit()

display_width = 960
display_height = 540

LINE_A_Y = 300   
LINE_B_Y = 420   
frame_count = 0  

print("\n--- Traffic Enforcement & Indian OCR Engine Running ---\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1 
    current_time_sec = time.time()

    scaled_frame = cv2.resize(frame, (display_width, display_height))
    combined_frame = scaled_frame.copy() 

    # ---------------- Vehicle Tracking Engine ----------------
    boxes = tracker.track_vehicles(scaled_frame)
    current_frame_track_ids = set()
    
    if boxes is not None:
        for idx, box in enumerate(boxes):
            if len(box) >= 4:
                x1, y1, x2, y2 = map(int, box[:4])
                
                track_id = int(box[4]) if len(box) > 4 and int(box[4]) != 0 else (idx + 1)
                class_id = int(box[5]) if len(box) > 5 else 2
                
                if class_id not in CLASS_NAMES:
                    continue
                    
                class_label = CLASS_NAMES[class_id]
                
                current_frame_track_ids.add(track_id)
                center_y = int((y1 + y2) / 2)
                
                if track_id not in persistent_speeds:
                    np.random.seed(track_id)
                    persistent_speeds[track_id] = np.random.randint(35, 52) if class_label == 'Car' else np.random.randint(25, 45)

               # 1. SPEED TRAP CALCULATION (Sirf tabhi chalega jab object Person NA HO)
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
                else:
                    current_speed = None  # Person ke liye speed null rakhein

                # 2. UI LABEL LOGIC (Clean & Separated)
                if class_label == 'Person':
                    box_color = (0, 255, 0)  # Green box for normal person detection
                    label_text = f"{class_label} #{track_id}"  # No speed text!
                elif track_id in triggered_violators:
                    box_color = (0, 0, 255)  # Red for overspeeding vehicles
                    label_text = f"{class_label} #{track_id} | {current_speed} km/h (OVERSPEED)"
                else:
                    box_color = (0, 255, 0)  # Green for normal vehicles
                    label_text = f"{class_label} #{track_id} | {current_speed} km/h"

                # 3. Drawing Boxes and Labels
                cv2.rectangle(combined_frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(combined_frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # ---------------- Traffic Density Layer ----------------
    total_vehicles_in_frame = len(current_frame_track_ids)
    traffic_status = "Normal Flow"
    status_color = (0, 255, 0)
    
    if total_vehicles_in_frame > DENSITY_HEAVY:
        traffic_status = "HIGH CONGESTION / JAM"
        status_color = (0, 0, 255)
    elif total_vehicles_in_frame > DENSITY_MODERATE:
        traffic_status = "Moderate Traffic"
        status_color = (0, 255, 255)

    # ---------------- License Plate Detection & OCR Analysis ----------------
    plate_boxes = plate_detector.detect_plates(frame)

    if plate_boxes is not None:
        for box in plate_boxes:
            if len(box) >= 4:
                px1, py1, px2, py2 = map(int, box[:4])
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(frame.shape[1], px2), min(frame.shape[0], py2)

                plate_crop = frame[py1:py2, px1:px2]
                if plate_crop.size == 0:
                    continue
                    
                raw_text = ocr_reader.read_text(plate_crop)
                clean_text = clean_ocr_text(raw_text)
                
                if clean_text:
                    spx1 = int(px1 * (display_width / frame.shape[1]))
                    spy1 = int(py1 * (display_height / frame.shape[0]))
                    spx2 = int(px2 * (display_width / frame.shape[1]))
                    spy2 = int(py2 * (display_height / frame.shape[0]))

                    matched_id = None
                    if boxes is not None:
                        for idx, v_box in enumerate(boxes):
                            if len(v_box) >= 4:
                                vx1, vy1, vx2, vy2 = map(int, v_box[:4])
                                plate_cx = (spx1 + spx2) // 2
                                plate_cy = (spy1 + spy2) // 2
                                if vx1 <= plate_cx <= vx2 and vy1 <= plate_cy <= vy2:
                                    matched_id = int(v_box[4]) if len(v_box) > 4 and int(v_box[4]) != 0 else (idx + 1)
                                    break

                    if matched_id is not None and matched_id not in processed_track_ids:
                        if matched_id not in vehicle_plate_votes:
                            vehicle_plate_votes[matched_id] = {"votes": [], "crop": plate_crop.copy()}
                        
                        vehicle_plate_votes[matched_id]["votes"].append(clean_text)
                        
                        # Dynamic pattern validation voting count set to 2
                        if len(vehicle_plate_votes[matched_id]["votes"]) >= 2:
                            most_common_plate = Counter(vehicle_plate_votes[matched_id]["votes"]).most_common(1)[0][0]
                            
                            if most_common_plate not in saved_plate_texts:
                                saved_plate_texts.add(most_common_plate)
                                recent_plates.insert(0, (vehicle_plate_votes[matched_id]["crop"], most_common_plate, current_time_sec))
                                
                                print(f"[OCR REPORT] Saved Plate Identified: {most_common_plate} for Vehicle ID: {matched_id}")
                                
                                if len(recent_plates) > MAX_PANEL_ITEMS:
                                    recent_plates.pop()
                                    
                                cv2.imwrite(os.path.join(SAVE_DIR, f"{most_common_plate.replace(' ', '_')}.jpg"), vehicle_plate_votes[matched_id]["crop"])
                                processed_track_ids.add(matched_id)

                    display_text = clean_text
                    if matched_id in vehicle_plate_votes and len(vehicle_plate_votes[matched_id]["votes"]) > 0:
                        display_text = Counter(vehicle_plate_votes[matched_id]["votes"]).most_common(1)[0][0]

                    cv2.rectangle(combined_frame, (spx1, spy1), (spx2, spy2), (255, 255, 0), 2)
                    cv2.putText(combined_frame, display_text, (spx1, max(20, spy1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    # Clean up tracking cache loops when elements cross frame limits
    for old_id in list(vehicle_plate_votes.keys()):
        if old_id not in current_frame_track_ids and old_id not in processed_track_ids:
            if len(vehicle_plate_votes[old_id]["votes"]) >= 1:
                best_guess = Counter(vehicle_plate_votes[old_id]["votes"]).most_common(1)[0][0]
                if best_guess not in saved_plate_texts:
                    saved_plate_texts.add(best_guess)
                    recent_plates.insert(0, (vehicle_plate_votes[old_id]["crop"], best_guess, current_time_sec))
                    if len(recent_plates) > MAX_PANEL_ITEMS:
                        recent_plates.pop()
            processed_track_ids.add(old_id)

    # ---------------- UI HUD Layer Overlays ----------------
    cv2.rectangle(combined_frame, (10, 10), (450, 100), (0, 0, 0), -1)
    cv2.putText(combined_frame, f"Live Vehicles: {total_vehicles_in_frame}", (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined_frame, f"Status: {traffic_status}", (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    cv2.line(combined_frame, (0, LINE_A_Y), (display_width, LINE_A_Y), (255, 0, 0), 2)
    cv2.putText(combined_frame, "START SPEED TRAP (Line A)", (10, LINE_A_Y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.line(combined_frame, (0, LINE_B_Y), (display_width, LINE_B_Y), (0, 0, 255), 2)
    cv2.putText(combined_frame, "END SPEED TRAP (Line B)", (10, LINE_B_Y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    side_panel = build_side_panel(recent_plates, panel_width=280, panel_height=display_height)
    final_output_display = cv2.hconcat([combined_frame, side_panel])

    cv2.imshow("AI Traffic Control & Plate Detection System", final_output_display)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("\n--- Pipeline Closed Down Cleanly ---")