import cv2
import json
import os
import numpy as np

# Resolve path configurations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "videos", "no_helmet.mp4")
CONFIG_PATH = os.path.join(BASE_DIR, "config", "zones.json")

# State variables for mouse callback
points = []

def mouse_callback(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        # Left click: add vertex
        points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click: remove last vertex
        if points:
            points.pop()

def main():
    global points
    print("\n==============================================")
    print("AI TRAFFIC ZEBRA CROSSING ANNOTATION TOOL")
    print("==============================================")
    print("Instructions:")
    print("1. Click LEFT MOUSE BUTTON to add polygon corners on the frame.")
    print("2. Click RIGHT MOUSE BUTTON to undo the last clicked corner.")
    print("3. Press 's' to Save the polygon to zones.json.")
    print("4. Press 'c' to Clear all drawn corners.")
    print("5. Press 'q' or 'Esc' to Quit.")
    print("==============================================\n")

    # Ingest video frame
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video feed at: {VIDEO_PATH}")
        return

    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("[ERROR] Failed to read first frame of video.")
        return

    # Resize frame to match pipeline's display resolution (960x540)
    display_w, display_h = 960, 540
    frame_display = cv2.resize(frame, (display_w, display_h))

    cv2.namedWindow("Zebra Zone Annotator")
    cv2.setMouseCallback("Zebra Zone Annotator", mouse_callback)

    while True:
        img_copy = frame_display.copy()
        
        # Draw current points and connecting lines
        if len(points) > 0:
            for p in points:
                cv2.circle(img_copy, p, 4, (0, 0, 255), -1)
            if len(points) > 1:
                cv2.polylines(img_copy, [np.array(points, dtype=np.int32)], False, (0, 255, 255), 2)
            # If closing the polygon
            if len(points) >= 3:
                cv2.polylines(img_copy, [np.array(points, dtype=np.int32)], True, (0, 255, 0), 2)

        cv2.imshow("Zebra Zone Annotator", img_copy)
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == 27: # Esc or q
            print("[INFO] Quitting without saving.")
            break
        elif key == ord('c'):
            points = []
            print("[INFO] Cleared current vertices.")
        elif key == ord('s'):
            if len(points) < 3:
                print("[WARNING] A polygon requires at least 3 vertices to save!")
                continue
                
            # Input zone configuration details from terminal
            print("\n--- Enter Zone Properties ---")
            zone_id = input("Zone ID (e.g. zebra_1): ").strip()
            if not zone_id:
                zone_id = f"zebra_{np.random.randint(100, 999)}"
                
            print("Select Detection Mode:")
            print("1. point (checks vehicle bottom-center inside polygon)")
            print("2. iou (checks overlap of vehicle box with polygon bounding box)")
            print("3. both (checks both conditions, recommended)")
            mode_choice = input("Enter choice (1/2/3, default 3): ").strip()
            if mode_choice == "1":
                mode = "point"
            elif mode_choice == "2":
                mode = "iou"
            else:
                mode = "both"
                
            dwell_str = input("Dwell Threshold in seconds (default 12): ").strip()
            dwell_val = 12.0
            if dwell_str:
                try:
                    dwell_val = float(dwell_str)
                except ValueError:
                    pass

            # Load existing config zones
            config_data = {"zones": []}
            if os.path.exists(CONFIG_PATH):
                try:
                    with open(CONFIG_PATH, 'r') as f:
                        config_data = json.load(f)
                except Exception:
                    pass

            # Construct new zone record
            new_zone = {
                "zone_id": zone_id,
                "type": "zebra_crossing",
                "polygon": points,
                "mode": mode,
                "dwell_threshold_seconds": dwell_val
            }
            
            # Ensure "zones" array exists
            if "zones" not in config_data:
                config_data["zones"] = []
                
            # Avoid duplicate zone_id
            config_data["zones"] = [z for z in config_data["zones"] if z.get("zone_id") != zone_id]
            config_data["zones"].append(new_zone)

            # Write config back
            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            with open(CONFIG_PATH, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            print(f"\n[SUCCESS] Saved zone '{zone_id}' to: {CONFIG_PATH}!")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
