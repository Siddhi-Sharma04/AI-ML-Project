import cv2
import numpy as np
import json
import os

class ZebraObstructionDetector:
    def __init__(self, config_path):
        """
        Zebra Obstruction Detection Module using dwell time tracking and dual-mode polygon intersection.
        """
        self.zones = {}
        # Dwell tracking database format: 
        # { (track_id, zone_id): { "entry_time": float, "flagged": bool } }
        self.dwell_tracker = {}
        self.load_zones(config_path)

    def load_zones(self, config_path):
        """
        Loads zebra crossing polygons, detection modes, and dwell thresholds from config/zones.json.
        """
        if not os.path.exists(config_path):
            print(f"[WARNING] Zones config file not found at: {config_path}")
            return
            
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                
            for zone_config in data.get("zones", []):
                if zone_config.get("type") != "zebra_crossing":
                    continue
                    
                zone_id = zone_config.get("zone_id")
                polygon_coords = zone_config.get("polygon")
                mode = zone_config.get("mode", "both").lower().strip()
                dwell_threshold = float(zone_config.get("dwell_threshold_seconds", 12.0))
                
                polygon = np.array(polygon_coords, dtype=np.int32)
                
                # Precompute the bounding box of the zebra crossing polygon for IoU calculations
                poly_x1 = int(min(pt[0] for pt in polygon_coords))
                poly_y1 = int(min(pt[1] for pt in polygon_coords))
                poly_x2 = int(max(pt[0] for pt in polygon_coords))
                poly_y2 = int(max(pt[1] for pt in polygon_coords))
                
                self.zones[zone_id] = {
                    "polygon": polygon,
                    "bbox": (poly_x1, poly_y1, poly_x2, poly_y2),
                    "mode": mode,
                    "dwell_threshold": dwell_threshold
                }
            print(f"[INFO] Loaded {len(self.zones)} Zebra Crossing zones for obstruction detection.")
        except Exception as e:
            print(f"[ERROR] Failed to load zebra zones config: {e}")

    def compute_iou(self, bbox_a, bbox_b):
        """
        Calculates Intersection over Union (IoU) of two bounding boxes.
        """
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        
        if ix2 > ix1 and iy2 > iy1:
            intersection_area = (ix2 - ix1) * (iy2 - iy1)
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            union_area = area_a + area_b - intersection_area
            return intersection_area / union_area if union_area > 0 else 0.0
        return 0.0

    def check_zebra_obstruction(self, track_id, vehicle_box, current_time):
        """
        Evaluates zebra crossing obstruction. Returns True ONLY on the transition frame where
        a vehicle has continuously obstructed a zebra zone for longer than its configured dwell threshold.
        
        Args:
            track_id (int): Unique track identifier.
            vehicle_box (tuple): Bounding box of the vehicle (x1, y1, x2, y2).
            current_time (float): Current frame timestamp in seconds.
        """
        x1, y1, x2, y2 = vehicle_box
        
        # --- VIVA EXPLANATION: GEOMETRY DETECTION MODES ---
        # 1. Point Check: Bottom-center point of the vehicle represents contact with the road.
        #    If wheels are on the zebra crossing, bottom-center is inside the polygon.
        # 2. IoU Check: Sometimes large vehicles (trucks/buses) obstruct the crossing but their bottom-center
        #    coordinate falls outside the polygon boundaries. Calculating the overlap (IoU) of the vehicle box 
        #    with the zebra polygon's bounding box catches these cases.
        bottom_center = ((x1 + x2) // 2, y2)
        
        violation_triggered = False

        for zone_id, zone_data in self.zones.items():
            mode = zone_data["mode"]
            polygon = zone_data["polygon"]
            zone_bbox = zone_data["bbox"]
            dwell_threshold = zone_data["dwell_threshold"]
            
            # 1. Evaluate point overlap
            point_inside = cv2.pointPolygonTest(polygon, bottom_center, False) >= 0
            
            # 2. Evaluate IoU overlap (threshold > 0.01 signifies non-trivial box overlap)
            iou_val = self.compute_iou(vehicle_box, zone_bbox)
            iou_inside = iou_val > 0.01
            
            # 3. Determine if vehicle is obstructing based on mode
            is_obstructing = False
            if mode == "point":
                is_obstructing = point_inside
            elif mode == "iou":
                is_obstructing = iou_inside
            elif mode == "both":
                is_obstructing = point_inside or iou_inside
                
            state_key = (track_id, zone_id)
            
            if is_obstructing:
                if state_key not in self.dwell_tracker:
                    # Vehicle just entered the zebra zone, start tracking dwell time
                    self.dwell_tracker[state_key] = {
                        "entry_time": current_time,
                        "flagged": False
                    }
                else:
                    state = self.dwell_tracker[state_key]
                    if state["flagged"]:
                        continue  # Already flagged for this zone, bypass
                        
                    dwell_duration = current_time - state["entry_time"]
                    if dwell_duration >= dwell_threshold:
                        state["flagged"] = True
                        violation_triggered = True
            else:
                # Vehicle left the zebra crossing: clear dwell timer immediately
                self.dwell_tracker.pop(state_key, None)
                
        return violation_triggered

    def clean_track(self, track_id):
        """
        Cleans tracking states when a vehicle leaves the display screen.
        """
        for zone_id in list(self.zones.keys()):
            self.dwell_tracker.pop((track_id, zone_id), None)
