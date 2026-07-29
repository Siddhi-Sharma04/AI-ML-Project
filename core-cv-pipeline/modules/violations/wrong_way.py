import cv2
import numpy as np
import json
import os

class WrongWayDetector:
    def __init__(self, config_path):
        """
        Wrong-Way Detection Module using designated flow polygons and vector geometry.
        """
        self.zones = {}
        # Track history format: 
        # { track_id: { "first_frame": int, "first_pos": (cx, cy), "buffer": [(cx, cy), ...], "flagged": bool } }
        self.history = {}
        self.load_zones(config_path)

    def load_zones(self, config_path):
        """
        Loads lane polygons and expected flow direction vectors from config/zones.json.
        """
        if not os.path.exists(config_path):
            print(f"[WARNING] Zones config file not found at: {config_path}")
            return
            
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
                
            for zone_config in data.get("zones", []):
                if zone_config.get("type") == "zebra_crossing":
                    continue
                zone_id = zone_config.get("zone_id")
                polygon_coords = zone_config.get("polygon")
                raw_dir = zone_config.get("expected_direction")
                
                # Convert coords list to numpy array for OpenCV's polygon checks
                polygon = np.array(polygon_coords, dtype=np.int32)
                
                # Convert expected_direction (either simple string or custom vector) to a normalized unit vector
                expected_vector = [0.0, 0.0]
                if isinstance(raw_dir, str):
                    direction_str = raw_dir.lower().strip()
                    if direction_str == "down":
                        # In image/OpenCV coordinate system: y increases downward.
                        expected_vector = [0.0, 1.0]
                    elif direction_str == "up":
                        expected_vector = [0.0, -1.0]
                    elif direction_str == "left":
                        expected_vector = [-1.0, 0.0]
                    elif direction_str == "right":
                        expected_vector = [1.0, 0.0]
                elif isinstance(raw_dir, list) and len(raw_dir) == 2:
                    expected_vector = [float(raw_dir[0]), float(raw_dir[1])]
                
                # Normalize the vector to unit length (|v| = 1)
                vector_magnitude = np.linalg.norm(expected_vector)
                if vector_magnitude > 0:
                    expected_vector = [expected_vector[0] / vector_magnitude, expected_vector[1] / vector_magnitude]
                
                self.zones[zone_id] = {
                    "polygon": polygon,
                    "expected_direction": expected_vector
                }
            print(f"[INFO] Loaded {len(self.zones)} zones for Wrong Way detection.")
        except Exception as e:
            print(f"[ERROR] Failed to load zones config: {e}")

    def get_zone_for_point(self, pt):
        """
        Uses cv2.pointPolygonTest to find which lane polygon contains the vehicle center point (x, y).
        Returns the matching zone_id or None if point lies outside all polygons.
        """
        for zone_id, zone_data in self.zones.items():
            # pointPolygonTest returns positive value if inside, negative if outside, 0 if on edge.
            dist = cv2.pointPolygonTest(zone_data["polygon"], pt, False)
            if dist >= 0:
                return zone_id
        return None

    def check_wrong_way(self, track_id, current_center, frame_num, min_frames=10, min_displacement=20, cosine_threshold=-0.7):
        """
        Checks if the vehicle is moving against the designated traffic flow in its zone.
        
        Args:
            track_id (int): Unique tracking identifier.
            current_center (tuple): (cx, cy) coordinates of vehicle center.
            frame_num (int): Current frame index.
            min_frames (int): Minimum frames to track before evaluating (prevents noise at entry).
            min_displacement (int): Minimum movement distance (in pixels) to ignore stationary jitter.
            cosine_threshold (float): Similarity limit. Strong negative similarity (<= -0.7) flags violation.
            
        Returns:
            bool: True if wrong-way violation is detected and confirmed, False otherwise.
        """
        # Resolve which zone/lane the vehicle is currently traveling in
        zone_id = self.get_zone_for_point(current_center)
        if zone_id is None or zone_id not in self.zones:
            return False

        zone_data = self.zones[zone_id]
        expected_dir = zone_data["expected_direction"]

        # Initialize tracking history for new vehicle IDs
        if track_id not in self.history:
            self.history[track_id] = {
                "first_frame": frame_num,
                "first_pos": current_center,
                "buffer": [current_center],
                "flagged": False
            }
            return False

        hist = self.history[track_id]
        if hist["flagged"]:
            return False  # Already flagged in this session, don't generate duplicate violations

        # Append current position to smoothing/rolling buffer and limit its length
        hist["buffer"].append(current_center)
        if len(hist["buffer"]) > 15:
            hist["buffer"].pop(0)

        # Ensure enough frames have accumulated to establish a solid trajectory direction
        total_frames_tracked = frame_num - hist["first_frame"]
        if total_frames_tracked < min_frames or len(hist["buffer"]) < min_frames:
            return False

        # Calculate total trajectory displacement vector between the earliest and latest buffer coordinates
        # This smoothing avoids frame-to-frame tracking jitter
        earliest_pos = hist["buffer"][0]
        latest_pos = hist["buffer"][-1]
        
        dx = latest_pos[0] - earliest_pos[0]
        dy = latest_pos[1] - earliest_pos[1]
        
        # Calculate straight-line distance traveled
        displacement_mag = np.hypot(dx, dy)
        if displacement_mag < min_displacement:
            # Vehicle is stationary or idling, not moving enough to determine wrong-way path
            return False

        # Normalize the actual movement vector to unit length
        actual_dir = [dx / displacement_mag, dy / displacement_mag]

        # --- VIVA EXPLANATION: COSINE SIMILARITY GEOMETRY ---
        # Cosine similarity is the dot product of two normalized unit vectors:
        #   cosine_sim = actual_dir . expected_dir = cos(theta)
        #
        # - If cosine_sim is close to 1: Vehicles are moving in the same direction (angle theta ~ 0 deg).
        # - If cosine_sim is close to 0: Movement is perpendicular (angle theta ~ 90 deg).
        # - If cosine_sim is close to -1: Vehicles are moving in opposite directions (angle theta ~ 180 deg).
        #
        # A threshold of <= -0.7 means theta >= 135 degrees (moving strongly against expected traffic flow).
        cosine_sim = (actual_dir[0] * expected_dir[0]) + (actual_dir[1] * expected_dir[1])

        if cosine_sim <= cosine_threshold:
            hist["flagged"] = True
            return True

        return False

    def clean_track(self, track_id):
        """
        Removes tracking history when vehicle leaves display frame to prevent memory leak.
        """
        self.history.pop(track_id, None)
