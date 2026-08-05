import logging
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Union, Set
from collections import Counter
import numpy as np

# Configure logging
logger = logging.getLogger("DecisionEngine")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] [DecisionEngine] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

@dataclass
class RawViolation:
    """Represents a raw violation detected in a single frame."""
    track_id: int
    violation_type: str
    confidence: float
    timestamp: float
    bbox: Tuple[int, int, int, int]
    vehicle_crop: Optional[np.ndarray] = None
    plate_crop: Optional[np.ndarray] = None
    plate_number: Optional[str] = None

@dataclass
class ConfirmedViolation:
    """Represents a consolidated and confirmed violation ready for challan generation."""
    track_id: int
    plate_number: Optional[str]
    violations: List[str]  # Consolidated list of violation types, sorted by severity
    timestamp: float       # Sample timestamp of the peak violation frame
    bbox: Tuple[int, int, int, int]  # Bounding box at the peak confidence frame
    peak_confidence: float # Maximum confidence score among the confirmed violations
    severity_score: int    # Numerical score based on the highest severity violation
    vehicle_crop: Optional[np.ndarray] = field(default=None, repr=False)
    plate_crop: Optional[np.ndarray] = field(default=None, repr=False)

class DecisionEngine:
    def __init__(
        self,
        cooldown_seconds: float = 300.0,
        min_violation_frames: int = 5,
        severity_ranking: Optional[List[str]] = None,
        cleanup_interval_seconds: float = 10.0
    ):
        """
        Initializes the DecisionEngine.

        Args:
            cooldown_seconds (float): Cool-down window in seconds to suppress duplicate challans for a vehicle.
            min_violation_frames (int): Minimum frames a violation must be active to be confirmed.
            severity_ranking (List[str], optional): Custom violation types ordered from highest to lowest severity.
            cleanup_interval_seconds (float): Interval in seconds to clean up stale tracks and cooldown caches.
        """
        self.cooldown_seconds = cooldown_seconds
        self.min_violation_frames = min_violation_frames
        
        # Default severity ranking (highest to lowest)
        self.severity_ranking = severity_ranking or [
            "WRONG WAY",
            "OVERSPEEDING",
            "TRIPLE RIDING",
            "NO HELMET",
            "ZEBRA OBSTRUCTION"
        ]
        
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self.last_cleanup_time = 0.0

        # In-memory tracking state: track_id -> state dictionary
        self.track_states: Dict[int, Dict[str, Any]] = {}

        # Cooldown caches: identifier -> confirmation timestamp
        self.cooldown_by_track_id: Dict[int, float] = {}
        self.cooldown_by_plate: Dict[str, float] = {}

    def _get_severity_score(self, violation_type: str) -> int:
        """Returns the numerical severity score. Higher score means higher severity."""
        try:
            # Score is higher for lower index in severity_ranking
            idx = self.severity_ranking.index(violation_type.upper().strip())
            return len(self.severity_ranking) - idx
        except ValueError:
            return 0

    def _cleanup_cache(self, current_time: float):
        """Removes expired entries from cooldown caches and stale track states."""
        # 1. Clean cooldown caches
        expired_tracks = [tid for tid, ts in self.cooldown_by_track_id.items() 
                          if current_time - ts > self.cooldown_seconds]
        for tid in expired_tracks:
            del self.cooldown_by_track_id[tid]

        expired_plates = [plate for plate, ts in self.cooldown_by_plate.items() 
                          if current_time - ts > self.cooldown_seconds]
        for plate in expired_plates:
            del self.cooldown_by_plate[plate]

        # 2. Clean stale track states (inactive for more than cooldown_seconds or 60 seconds)
        inactive_threshold = max(60.0, self.cooldown_seconds)
        stale_tracks = []
        for tid, state in self.track_states.items():
            if current_time - state["last_seen"] > inactive_threshold:
                stale_tracks.append(tid)
        
        for tid in stale_tracks:
            del self.track_states[tid]

        self.last_cleanup_time = current_time
        logger.debug(
            f"Cleaned caches. Active track states: {len(self.track_states)}, "
            f"Track cooldowns: {len(self.cooldown_by_track_id)}, "
            f"Plate cooldowns: {len(self.cooldown_by_plate)}"
        )

    def process_frame_detections(
        self,
        frame_idx: int,
        timestamp: float,
        detections: Union[Dict[int, Dict[str, Any]], List[Dict[str, Any]]],
        violations: Union[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]
    ) -> List[ConfirmedViolation]:
        """
        Processes frame detections and raw violations to return confirmed consolidated violations.

        Args:
            frame_idx (int): Current frame index.
            timestamp (float): Current frame timestamp in seconds (epoch or video relative).
            detections (dict or list): Active vehicle detections.
                If dict: {track_id: {bbox, class, confidence, crop, plate_number, plate_crop}}
                If list: List of dicts, each having a 'track_id' key.
            violations (list or dict): Raw violation signals for the current frame.
                Format 1: List of raw violation dicts (e.g. [{'track_id', 'violation_type', 'confidence'}])
                Format 2: Dict of track_id -> List of violation dicts/strings.

        Returns:
            List[ConfirmedViolation]: List of newly confirmed and consolidated violations.
        """
        # Periodic cleanup of in-memory caches
        if timestamp - self.last_cleanup_time > self.cleanup_interval_seconds:
            self._cleanup_cache(timestamp)

        # Standardize detections into Dict[track_id, Dict]
        std_detections: Dict[int, Dict[str, Any]] = {}
        if isinstance(detections, list):
            for det in detections:
                if isinstance(det, dict) and "track_id" in det:
                    std_detections[int(det["track_id"])] = det
        elif isinstance(detections, dict):
            std_detections = {int(k): v for k, v in detections.items()}

        # Standardize violations into Dict[track_id, List[RawViolation]]
        std_violations: Dict[int, List[RawViolation]] = {}
        
        if isinstance(violations, list):
            for viol in violations:
                if not isinstance(viol, dict) or "track_id" not in viol or "violation_type" not in viol:
                    continue
                tid = int(viol["track_id"])
                v_type = str(viol["violation_type"]).upper().strip()
                conf = float(viol.get("confidence", 1.0))
                
                # Fetch spatial and visual details from detections if available
                det_info = std_detections.get(tid, {})
                bbox = det_info.get("bbox", (0, 0, 0, 0))
                veh_crop = det_info.get("crop", None)
                plate_crop = det_info.get("plate_crop", None)
                plate_num = det_info.get("plate_number", None)

                raw_v = RawViolation(
                    track_id=tid,
                    violation_type=v_type,
                    confidence=conf,
                    timestamp=timestamp,
                    bbox=bbox,
                    vehicle_crop=veh_crop,
                    plate_crop=plate_crop,
                    plate_number=plate_num
                )
                std_violations.setdefault(tid, []).append(raw_v)
                
        elif isinstance(violations, dict):
            for k, val_list in violations.items():
                tid = int(k)
                det_info = std_detections.get(tid, {})
                bbox = det_info.get("bbox", (0, 0, 0, 0))
                veh_crop = det_info.get("crop", None)
                plate_crop = det_info.get("plate_crop", None)
                plate_num = det_info.get("plate_number", None)

                for item in val_list:
                    if isinstance(item, str):
                        v_type = item.upper().strip()
                        conf = 1.0
                    elif isinstance(item, dict):
                        v_type = str(item.get("violation_type", item.get("type", ""))).upper().strip()
                        conf = float(item.get("confidence", 1.0))
                    else:
                        continue
                    
                    if not v_type:
                        continue

                    raw_v = RawViolation(
                        track_id=tid,
                        violation_type=v_type,
                        confidence=conf,
                        timestamp=timestamp,
                        bbox=bbox,
                        vehicle_crop=veh_crop,
                        plate_crop=plate_crop,
                        plate_number=plate_num
                    )
                    std_violations.setdefault(tid, []).append(raw_v)

        confirmed_violations_this_frame: List[ConfirmedViolation] = []

        # Process each active detection
        for tid, det_info in std_detections.items():
            # Check if this track ID is currently in cooldown
            if tid in self.cooldown_by_track_id:
                continue

            # Update last seen timestamp
            if tid not in self.track_states:
                self.track_states[tid] = {
                    "track_id": tid,
                    "class_label": det_info.get("class", "Vehicle"),
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    # violation_type -> {count, peak_confidence, best_frame_ts, best_bbox, best_vehicle_crop, best_plate_crop}
                    "violation_buffers": {},
                    "plate_votes": Counter(),
                    "emitted": False
                }
            
            state = self.track_states[tid]
            state["last_seen"] = timestamp

            # Accumulate plate number if provided
            current_plate = det_info.get("plate_number")
            if current_plate:
                clean_plate = "".join(e for e in current_plate if e.isalnum()).upper().strip()
                if clean_plate:
                    state["plate_votes"][clean_plate] += 1
            
            # Retrieve the most voted plate number so far
            best_plate = None
            if state["plate_votes"]:
                best_plate = state["plate_votes"].most_common(1)[0][0]

            # If the best plate number is in global cooldown, suppress violations
            if best_plate and best_plate in self.cooldown_by_plate:
                continue

            # Update violation states
            active_viol_types_this_frame = set()
            frame_viols = std_violations.get(tid, [])

            for raw_v in frame_viols:
                v_type = raw_v.violation_type
                active_viol_types_this_frame.add(v_type)

                # Initialize violation buffer if first time
                if v_type not in state["violation_buffers"]:
                    state["violation_buffers"][v_type] = {
                        "count": 0,
                        "peak_confidence": 0.0,
                        "best_frame_ts": timestamp,
                        "best_bbox": raw_v.bbox,
                        "best_vehicle_crop": raw_v.vehicle_crop,
                        "best_plate_crop": raw_v.plate_crop
                    }

                buf = state["violation_buffers"][v_type]
                buf["count"] += 1

                # Update metadata if peak confidence is exceeded
                if raw_v.confidence >= buf["peak_confidence"]:
                    buf["peak_confidence"] = raw_v.confidence
                    buf["best_frame_ts"] = raw_v.timestamp
                    buf["best_bbox"] = raw_v.bbox
                    if raw_v.vehicle_crop is not None:
                        buf["best_vehicle_crop"] = raw_v.vehicle_crop
                    if raw_v.plate_crop is not None:
                        buf["best_plate_crop"] = raw_v.plate_crop

            # Check if any violation has reached the min_violation_frames threshold
            # and we haven't emitted a confirmed violation for this track yet.
            if not state["emitted"]:
                confirmed_types = []
                for v_type, buf in state["violation_buffers"].items():
                    if buf["count"] >= self.min_violation_frames:
                        confirmed_types.append(v_type)

                if confirmed_types:
                    # We have at least one confirmed violation!
                    # Consolidate all confirmed violations for this track
                    
                    # Sort confirmed types by severity ranking
                    confirmed_types.sort(key=lambda t: self._get_severity_score(t), reverse=True)
                    highest_severity_type = confirmed_types[0]
                    highest_severity_score = self._get_severity_score(highest_severity_type)

                    # Gather the best metadata from the violation type with the highest peak confidence
                    best_violation_buf = max(
                        [state["violation_buffers"][t] for t in confirmed_types],
                        key=lambda b: b["peak_confidence"]
                    )

                    # Retrieve best crops from either the peak frame, or general state
                    vehicle_crop = best_violation_buf["best_vehicle_crop"]
                    plate_crop = best_violation_buf["best_plate_crop"]
                    
                    # Fallback plate crops if not present in peak violation buffer
                    if plate_crop is None:
                        plate_crop = det_info.get("plate_crop", None)
                    if vehicle_crop is None:
                        vehicle_crop = det_info.get("crop", None)

                    # Instantiate ConfirmedViolation
                    confirmed_viol = ConfirmedViolation(
                        track_id=tid,
                        plate_number=best_plate,
                        violations=confirmed_types,
                        timestamp=best_violation_buf["best_frame_ts"],
                        bbox=best_violation_buf["best_bbox"],
                        peak_confidence=best_violation_buf["peak_confidence"],
                        severity_score=highest_severity_score,
                        vehicle_crop=vehicle_crop,
                        plate_crop=plate_crop
                    )

                    # Store in cooldown cache to prevent duplicates
                    self.cooldown_by_track_id[tid] = timestamp
                    if best_plate:
                        self.cooldown_by_plate[best_plate] = timestamp

                    # Mark state as emitted
                    state["emitted"] = True
                    confirmed_violations_this_frame.append(confirmed_viol)
                    logger.info(
                        f"CONFIRMED VIOLATION for vehicle ID {tid} (Plate: {best_plate}): "
                        f"{confirmed_types} | Peak Conf: {confirmed_viol.peak_confidence:.2f} | "
                        f"Severity Score: {highest_severity_score}"
                    )

        return confirmed_violations_this_frame
