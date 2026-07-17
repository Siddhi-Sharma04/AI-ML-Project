import numpy as np

def process_speed_trap(
    track_id, center_y, frame_count, class_label, box_coords,
    LINE_A_Y, LINE_B_Y, SPEED_LIMIT, MAX_REALISTIC_SPEED,
    speed_timers, active_violations, triggered_violators, persistent_speeds
):
    """
    Ekdam safe speed trap calculation engine. 
    Ye original dicts ko in-place modify karega bina main logic break kiye.
    """
    # 1. Agar vehicle Line A aur Line B ke beech me hai
    if LINE_A_Y <= center_y < LINE_B_Y:
        if track_id not in speed_timers:
            speed_timers[track_id] = frame_count
            if track_id not in active_violations:
                active_violations[track_id] = {
                    "speed": persistent_speeds[track_id],
                    "box_coords": box_coords
                }
                
    # 2. Agar vehicle Line B ko cross kar gaya hai
    elif center_y >= LINE_B_Y:
        if track_id in speed_timers and track_id not in triggered_violators:
            start_frame = speed_timers[track_id]
            total_frames = frame_count - start_frame
            
            if total_frames > 12: 
                time_taken = total_frames / 30.0  
                ROAD_DISTANCE = 12.0  
                speed_mps = ROAD_DISTANCE / time_taken
                calculated_speed = int(speed_mps * 3.6)
                
                speed_kmh = calculated_speed if calculated_speed < MAX_REALISTIC_SPEED else persistent_speeds[track_id]
                
                if speed_kmh > SPEED_LIMIT:
                    triggered_violators.add(track_id)
                    if track_id in active_violations:
                        active_violations[track_id]["speed"] = speed_kmh
                    print(f"[VIOLATION ALERT] {class_label} ID {track_id} OVERSPEEDING at {speed_kmh} km/h!")
                else:
                    if track_id in active_violations:
                        active_violations[track_id]["speed"] = speed_kmh

    # Return current speed taaki UI variables properly set ho sakein
    return active_violations.get(track_id, {}).get("speed", persistent_speeds[track_id])