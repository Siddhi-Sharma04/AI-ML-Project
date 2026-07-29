import cv2

def show_frame(frame):

    cv2.imshow("AI Traffic System", frame)

def close():

    cv2.destroyAllWindows()

def levenshtein_distance(s1, s2):
    """
    Standard space-optimized edit distance calculator.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def find_canonical_plate(plate, saved_plates, max_distance=2):
    """
    Checks if a plate matches any already-saved plates within Levenshtein max_distance.
    If matching plate is found, returns that canonical representation.
    """
    if not plate:
        return plate
        
    p_clean = plate.replace(" ", "").upper()
    for saved in saved_plates:
        if not saved:
            continue
        s_clean = saved.replace(" ", "").upper()
        if levenshtein_distance(p_clean, s_clean) <= max_distance:
            return saved
    return plate