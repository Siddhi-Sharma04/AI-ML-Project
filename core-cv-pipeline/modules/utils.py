import cv2

def show_frame(frame):

    cv2.imshow("AI Traffic System", frame)

def close():

    cv2.destroyAllWindows()