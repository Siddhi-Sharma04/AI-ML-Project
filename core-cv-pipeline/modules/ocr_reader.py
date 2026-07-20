import cv2
# Note: EasyOCR parse loop (Agar PaddleOCR use kar rahe ho toh imports badal lena)
import easyocr

class OCRReader:
    def __init__(self, languages=['en']):
        """
        EasyOCR Model Initialization Engine.
        """
        print(f"[INFO] Initializing EasyOCR Engine for languages: {languages}")
        self.reader = easyocr.Reader(languages, gpu=True) # GPU use karne ke liye True, nahi toh False
        
    def read_text(self, plate_crop):
        """
        Cropped plate image string text and confidence score data extract karega.
        """
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0
            
        results = self.reader.readtext(plate_crop)
        if len(results) > 0:
            # Sabse highest confidence score wala text aur confidence return karein
            # EasyOCR format: [([x,y arrays], text, confidence)]
            return results[0][1], results[0][2]
        return "", 0.0