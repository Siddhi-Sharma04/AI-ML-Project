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
        
    def read_text(self, plate_crop, conf_threshold=0.15):
        """
        Cropped plate image string text and confidence score data extract karega.
        Applies padding, upscaling, sharpening, and CLAHE preprocessing.
        """
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0
            
        # 1. Add border padding to prevent characters on edge from being clipped/distorted
        padded = cv2.copyMakeBorder(plate_crop, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        
        # 2. Grayscale conversion
        gray = cv2.cvtColor(padded, cv2.COLOR_BGR2GRAY)
        
        # 3. Upscale crop by 3x using cubic interpolation to make small text larger and sharper
        h, w = gray.shape[:2]
        resized = cv2.resize(gray, (w * 3, h * 3), interpolation=cv2.INTER_CUBIC)
        
        # 4. Apply bilateral filtering to reduce sensor noise while preserving edge boundaries
        smoothed = cv2.bilateralFilter(resized, 9, 75, 75)
        
        # 5. Apply CLAHE contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(smoothed)
        
        # 6. Apply Unsharp Masking (sharpening) to enhance character boundaries
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 3)
        preprocessed = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
        
        results = self.reader.readtext(preprocessed)
        if len(results) > 0:
            # EasyOCR format: [([x,y arrays], text, confidence)]
            text = results[0][1]
            confidence = results[0][2]
            if confidence >= conf_threshold:
                return text, confidence
        return "", 0.0