import base64
import cv2
import numpy as np
import logging
from ultralytics import YOLO

# 1. Secure Logging Setup: Keeps errors on the server, hidden from the frontend
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Load the AI model once when the server starts
model = YOLO('yolov8n.pt')

# Security Constant: Max payload size (approx. 2MB). Drops massive malicious payloads.
MAX_PAYLOAD_SIZE = 2 * 1024 * 1024 

def process_frame_secure(base64_string):
    """
    Takes a base64 image string, securely sanitizes it, runs YOLOv8, 
    and returns a navigation command.
    """
    
    # SECURITY CHECK 1: Prevent DoS via Memory Exhaustion
    if not isinstance(base64_string, str) or len(base64_string) > MAX_PAYLOAD_SIZE:
        logging.warning("Security trigger: Payload rejected (Too large or invalid type).")
        return "Path clear." # Fail silently to the user

    try:
        # SECURITY CHECK 2: Sanitize the HTML Canvas header if present
        if ',' in base64_string:
            base64_data = base64_string.split(',')[1]
        else:
            base64_data = base64_string

        # SECURITY CHECK 3: Strict decoding (rejects injected non-base64 characters)
        img_bytes = base64.b64decode(base64_data, validate=True)
        
        # Convert memory buffer to an OpenCV image safely
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            logging.warning("Security trigger: Decoded image is corrupt/empty.")
            return "Path clear."

        #  AI & MATH LOGIC 
        height, width, _ = img.shape
        total_screen_area = height * width
        
        # Run inference (verbose=False keeps the server terminal clean)
        results = model.predict(img, verbose=False)
        
        for r in results:
            for box in r.boxes:
                # Get the class ID of the object (0 = person, 56 = chair, etc.)
                cls_id = int(box.cls[0])
                
                # care about specific obstacles (fadd more IDs as needed)
                if cls_id in [0, 56]: 
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    
                    # Calculate how much of the screen the object takes up
                    obstacle_area = (x2 - x1) * (y2 - y1)
                    coverage_percentage = obstacle_area / total_screen_area
                    
                    # If the obstacle covers more than 40% of the screen, it's too close!
                    if coverage_percentage > 0.40:
                        return "Obstacle ahead. Step left."
                        
        return "Path clear."

    except Exception as e:
        # SECURITY CHECK 4: Catch all fatal errors but do not leak the traceback
        logging.error(f"Internal processing error safely caught: {str(e)}")
        return "Path clear."