import cv2
import torch
import cv2
import re
import pytesseract
import numpy as np
from PIL import Image
from ultralytics import YOLO

pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
model = YOLO('model.pt')

img = cv2.imread('image/image2.png')
img_resized = cv2.resize(img, (640, 480))

results = model(img_resized)

def extract_text_from_image(image_path):
    image = Image.open(image_path).convert('L')
    image_cv = np.array(image)
    _, binary_image = cv2.threshold(image_cv, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_image = cv2.resize(binary_image, None, fx=2, fy=2)
    image = Image.fromarray(binary_image)

    custom_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(image, config=custom_config)

    text = re.sub(r'\W+', '', text)
    return text


if len(results[0].boxes) > 0:
    
    box = results[0].boxes[0] 
    xyxy = box.xyxy[0].cpu().numpy() 

    x1, y1, x2, y2 = map(int, xyxy) 
    license_plate_img = img_resized[y1:y2, x1:x2] 

    cv2.imwrite('output/license_plate.jpg', license_plate_img)
    image_path = 'output/license_plate.jpg'
    extracted_text = extract_text_from_image(image_path)
    print("license plate detecte : ",extracted_text)
    
else:
    print("No license plate detected.")





