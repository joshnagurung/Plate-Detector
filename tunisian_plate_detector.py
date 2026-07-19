import cv2
import easyocr
import numpy as np
import re
import os
import torch
import sys
import time
from pathlib import Path
from flask import Flask, request, jsonify
import base64

# Try importing YOLO, but don't fail if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("YOLO module loaded successfully")
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO module not available. Falling back to traditional detection methods.")

class NepaliPlateDetector:
    def __init__(self):
        try:
            # Initialize CUDA if available
            if not torch.cuda.is_available():
                print("WARNING: CUDA is not available. Using CPU instead.")
                self.device = 'cpu'
            else:
                try:
                    torch.cuda.init()
                    self.device = 'cuda'
                    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                except Exception as e:
                    print(f"CUDA initialization failed: {e}")
                    self.device = 'cpu'

            # Initialize EasyOCR for Nepali (Devanagari script) + English plates.
            # Nepali plates mix Devanagari province/category text with digits; newer
            # embossed plates are often plain Roman digits. No recog_network override
            # here — EasyOCR picks the right Devanagari-group model automatically for
            # this language pair (unlike the old Arabic setup, which needed 'arabic_g1'
            # forced explicitly).
            gpu_status = True if self.device == 'cuda' else False
            self.reader = easyocr.Reader(['ne', 'en'], gpu=gpu_status)

            # Initialize YOLO model for license plate detection if available
            self.plate_detector = None
            if YOLO_AVAILABLE:
                try:
                    model_path = os.path.join(os.path.dirname(__file__), "model/license_plate_yolov8n.pt")
                    if os.path.exists(model_path):
                        self.plate_detector = YOLO(model_path)
                        print(f"YOLO model loaded from: {model_path}")
                    else:
                        print(f"YOLO model not found at {model_path}. Using general object detection.")
                        self.plate_detector = YOLO('yolov8n.pt')
                except Exception as e:
                    print(f"YOLO initialization failed: {e}")

            # Always initialize cascade classifier as backup
            cascade_path = os.path.join(os.path.dirname(__file__), "model/haarcascade_russian_plate_number.xml")
            if not os.path.exists(cascade_path):
                cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"

            if os.path.exists(cascade_path):
                self.plate_cascade = cv2.CascadeClassifier(cascade_path)
                print(f"Cascade classifier loaded from: {cascade_path}")
            else:
                self.plate_cascade = cv2.CascadeClassifier()

            # Devanagari script range, used to spot Nepali province/category text
            # (e.g. बागमती, प्रदेश २) on the plate, if present.
            self.devanagari_pattern = re.compile(r'[ऀ-ॿ]+')
            self.number_pattern = re.compile(r'\d+')

            # Parameters
            self.min_plate_area = 500
            self.min_confidence = 0.2

        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            sys.exit(1)

    def ensure_image_format(self, img):
        """Make sure image is in the correct format for processing"""
        try:
            if img is None or img.size == 0:
                print("Invalid image input")
                return None

            if len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if img.dtype != np.uint8:
                img = img.astype(np.uint8)

            return img
        except Exception as e:
            print(f"Error in image format conversion: {e}")
            return np.ones((100, 100, 3), dtype=np.uint8) * 255

    def detect_plate(self, img):
        """Detect license plates using available methods"""
        try:
            height, width = img.shape[:2]
            max_dimension = 1600
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                img = cv2.resize(img, None, fx=scale, fy=scale)

            detected_plates = []

            # Try YOLO detection if available
            if YOLO_AVAILABLE and self.plate_detector is not None:
                try:
                    detected_plates = self.detect_with_yolo(img)
                except Exception as e:
                    print(f"YOLO detection failed: {e}")

            # If no plates found or YOLO not available, try cascade
            if not detected_plates:
                try:
                    detected_plates = self.detect_with_cascade(img)
                except Exception as e:
                    print(f"Cascade detection failed: {e}")

            # If still no plates, try direct OCR on the full image
            if not detected_plates:
                try:
                    print("Attempting direct OCR on full image...")
                    h, w = img.shape[:2]
                    detected_plates.append((0, 0, w, h, 0.3))
                except Exception as e:
                    print(f"Direct OCR preparation failed: {e}")

            return detected_plates

        except Exception as e:
            print(f"Error in plate detection: {str(e)}")
            return []

    def detect_with_yolo(self, img):
        """Detect license plates using YOLO"""
        if not YOLO_AVAILABLE or self.plate_detector is None:
            return []

        try:
            results = self.plate_detector(img, conf=self.min_confidence)
            yolo_plates = []

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if box.conf.item() > self.min_confidence:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        w, h = x2 - x1, y2 - y1
                        aspect_ratio = w / float(h) if h > 0 else 0

                        if 1.5 <= aspect_ratio <= 6.0 and w * h >= self.min_plate_area:
                            yolo_plates.append((x1, y1, w, h, box.conf.item()))

            return yolo_plates
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return []

    def detect_with_cascade(self, img):
        """Detect license plates using Haar cascade"""
        if self.plate_cascade is None:
            return []

        try:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.equalizeHist(img_gray)
            img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

            all_detections = []

            cascade_plates = self.plate_cascade.detectMultiScale(
                img_gray,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(60, 20),
                maxSize=(300, 100)
            )
            all_detections.extend(cascade_plates)

            filtered_plates = []
            seen_regions = set()

            for (x, y, w, h) in all_detections:
                region_id = f"{x//10}_{y//10}_{w//10}_{h//10}"

                if region_id in seen_regions:
                    continue

                seen_regions.add(region_id)
                area = w * h
                aspect_ratio = w / float(h)

                if 1.5 <= aspect_ratio <= 6.0 and area > self.min_plate_area:
                    filtered_plates.append((x, y, w, h, 0.5))

            return filtered_plates
        except Exception as e:
            print(f"Error in cascade detection: {e}")
            return []

    def preprocess_image(self, img):
        """Apply minimal preprocessing techniques for detection"""
        try:
            img = self.ensure_image_format(img)
            if img is None:
                return [np.ones((100, 100, 3), dtype=np.uint8) * 255]

            preprocessed_images = []
            preprocessed_images.append(img.copy())

            # Add grayscale variant
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                preprocessed_images.append(gray_3ch)
            except Exception as e:
                print(f"Grayscale conversion error: {e}")

            return preprocessed_images
        except Exception as e:
            print(f"Critical error in preprocessing: {str(e)}")
            if img is not None and img.size > 0:
                return [img.copy()]
            return [np.ones((100, 100, 3), dtype=np.uint8) * 255]

    def process_ocr_results(self, ocr_results):
        """Process OCR results to extract Nepali plate information with confidence scoring.

        Nepali plates aren't built around one fixed embedded word the way Tunisian
        plates always include 'تونس' — they show a Devanagari province/category
        prefix (e.g. बागमती, प्र.दि.का.) OR are plain embossed Roman digits on newer
        vehicles. So unlike the old logic, we don't force any text into the result:
        we report whatever combination of Devanagari text and digits we actually find.
        """
        # Sort results by confidence
        sorted_results = sorted(ocr_results, key=lambda x: x[2], reverse=True)

        # Debug OCR results
        print("OCR Results:", [(text, conf) for _, text, conf in sorted_results])

        # Look for Devanagari province/category text with the best confidence
        devanagari_text = ""
        devanagari_confidence = 0.0
        for _, text, prob in sorted_results:
            match = self.devanagari_pattern.search(text)
            if match:
                devanagari_text = match.group(0)
                devanagari_confidence = prob
                print(f"Found Devanagari text in: {text}")
                break

        # Extract all numbers using regex (handles Latin digits; Devanagari digits
        # are normalized to Latin first)
        devanagari_digit_map = str.maketrans('०१२३४५६७८९', '0123456789')
        all_text = ' '.join([item[1] for item in sorted_results]).translate(devanagari_digit_map)
        numbers = re.findall(r'\d+', all_text)

        # Remove duplicates while preserving order
        unique_numbers = []
        seen = set()
        for num in numbers:
            if num not in seen:
                unique_numbers.append(num)
                seen.add(num)
        numbers = unique_numbers

        # Format the plate text from whatever was actually detected
        if devanagari_text and numbers:
            formatted_text = f"{devanagari_text} {' '.join(numbers)}"
            confidence_score = 0.7
        elif numbers:
            # Common on newer embossed plates with no Devanagari text visible
            formatted_text = ' '.join(numbers)
            confidence_score = 0.5
        elif devanagari_text:
            formatted_text = devanagari_text
            confidence_score = 0.3
        else:
            formatted_text = None
            confidence_score = 0.0

        print(f"Final plate text: '{formatted_text}' with confidence: {confidence_score}")

        return formatted_text, numbers, devanagari_text, confidence_score

    def process_image_array(self, img):
        """Process an image array directly with improved detection and OCR"""
        if img is None:
            return None, 0.0

        try:
            # First ensure valid image format
            img = self.ensure_image_format(img)
            if img is None:
                return None, 0.0

            # Detect plate regions with confidence scores
            plate_regions = self.detect_plate(img)
            if not plate_regions:
                print("No license plates detected")
                return None, 0.0

            best_confidence = 0.0
            best_plate_text = None

            for idx, (x, y, w, h, detection_conf) in enumerate(plate_regions):
                try:
                    # Make sure coordinates are valid
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    if x < 0: x = 0
                    if y < 0: y = 0
                    if w <= 0 or h <= 0 or x+w > img.shape[1] or y+h > img.shape[0]:
                        print(f"Invalid plate region #{idx}: ({x}, {y}, {w}, {h}) for image of size {img.shape}")
                        continue

                    # Extract plate region
                    plate_roi = img[y:y+h, x:x+w].copy()

                    # Apply preprocessing to get multiple variants
                    preprocessed_variants = self.preprocess_image(plate_roi)

                    # Apply OCR
                    ocr_results = []
                    for variant in preprocessed_variants:
                        try:
                            results = self.reader.readtext(variant)
                            ocr_results.extend(results)
                        except Exception as e:
                            print(f"OCR error: {e}")

                    if ocr_results:
                        # Process the text with confidence scoring
                        plate_text, numbers, devanagari_text, text_conf = self.process_ocr_results(ocr_results)
                        if not plate_text:
                            continue

                        # Combine detection and OCR confidence
                        combined_conf = detection_conf * 0.3 + text_conf * 0.7

                        if combined_conf > best_confidence:
                            best_confidence = combined_conf
                            best_plate_text = plate_text
                except Exception as e:
                    print(f"Error processing plate region #{idx}: {e}")
                    continue

            if best_plate_text:
                return best_plate_text, best_confidence

            return None, 0.0
        except Exception as e:
            print(f"Error in image processing: {e}")
            return None, 0.0

# Initialize Flask application
app = Flask(__name__)
detector = NepaliPlateDetector()

@app.route('/detect_plate', methods=['POST'])
def detect_plate():
    try:
        # Check if the request contains an image
        if 'image' not in request.files and 'image' not in request.json:
            return jsonify({'error': 'No image provided'}), 400

        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Handle base64 encoded image
        elif 'image' in request.json:
            base64_img = request.json['image']
            if base64_img.startswith('data:image'):
                base64_img = base64_img.split(',')[1]
            img_bytes = base64.b64decode(base64_img)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Process the image
        plate_text, confidence = detector.process_image_array(img)

        if plate_text is None:
            return jsonify({
                'success': False,
                'plateText': None,
                'confidence': 0,
                'noPlateDetected': True
            })

        print(f"Sending response with plate text: {plate_text}")
        return jsonify({
            'success': True,
            'plateText': plate_text,
            'confidence': float(confidence),
            'noPlateDetected': False
        })

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500

# Add a basic root route
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'Nepali License Plate Detection API',
        'status': 'active',
        'sample': 'बागमती 1234'  # Test Devanagari encoding
    })

if __name__ == '__main__':
    # Run Flask app with UTF-8 support
    app.run(host='0.0.0.0', port=5000, debug=False)
