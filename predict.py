import cv2
import mediapipe as mp
import numpy as np
import time
import os
from model import GestureModel
from data_processor import DataProcessor
from config import GESTURES, CAMERA_ID, WINDOW_NAME, DETECTION_CONFIDENCE, TRACKING_CONFIDENCE
from utils import draw_text

class GestureRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        self.model_handler = GestureModel(num_classes=len(GESTURES))
        self.model_loaded = self.model_handler.load_trained_model()
        
        self.processor = DataProcessor()
        
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()
        
        self.last_predictions = []
        self.max_predictions = 5  # Number of predictions to average
        
        print(f"Model loaded: {'Success' if self.model_loaded else 'Failed'}")
    
    def detect_hands(self, frame):
        """Detect hand landmarks in a frame"""
        if frame is None:
            return None, False, None, None
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(frame_rgb)
        
        hand_detected = False
        hand_img = None
        hand_bbox = None
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                h, w, _ = frame.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                
                for landmark in hand_landmarks.landmark:
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    x_min = min(x_min, x)
                    y_min = min(y_min, y)
                    x_max = max(x_max, x)
                    y_max = max(y_max, y)
                
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                if x_max > x_min and y_max > y_min:
                    hand_img = frame[y_min:y_max, x_min:x_max].copy()
                    hand_bbox = (x_min, y_min, x_max, y_max)
                    hand_detected = True
        
        return frame, hand_detected, hand_img, hand_bbox
    
    def predict_gesture(self, hand_img):
        """Predict gesture from hand image"""
        if hand_img is None or not self.model_loaded:
            return None, 0.0
        
        try:
            preprocessed = self.processor.preprocess_frame(hand_img)
            
            predictions = self.model_handler.predict(preprocessed)[0]
            
            top_prediction_idx = np.argmax(predictions)
            top_prediction_score = predictions[top_prediction_idx]
            
            self.last_predictions.append(top_prediction_idx)
            if len(self.last_predictions) > self.max_predictions:
                self.last_predictions.pop(0)
            
            from collections import Counter
            counter = Counter(self.last_predictions)
            most_common_prediction = counter.most_common(1)[0][0]
            
            predicted_gesture = GESTURES[most_common_prediction]
            confidence = predictions[most_common_prediction]
            
            return predicted_gesture, confidence
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None, 0.0
    
    def calculate_fps(self):
        """Calculate frames per second"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
    
    def run(self):
        """Run real-time hand gesture recognition"""
        if not self.model_loaded:
            print("Error: Model not loaded! Please train the model first.")
            return
        
        cap = cv2.VideoCapture(CAMERA_ID)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera with ID {CAMERA_ID}.")
            print("Try changing the CAMERA_ID in config.py")
            return
        
        print("Starting hand gesture recognition...")
        print("Press 'ESC' to exit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                cap.release()
                cap = cv2.VideoCapture(CAMERA_ID)
                if not cap.isOpened():
                    print("Could not reopen camera. Exiting.")
                    break
                continue
            
            frame = cv2.flip(frame, 1)
            
            frame, hand_detected, hand_img, hand_bbox = self.detect_hands(frame)
            
            prediction_text = "No hand detected"
            if hand_detected and hand_img is not None:
                predicted_gesture, confidence = self.predict_gesture(hand_img)
                
                if predicted_gesture:
                    x_min, y_min, _, _ = hand_bbox
                    prediction_text = f"{predicted_gesture} ({confidence:.2f})"
                    cv2.putText(
                        frame,
                        prediction_text,
                        (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )
            
            self.calculate_fps()
            
            fps_text = f"FPS: {self.fps:.1f}"
            cv2.putText(
                frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            
            cv2.putText(
                frame,
                f"Prediction: {prediction_text}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            cv2.putText(
                frame,
                "Press ESC to exit",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            
            cv2.imshow(WINDOW_NAME, frame)
            
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Hand gesture recognition stopped.")

if __name__ == "__main__":
    recognizer = GestureRecognizer()
    recognizer.run()