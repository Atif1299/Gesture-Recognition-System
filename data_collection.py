import os
import cv2
import mediapipe as mp
import numpy as np
import time
from config import DATA_DIR, GESTURES, SAMPLES_PER_GESTURE, IMG_SIZE, CAMERA_ID
from utils import create_directory, draw_text

class DataCollector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        create_directory(DATA_DIR)
        for gesture in GESTURES:
            create_directory(os.path.join(DATA_DIR, gesture))
    
    def process_frame(self, frame):
        """Process frame and detect hand landmarks"""
        if frame is None:
            return None, None
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.hands.process(frame_rgb)
        
        hand_detected = False
        hand_img = None
        
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
                
                # Add padding
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(w, x_max + padding)
                y_max = min(h, y_max + padding)
                
                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                if x_max > x_min and y_max > y_min:
                    # Crop hand region
                    hand_img = frame[y_min:y_max, x_min:x_max].copy()
                    hand_detected = True
        
        return frame, hand_img
    
    def save_hand_image(self, hand_img, gesture, sample_count):
        """Save hand image to dataset directory"""
        if hand_img is None:
            return False
            
        hand_img = cv2.resize(hand_img, IMG_SIZE)
        
        save_path = os.path.join(DATA_DIR, gesture, f"{gesture}_{sample_count:04d}.jpg")
        
        try:
            cv2.imwrite(save_path, hand_img)
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False
    
    def collect_data(self):
        """Collect hand gesture images for training"""
        print("Starting data collection...")
        print(f"Gestures to collect: {', '.join(GESTURES)}")
        print(f"Samples per gesture: {SAMPLES_PER_GESTURE}")
        print("Press 'ESC' to exit at any time.")
        
        cap = cv2.VideoCapture(CAMERA_ID)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera with ID {CAMERA_ID}.")
            print("Try changing the CAMERA_ID in config.py")
            return False
        
        for gesture in GESTURES:
            gesture_dir = os.path.join(DATA_DIR, gesture)
            existing_samples = len([f for f in os.listdir(gesture_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            
            if existing_samples >= SAMPLES_PER_GESTURE:
                print(f"Skipping gesture '{gesture}' - already have {existing_samples} samples.")
                continue
            
            print(f"\nPreparing to collect data for gesture: {gesture}")
            print("Position your hand in the camera view and press 'C' to start collection.")
            
            collecting = False
            sample_count = existing_samples
            start_time = time.time()
            
            while sample_count < SAMPLES_PER_GESTURE:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to capture frame.")
                    cap.release()
                    cap = cv2.VideoCapture(CAMERA_ID)
                    if not cap.isOpened():
                        print("Could not reopen camera. Exiting.")
                        return False
                    continue
                
                frame = cv2.flip(frame, 1)
                
                frame, hand_img = self.process_frame(frame)
                
                if collecting:
                    text = f"Collecting {gesture}: {sample_count}/{SAMPLES_PER_GESTURE}"
                    color = (0, 255, 0)  # Green
                else:
                    text = f"Ready for {gesture} - Press 'C' to start"
                    color = (0, 0, 255)  # Red
                
                cv2.putText(
                    frame,
                    text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )
                
                cv2.putText(
                    frame,
                    "Press 'ESC' to exit",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
                cv2.imshow("Hand Gesture Data Collection", frame)
                
                key = cv2.waitKey(1)
                if key == 27:  # ESC key
                    print("\nData collection stopped by user.")
                    break
                elif key == ord('c') or key == ord('C'):
                    collecting = True
                    print(f"Starting collection for {gesture}...")
                
                if collecting and hand_img is not None:
                    current_time = time.time()
                    if current_time - start_time > 0.2:  # 200ms delay
                        if self.save_hand_image(hand_img, gesture, sample_count):
                            sample_count += 1
                            start_time = current_time
                            
                            if sample_count % 10 == 0:
                                print(f"Collected {sample_count}/{SAMPLES_PER_GESTURE} samples for {gesture}")
                
                if sample_count >= SAMPLES_PER_GESTURE:
                    print(f"Completed collection for {gesture}: {sample_count} samples.")
                    break
        
        cap.release()
        cv2.destroyAllWindows()
        
        all_collected = True
        for gesture in GESTURES:
            gesture_dir = os.path.join(DATA_DIR, gesture)
            sample_count = len([f for f in os.listdir(gesture_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            
            if sample_count < SAMPLES_PER_GESTURE:
                print(f"Warning: Only {sample_count}/{SAMPLES_PER_GESTURE} samples collected for {gesture}.")
                all_collected = False
        
        if all_collected:
            print("\nData collection completed successfully for all gestures!")
        else:
            print("\nData collection completed with some gestures having fewer than required samples.")
            print("You can run the collection again to add more samples.")
        
        return True

if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_data()