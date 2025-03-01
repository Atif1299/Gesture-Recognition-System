import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from config import DATA_DIR, GESTURES, IMG_SIZE

class DataProcessor:
    def __init__(self):
        self.data_dir = DATA_DIR
        self.gestures = GESTURES
        self.img_size = IMG_SIZE
    
    def load_data(self):
        """Load and preprocess image data for training"""
        X = []  
        y = []  
        
        print("Loading dataset...")
        
        for label_idx, gesture in enumerate(self.gestures):
            gesture_dir = os.path.join(self.data_dir, gesture)
            
            if not os.path.exists(gesture_dir):
                print(f"Warning: Directory for gesture '{gesture}' not found.")
                continue
                
            print(f"Processing gesture: {gesture}")
            
            image_files = [f for f in os.listdir(gesture_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if not image_files:
                print(f"Warning: No images found for gesture '{gesture}'.")
                continue
                
            print(f"Found {len(image_files)} images for gesture '{gesture}'.")
            
            for img_file in image_files:
                img_path = os.path.join(gesture_dir, img_file)
                
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                    
                img = cv2.resize(img, self.img_size)
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                img = img / 255.0
                
                X.append(img)
                y.append(label_idx)
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)
        
        if len(X) == 0:
            raise ValueError("No valid images found in the dataset!")
        
        print(f"Dataset loaded: {len(X)} images with shape {X.shape[1:]} and {len(np.unique(y))} classes.")
        
        y_one_hot = to_categorical(y, num_classes=len(self.gestures))
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_one_hot, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Validation set: {len(X_val)} samples")
        
        return X_train, X_val, y_train, y_val
    
    def preprocess_frame(self, frame):
        """Preprocess a single frame for prediction"""
        img = cv2.resize(frame, self.img_size)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = img / 255.0
        
        img = np.expand_dims(img, axis=0)
        
        return img