import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
from config import IMG_SIZE, MODEL_PATH, LEARNING_RATE

class GestureModel:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.input_shape = (*IMG_SIZE, 3)  
        self.model = None
    
    def build_model(self):
        """Build CNN model for hand gesture recognition"""
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.input_shape),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print(model.summary())
        return model
    
    def get_callbacks(self):
        """Define callbacks for training"""
        callbacks = [
            ModelCheckpoint(
                MODEL_PATH,
                save_best_only=True,
                monitor='val_accuracy',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                verbose=1,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        return callbacks
    
    def load_trained_model(self, model_path=None):
        """Load a pre-trained model"""
        path = model_path or MODEL_PATH
        try:
            if not os.path.exists(path):
                print(f"Model file not found at {path}")
                return False
                
            self.model = load_model(path)
            print(f"Model loaded successfully from {path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def save_model(self, model_path=None):
        """Save the trained model"""
        if self.model is None:
            print("No model to save!")
            return False
            
        path = model_path or MODEL_PATH
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            self.model.save(path)
            print(f"Model saved successfully to {path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def predict(self, img):
        """Make a prediction for a single preprocessed image"""
        if self.model is None:
            print("No model loaded!")
            return None
            
        predictions = self.model.predict(img, verbose=0)
        return predictions