import tensorflow as tf
import numpy as np
import os
import time
from data_processor import DataProcessor
from model import GestureModel
from config import GESTURES, BATCH_SIZE, EPOCHS, MODEL_PATH
from utils import plot_training_history, plot_confusion_matrix, print_classification_report, create_directory

def train_model():
    print("Starting model training process...")
    start_time = time.time()
    
    # Load and preprocess data
    processor = DataProcessor()
    X_train, X_val, y_train, y_val = processor.load_data()
    
    # Build model
    model_handler = GestureModel(num_classes=len(GESTURES))
    model = model_handler.build_model()
    
    # Create directory for model if it doesn't exist
    create_directory(os.path.dirname(MODEL_PATH))
    
    # Get callbacks
    callbacks = model_handler.get_callbacks()
    
    # Train the model
    print(f"\nTraining model with {len(X_train)} samples...")
    try:
        history = model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the final model
        model_handler.save_model()
        
        # Plot training history
        plot_training_history(history)
        
        # Evaluate the model
        print("\nEvaluating model on validation data...")
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation Loss: {loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        # Generate predictions for confusion matrix
        y_pred = model.predict(X_val, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)
        
        # Plot confusion matrix
        plot_confusion_matrix(y_true_classes, y_pred_classes, GESTURES)
        
        # Print classification report
        print_classification_report(y_true_classes, y_pred_classes, GESTURES)
        
        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.2f} seconds!")
        print(f"Model saved to: {MODEL_PATH}")
        
        return history, model
        
    except Exception as e:
        print(f"Error during training: {e}")
        return None, None

if __name__ == "__main__":
    # Enable memory growth for GPU to prevent OOM errors
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"Found {len(physical_devices)} GPU(s)")
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("Memory growth enabled for GPU")
        except Exception as e:
            print(f"Error configuring GPU: {e}")
    
    train_model()