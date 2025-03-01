import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def create_directory(directory_path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")

def plot_training_history(history):
    """Plot training and validation accuracy/loss"""
    try:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history plot saved as 'training_history.png'")
        plt.close()
    except Exception as e:
        print(f"Error plotting training history: {e}")

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    try:
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            xticklabels=classes, 
            yticklabels=classes
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix saved as 'confusion_matrix.png'")
        plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

def print_classification_report(y_true, y_pred, classes):
    """Print classification report"""
    try:
        report = classification_report(y_true, y_pred, target_names=classes)
        print("\nClassification Report:")
        print(report)
        
        # Save to file
        with open('classification_report.txt', 'w') as f:
            f.write(report)
        print("Classification report saved as 'classification_report.txt'")
    except Exception as e:
        print(f"Error generating classification report: {e}")

def draw_text(frame, text, position=None, color=None, font_scale=None, thickness=None):
    """Draw text on frame with default values from config if not provided"""
    from config import TEXT_COLOR, TEXT_POSITION, FONT_SCALE, FONT_THICKNESS
    
    if frame is None:
        return None
        
    position = position or TEXT_POSITION
    color = color or TEXT_COLOR
    font_scale = font_scale or FONT_SCALE
    thickness = thickness or FONT_THICKNESS
    
    cv2.putText(
        frame, 
        text, 
        position, 
        cv2.FONT_HERSHEY_SIMPLEX, 
        font_scale, 
        color, 
        thickness, 
        cv2.LINE_AA
    )
    
    return frame

def preprocess_for_visualization(img):
    """Prepare an image for visualization (denormalize if needed)"""
    if img is None:
        return None
        
    if np.max(img) <= 1.0:
        img = (img * 255).astype(np.uint8)
    
    return img