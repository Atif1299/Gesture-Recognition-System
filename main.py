import argparse
import os
import sys
from data_collection import DataCollector
from train import train_model
from predict import GestureRecognizer
from utils import create_directory
from config import DATA_DIR, GESTURES, MODEL_PATH

def setup_environment():
    """Setup initial environment for the application"""
    create_directory(DATA_DIR)
    
    create_directory(os.path.dirname(MODEL_PATH))
    
    print("Hand Gesture Recognition System")
    print("==============================")
    print(f"Dataset directory: {DATA_DIR}")
    print(f"Gestures: {', '.join(GESTURES)}")
    print(f"Model path: {MODEL_PATH}")
    print("==============================")

def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description='Hand Gesture Recognition System')
    
    parser.add_argument('command', type=str, choices=['collect', 'train', 'predict'],
                        help='Command to execute: collect data, train model, or predict gestures')
    
    parser.add_argument('--camera', type=int, default=None,
                        help='Camera device ID (overrides config setting)')
    parser.add_argument('--samples', type=int, default=None,
                        help='Number of samples per gesture (overrides config setting)')
    
    args = parser.parse_args()
    
    if args.camera is not None:
        from config import CAMERA_ID
        print(f"Overriding camera ID: {CAMERA_ID} -> {args.camera}")
        import config
        config.CAMERA_ID = args.camera
    
    if args.samples is not None:
        from config import SAMPLES_PER_GESTURE
        print(f"Overriding samples per gesture: {SAMPLES_PER_GESTURE} -> {args.samples}")
        import config
        config.SAMPLES_PER_GESTURE = args.samples
    
    setup_environment()
    
    if args.command == 'collect':
        print("Starting data collection...")
        collector = DataCollector()
        success = collector.collect_data()
        if success:
            print("Data collection completed successfully.")
            print(f"Collected data saved to '{DATA_DIR}' directory.")
            print("You can now train the model with: python main.py train")
    
    elif args.command == 'train':
        if not os.path.exists(DATA_DIR):
            print(f"Error: Dataset directory '{DATA_DIR}' not found.")
            print("Please run 'python main.py collect' first to collect training data.")
            return
        
        missing_gestures = []
        for gesture in GESTURES:
            gesture_dir = os.path.join(DATA_DIR, gesture)
            if not os.path.exists(gesture_dir) or len(os.listdir(gesture_dir)) == 0:
                missing_gestures.append(gesture)
        
        if missing_gestures:
            print(f"Error: Missing data for gestures: {', '.join(missing_gestures)}")
            print("Please run 'python main.py collect' to collect data for all gestures.")
            return
        
        print("Starting model training...")
        history, model = train_model()
        
        if history and model:
            print("Training completed successfully.")
            print(f"Model saved to '{MODEL_PATH}'")
            print("You can now run prediction with: python main.py predict")
        else:
            print("Training failed. Check the error messages above.")
    
    elif args.command == 'predict':
        if not os.path.exists(MODEL_PATH):
            print(f"Error: Model file not found at '{MODEL_PATH}'")
            print("Please run 'python main.py train' first to train the model.")
            return
        
        print("Starting gesture recognition...")
        recognizer = GestureRecognizer()
        recognizer.run()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)