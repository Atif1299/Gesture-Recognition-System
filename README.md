# Hand Gesture Recognition System

A complete system for recognizing hand gestures from video streams or camera feeds using deep learning techniques.

## Overview

This project implements a hand gesture recognition system that can:
1. Collect training data for various hand gestures
2. Train a Convolutional Neural Network (CNN) model
3. Recognize hand gestures in real-time from a camera feed

## Features

- Data collection interface for creating custom gesture datasets
- Deep learning model based on CNN architecture
- Real-time hand detection and landmark tracking using MediaPipe
- Gesture prediction with confidence scores
- Performance monitoring (FPS)
- Modular and extensible codebase

## Requirements

- Python 3.7+
- OpenCV
- TensorFlow 2.x
- MediaPipe
- NumPy
- Matplotlib
- Scikit-learn

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/hand-gesture-recognition.git
cd hand-gesture-recognition
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

The system has three main modes of operation:

### 1. Data Collection

Collect training data for hand gestures:

```
python main.py collect
```

Follow the on-screen instructions to collect data for each gesture. The system will guide you through the process and save the collected images to the `dataset` directory.

### 2. Model Training

Train the hand gesture recognition model:

```
python main.py train
```

This will load the collected data, train a CNN model, and save it to the `models` directory. Training metrics and visualizations will be generated automatically.

### 3. Real-time Prediction

Run real-time hand gesture recognition:

```
python main.py predict
```

This will open a camera feed and display the detected hand gestures in real-time.

## Customization

You can customize the system by modifying the `config.py` file:

- Change the list of gestures to recognize
- Adjust camera settings
- Modify model parameters
- Customize display settings

## Project Structure

- `main.py` - Main entry point for the application
- `config.py` - Configuration settings
- `data_collection.py` - Script to collect training data
- `data_processor.py` - Data preprocessing utilities
- `model.py` - CNN model architecture
- `train.py` - Model training script
- `predict.py` - Real-time prediction from camera
- `utils.py` - Utility functions

## Supported Gestures

By default, the system recognizes the following gestures:
- Thumbs up
- Thumbs down
- Peace sign
- Fist
- Open palm

## Troubleshooting

### Common Issues:

1. **Camera not working**: Make sure your camera is properly connected and not in use by another application.

2. **Model training fails**: Ensure you have collected enough data for each gesture.

3. **Poor recognition accuracy**: Try collecting more diverse data samples or adjust the model architecture.

4. **Performance issues**: If the system runs slowly, try reducing the camera resolution in `config.py`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.