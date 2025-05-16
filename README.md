# Yoga Pose Correction with Deep Learning

This project is an AI-powered system that helps yoga practitioners improve their posture in real time. It uses MediaPipe for pose estimation and a deep learning model to classify yoga poses and provide corrective feedback.

## 🚀 Key Features

- **Real-Time Pose Detection:** Using MediaPipe for efficient human pose estimation.
- **Pose Classification:** Deep learning model classifies 8 yoga poses with high accuracy.
- **Feedback System:** Provides actionable feedback on body alignment to improve posture.
- **Pose Confidence:** Only shows results with >65% confidence for reliable guidance.

## 🧘‍♂️ Supported Yoga Poses

- ArdhaChandrasana  
- BaddhaKonasana  
- Downward Dog Pose  
- Natarajasana  
- Triangle Pose  
- UtkataKonasana  
- Veerabhadrasana  
- Vrukshasana  

## 📊 Model Details

- Trained on images labeled with body part angles (shoulders, knees, hips, elbows).
- Achieved **97.44% accuracy** on the validation dataset.
- Uses a neural network multi-class classifier for pose recognition.

## 🛠️ Tech Stack

- Python, TensorFlow, OpenCV, MediaPipe

## 📁 How It Works

1. Extract body angles from images using MediaPipe.
2. Feed these features to the trained neural network.
3. Predict the yoga pose with confidence score.
4. Display real-time feedback to the user for corrections.
