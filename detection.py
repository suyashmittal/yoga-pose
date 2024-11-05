import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
import tensorflow as tf
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = np.degrees(np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2))
    
    if angle < 0:
        angle += 360
    
    return angle

def angles_finder(landmarks):
    angles = []
    angle_pairs = [
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP),
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
        (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
        (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
        (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
        (mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
        (mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_ANKLE),
        (mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_ANKLE)
    ]
    
    for pair in angle_pairs:
        angle = calculateAngle(
            (landmarks[pair[0].value].x, landmarks[pair[0].value].y, landmarks[pair[0].value].z),
            (landmarks[pair[1].value].x, landmarks[pair[1].value].y, landmarks[pair[1].value].z),
            (landmarks[pair[2].value].x, landmarks[pair[2].value].y, landmarks[pair[2].value].z)
        )
        angles.append(angle)
    
    return angles

def predict_yoga_pose(angles, model):
    input_data = np.array(angles).reshape(1, -1)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    
    predicted_class_index = np.argmax(prediction)
    predicted_class = le.inverse_transform([predicted_class_index])[0]
    confidence = prediction[0][predicted_class_index]
    
    return predicted_class, confidence

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, model_complexity=2)
mp_drawing = mp.solutions.drawing_utils

# Load the trained model and scaler
loaded_model = tf.keras.models.load_model('yoga_pose_model.keras')
scaler = joblib.load('yoga_pose_scaler.joblib')
le = joblib.load('yoga_pose_label_encoder.joblib')


# Open webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow('Yoga Pose Prediction', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Yoga Pose Prediction', 1080, 720)
# Set the interval for processing frames (e.g., every 1 second)
process_interval = 1 # in seconds
last_process_time = time.time()

last_predicted_pose = ""
last_confidence = 0.0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and get the pose landmarks
    results = pose.process(image_rgb)

    # Draw the pose landmarks on the image
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Check if it's time to process a frame
        current_time = time.time()
        if current_time - last_process_time >= process_interval:
            # Extract angles from landmarks
            angles = angles_finder(results.pose_landmarks.landmark)

            # Predict yoga pose and get confidence
            predicted_pose, confidence = predict_yoga_pose(angles, loaded_model)

            # Update the last predicted pose and confidence
            last_predicted_pose = predicted_pose
            last_confidence = confidence

            last_process_time = current_time

    # Display the last predicted pose and confidence on every frame
    if last_predicted_pose:
        text = f"Pose: {last_predicted_pose} | Confidence: {last_confidence:.2f}"
        cv2.putText(image, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display the image
    cv2.imshow('Yoga Pose Prediction', image)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()