import os
import cv2
import mediapipe as mp

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils 

image_path = "TRAIN/ArdhaChandrasana/Images/4.png"
sample_img = cv2.imread(image_path)

# Perform pose detection after converting the image into RGB format.
results = pose.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))

# Create a copy of the sample image to draw landmarks on.
img_copy = sample_img.copy()

# Check if any landmarks are found.
if results.pose_landmarks:
    # Draw Pose landmarks on the sample image.
    mp_drawing.draw_landmarks(image=img_copy, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)
    output_image_path = os.path.join("./", "test")
    # cv2.imwrite(output_image_path, img_copy)
    # Display the output image with the landmarks drawn, also convert BGR to RGB for display. 
    cv2.imshow('Output', img_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
            

# Release mediapipe pose instance
pose.close()
