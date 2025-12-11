import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load trained models
emotion_model = load_model("emotion_model.h5")
age_model = load_model("age_prediction_model.h5")

# Define emotions and age categories
emotions = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
age_categories = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-60", "60+"]

# Load Haarcascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)

# Initialize emotion probabilities for the bar graph
emotion_probs = np.zeros(len(emotions))

# Set up Matplotlib figure for real-time bar graph
plt.ion()
fig, ax = plt.subplots()
bars = ax.bar(emotions, emotion_probs, color='skyblue')
ax.set_ylim(0, 1)
ax.set_ylabel("Probability")
ax.set_title("Emotion Level")

# Function to update bar graph dynamically
def update_graph(emotion_probs):
    for bar, new_prob in zip(bars, emotion_probs):
        bar.set_height(new_prob)
    plt.draw()
    plt.pause(0.01)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_gray = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_gray, (48, 48))
        face_resized = face_resized / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)
        face_resized = np.expand_dims(face_resized, axis=-1)

        # Predict emotion
        emotion_pred = emotion_model.predict(face_resized)[0]
        emotion_probs = emotion_pred  # Store probabilities for graph
        predicted_emotion = emotions[np.argmax(emotion_pred)]

        # Predict age
        age_pred = age_model.predict(face_resized)[0]
        predicted_age = age_categories[np.argmax(age_pred)]

        # Draw rectangle and labels
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{predicted_emotion}, {predicted_age}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show camera feed with detected face, emotion, and age
    cv2.imshow("Real-Time Age & Emotion Detection", frame)

    # Update bar graph
    update_graph(emotion_probs)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.show()