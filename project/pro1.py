import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

# Load Haar Cascade Classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load emotion recognition model
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(128, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))

# Compile emotion recognition model
emotion_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Load emotion dictionary
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Function to preprocess and predict emotions
def predict_emotion(face_roi):
    # Resize face ROI to 48x48 and convert to grayscale
    face_roi = cv2.resize(face_roi, (48, 48))
    gray_face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    
    # Reshape face ROI to (1, 48, 48, 1)
    gray_face_roi = gray_face_roi.reshape((1, 48, 48, 1))
    
    # Perform emotion prediction on the detected face
    emotion_prediction = emotion_model.predict(gray_face_roi)
    maxindex = int(np.argmax(emotion_prediction))
    
    return maxindex

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect faces in the frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        # Extract face ROI (Region of Interest)
        face_roi = frame[y:y+h, x:x+w]

        # Predict emotion
        emotion_index = predict_emotion(face_roi)
        emotion_label = emotion_dict[emotion_index]

        # Draw rectangle around face and label with predicted emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the video feed with overlays
    cv2.imshow('Video', cv2.resize(frame, (1200, 860), interpolation=cv2.INTER_CUBIC))

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
