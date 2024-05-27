import cv2
import numpy as np
import os
import pickle
import face_recognition

# Initialize webcam and face detector
video = cv2.VideoCapture(0)  # 0 for primary webcam, 1 for secondary
facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

first_name = input("Enter Your First Name: ")
last_name = input("Enter Your Last Name: ")
index_number = input("Enter Your Index Number: ")

data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

captured_image = None

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture image. Please check your webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w]
        resized_img = cv2.resize(crop_img, (150, 150))  # Resize to a larger size for better accuracy
        captured_image = resized_img
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 3)
        break  # Capture only the first detected face

    cv2.imshow("frame", frame)
    k = cv2.waitKey(1)
    if captured_image is not None:
        break

video.release()
cv2.destroyAllWindows()

if captured_image is not None:
    face_encoding = face_recognition.face_encodings(captured_image)[0]  # Generate face encoding

    # Save user details
    user_details = {'first_name': first_name, 'last_name': last_name, 'index': index_number}
    details_file = os.path.join(data_dir, 'user_details.pkl')
    if not os.path.isfile(details_file):
        details = [user_details]
    else:
        with open(details_file, 'rb') as f:
            details = pickle.load(f)
        details.append(user_details)
    with open(details_file, 'wb') as f:
        pickle.dump(details, f)

    # Save face data
    faces_file = os.path.join(data_dir, 'face_data.pkl')
    if not os.path.isfile(faces_file):
        face_data = [face_encoding]
    else:
        with open(faces_file, 'rb') as f:
            face_data = pickle.load(f)
        face_data.append(face_encoding)
    with open(faces_file, 'wb') as f:
        pickle.dump(face_data, f)

    print("Face data and user details saved successfully.")
else:
    print("No face captured. Please try again.")
