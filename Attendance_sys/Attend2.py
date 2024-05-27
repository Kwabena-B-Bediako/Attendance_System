import cv2
import numpy as np
import os
import csv
import time
import pickle
from datetime import datetime
import face_recognition

# Ensure the 'data' directory exists
data_dir = 'data'
if not os.path.exists(data_dir):
    print(f"Data directory '{data_dir}' does not exist. Please create it and add the necessary files.")
    exit()

# Ensure necessary data files exist
details_file = os.path.join(data_dir, 'user_details.pkl')
faces_file = os.path.join(data_dir, 'face_data.pkl')

if not os.path.isfile(details_file):
    print(f"File '{details_file}' not found. Please add it to the '{data_dir}' directory.")
    exit()

if not os.path.isfile(faces_file):
    print(f"File '{faces_file}' not found. Please add it to the '{data_dir}' directory.")
    exit()

# Load user details and face data
with open(details_file, 'rb') as f:
    user_details = pickle.load(f)

with open(faces_file, 'rb') as f:
    face_data = pickle.load(f)
face_data = np.array(face_data)

if len(user_details) != face_data.shape[0]:
    print("Mismatch between the number of user details and face data samples.")
    exit()

# Create the Attendance directory if it does not exist
attendance_dir = 'Attendance'
if not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir)

# Encode the known face data
known_face_encodings = []
known_face_names = []

for user, face_encoding in zip(user_details, face_data):
    try:
        known_face_encodings.append(face_encoding)
        known_face_names.append(user['index'])
    except Exception as e:
        print(f"Could not encode face for user {user['index']}: {e}")
        continue

video = cv2.VideoCapture(0)

COL_NAMES = ['FIRST_NAME', 'LAST_NAME', 'INDEX', 'TIME']

tolerance = 0.5

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to capture image. Please check your webcam.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_recognized = False  # Flag to track if a face is recognized

    for face_encoding in face_encodings:
        try:
            match = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
            if True in match:  # Check if any match is found
                best_match_index = match.index(True)
                index_number = known_face_names[best_match_index]
                ts = time.time()
                date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                file_path = os.path.join("Attendance", f"Attendance_{date}.csv")
                exist = os.path.isfile(file_path)

                # Check if the user is already marked present in the CSV file
                if exist:
                    with open(file_path, 'r', newline='') as csvfile:
                        reader = csv.DictReader(csvfile)
                        if any(row['INDEX'] == index_number for row in reader):
                            print(f"User with index {index_number} has already been marked present.")
                            video.release()
                            cv2.destroyAllWindows()
                            exit()

                # Find the user detail by index
                user_detail = next((user for user in user_details if user['index'] == index_number), None)

                if user_detail:
                    attendance = [user_detail['first_name'], user_detail['last_name'], index_number, str(timestamp)]
                    print(attendance)

                    with open(file_path, "a", newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        if not exist:
                            writer.writerow(COL_NAMES)
                        writer.writerow(attendance)
                    video.release()
                    cv2.destroyAllWindows()
                    exit()

            if not face_recognized:
                print("Face not recognized. Please make sure you are registered in the system.")
                video.release()
                cv2.destroyAllWindows()
                exit()
        except IndexError:
            print("No face detected.")
            break


    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Option to quit the loop if necessary
        break

video.release()
cv2.destroyAllWindows()
