import tkinter as tk
from tkinter import messagebox
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
    os.makedirs(data_dir)

details_file = os.path.join(data_dir, 'user_details.pkl')
faces_file = os.path.join(data_dir, 'face_data.pkl')
attendance_dir = 'Attendance'

if not os.path.exists(attendance_dir):
    os.makedirs(attendance_dir)

# Function to register a user
def register_user(first_name, last_name, index_number):
    #check if user has already been registered
    
    if os.path.isfile(details_file):
        with open(details_file, 'rb') as f:
            details = pickle.load(f)
        if any(user['index'] == index_number for user in details):
            messagebox.showerror("Error", "This index number is already registered.")
            return
    def capture_face():
        video = cv2.VideoCapture(0)
        facedetect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        captured_image = None

        while True:
            ret, frame = video.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture image. Please check your webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                crop_img = frame[y:y + h, x:x + w]
                resized_img = cv2.resize(crop_img, (150, 150))
                captured_image = resized_img
                cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 3)
                break

            cv2.imshow("frame", frame)
            k = cv2.waitKey(1)
            if captured_image is not None:
                break

        video.release()
        cv2.destroyAllWindows()

        if captured_image is not None:
            try:
                face_encoding = face_recognition.face_encodings(captured_image)[0]
            except IndexError:
                messagebox.showerror("Error", "No face detected. Please try again.")
                return
            user_details = {'first_name': first_name, 'last_name': last_name, 'index': index_number}
            if not os.path.isfile(details_file):
                details = [user_details]
            else:
                with open(details_file, 'rb') as f:
                    details = pickle.load(f)
                details.append(user_details)
            with open(details_file, 'wb') as f:
                pickle.dump(details, f)

            if not os.path.isfile(faces_file):
                face_data = [face_encoding]
            else:
                with open(faces_file, 'rb') as f:
                    face_data = pickle.load(f)
                face_data.append(face_encoding)
            with open(faces_file, 'wb') as f:
                pickle.dump(face_data, f)

            messagebox.showinfo("Success", "Face data and user details saved successfully.")
        else:
            messagebox.showerror("Error", "No face captured. Please try again.")

    capture_face()

# Function to mark attendance
def mark_attendance():
    def capture_and_mark():
        if not os.path.isfile(details_file) or not os.path.isfile(faces_file):
            messagebox.showerror("Error", "User details or face data files not found. Please register first.")
            return

        with open(details_file, 'rb') as f:
            user_details = pickle.load(f)

        with open(faces_file, 'rb') as f:
            face_data = pickle.load(f)
        face_data = np.array(face_data)

        if len(user_details) != face_data.shape[0]:
            messagebox.showerror("Error", "Mismatch between the number of user details and face data samples.")
            return

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

        tolerance = 0.5  # Tolerance variable
        COL_NAMES = ['FIRST_NAME', 'LAST_NAME', 'INDEX', 'TIME']

        while True:
            ret, frame = video.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture image. Please check your webcam.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            face_recognized = False

            for face_encoding in face_encodings:
                try:
                    match = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
                    if True in match:
                        best_match_index = match.index(True)
                        index_number = known_face_names[best_match_index]
                        ts = time.time()
                        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                        file_path = os.path.join(attendance_dir, f"Attendance_{date}.csv")
                        exist = os.path.isfile(file_path)

                        # Check if the user is already marked present in the CSV file
                        if exist:
                            with open(file_path, 'r', newline='') as csvfile:
                                reader = csv.DictReader(csvfile)
                                if any(row['INDEX'] == index_number for row in reader):
                                    messagebox.showinfo("Info", f"User with index {index_number} has already been marked present.")
                                    video.release()
                                    cv2.destroyAllWindows()
                                    return

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
                            messagebox.showinfo("Success", f"Attendance marked for {user_detail['first_name']} {user_detail['last_name']}")
                            video.release()
                            cv2.destroyAllWindows()
                            return
                except IndexError:
                    messagebox.showerror("Error", "No face detected. Please make sure you are registered in the system.")
                    video.release()
                    cv2.destroyAllWindows()
                    return
                
            if not face_recognized:
                messagebox.showerror("Error", "Face not recognized. Please make sure you are registered in the system.")
                video.release()
                cv2.destroyAllWindows()
                return

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

    capture_and_mark()

# Create the Tkinter GUI
def main_window():
    root = tk.Tk()
    root.title("Class Attendance System")
    root.geometry("600x500")
    root.resizable(False, False)

    title = tk.Label(root, text="CLASS ATTENDANCE SYSTEM", font=("Helvetica", 23, "bold"))
    title.pack(pady=(20,20))

    def open_register():
        register_window = tk.Toplevel(root)
        register_window.title("Register")
        register_window.geometry("400x300")
        register_window.resizable(False, False)

        tk.Label(register_window, text="First Name:", font=("Helvetica", 12)).pack(pady=5)
        first_name_entry = tk.Entry(register_window, font=("Helvetica", 12))
        first_name_entry.pack(pady=5)

        tk.Label(register_window, text="Last Name:", font=("Helvetica", 12)).pack(pady=5)
        last_name_entry = tk.Entry(register_window, font=("Helvetica", 12))
        last_name_entry.pack(pady=5)

        tk.Label(register_window, text="Index Number:", font=("Helvetica", 12)).pack(pady=5)
        index_entry = tk.Entry(register_window, font=("Helvetica", 12))
        index_entry.pack(pady=5)

        def submit_registration():
            first_name = first_name_entry.get()
            last_name = last_name_entry.get()
            index_number = index_entry.get()
            if first_name and last_name and index_number:
                register_user(first_name, last_name, index_number)
                register_window.destroy()
            else:
                messagebox.showerror("Error", "All fields are required.")

        tk.Button(register_window, text="Register", font=("Helvetica", 12, "bold"), command=submit_registration).pack(pady=20)

    def open_attendance():
        mark_attendance()

    tk.Button(root, text="Register Student", font=("Helvetica", 18, ), command=open_register, width=25, height=2).pack(pady=10)
    tk.Button(root, text="Mark Attendance", font=("Helvetica", 18, ), command=open_attendance, width=25, height=2).pack(pady=10)
    tk.Button(root, text="Exit", font=("Helvetica", 18, ), command=root.quit, width=25, height=2).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main_window()
