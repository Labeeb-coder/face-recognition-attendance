# real_time_attendance.py
import cv2
import pickle
import face_recognition
import os
from datetime import datetime

# Paths
ENCODINGS_FILE = "/home/muhammed/face-recognition-ai/encodings/known_faces.pkl"
ATTENDANCE_FILE = "/home/muhammed/face-recognition-ai/attendance.csv"

# Load known faces
if not os.path.exists(ENCODINGS_FILE):
    print("Error: known_faces.pkl not found. Run encode_faces.py first.")
    exit()

with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)

known_face_encodings = data["encodings"]
known_face_names = data["names"]

# Initialize attendance dict
attendance = {}

# Create/open CSV file
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w") as f:
        f.write("Name,Date,Time\n")

# Start webcam
video_capture = cv2.VideoCapture(0)

print("[INFO] Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Find all faces and encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = face_distances.argmin() if len(face_distances) > 0 else None
        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]

            # Mark attendance
            if name not in attendance:
                now = datetime.now()
                with open(ATTENDANCE_FILE, "a") as f:
                    f.write(f"{name},{now.date()},{now.strftime('%H:%M:%S')}\n")
                attendance[name] = True
                print(f"[ATTENDANCE] {name} marked at {now.strftime('%H:%M:%S')}")

        # Draw rectangle and name
        top, right, bottom, left = [v * 4 for v in face_location]  # Scale back
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
print("[INFO] Webcam stopped. Attendance saved to", ATTENDANCE_FILE)

