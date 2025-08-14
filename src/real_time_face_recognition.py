import cv2
import face_recognition
import os
import pickle

# Path to save known face encodings
ENCODINGS_PATH = "encodings/known_faces.pkl"
os.makedirs("encodings", exist_ok=True)

# Load existing encodings if available
known_face_encodings = []
known_face_names = []

if os.path.exists(ENCODINGS_PATH):
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
        known_face_encodings = data["encodings"]
        known_face_names = data["names"]
        print(f"Loaded {len(known_face_names)} known faces.")

# Initialize webcam
video_capture = cv2.VideoCapture(0)

print("Press 'q' to quit, 's' to save a new face.")

while True:
    ret, frame = video_capture.read()
    rgb_frame = frame[:, :, ::-1]  # BGR → RGB

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # Compare to known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_face_names[match_index]

        face_names.append(name)

    # Draw rectangles and names
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow('Face Recognition', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('s') and len(face_encodings) == 1:
        # Save a new face
        name = input("Enter name for this face: ")
        known_face_encodings.append(face_encodings[0])
        known_face_names.append(name)
        with open(ENCODINGS_PATH, "wb") as f:
            pickle.dump({"encodings": known_face_encodings, "names": known_face_names}, f)
        print(f"Saved encoding for {name}")

video_capture.release()
cv2.destroyAllWindows()
