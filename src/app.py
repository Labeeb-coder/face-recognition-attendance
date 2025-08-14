import cv2
import face_recognition
import pickle
from pathlib import Path
import numpy as np

ENCODINGS_PATH = Path(__file__).parent.parent / "encodings" / "known_faces.pkl"

def load_known():
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["labels"]

def main():
    known_encodings, known_labels = load_known()
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Could not open webcam")
        return

    print("Starting webcam. Press 'q' to quit.")
    process_every_n_frames = 2
    frame_count = 0

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = small_frame[:, :, ::-1]

        if frame_count % process_every_n_frames == 0:
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                if len(face_distances) > 0:
                    best_idx = np.argmin(face_distances)
                    if matches[best_idx]:
                        name = known_labels[best_idx]
                face_names.append(name)

        frame_count += 1

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 2; right *= 2; bottom *= 2; left *= 2
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
