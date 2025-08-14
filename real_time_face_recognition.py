import os
import cv2
import pickle
import face_recognition

# Paths
dataset_path = "/home/muhammed/scikit_learn_data/lfw_subset"
encodings_file = "encodings/known_faces.pkl"

os.makedirs("encodings", exist_ok=True)

# STEP 1: Encode faces from dataset
known_encodings = []
known_names = []

print("[INFO] Encoding faces from dataset...")
for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)
        image = face_recognition.load_image_file(image_path)
        boxes = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, boxes)
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

# Save encodings
data = {"encodings": known_encodings, "names": known_names}
with open(encodings_file, "wb") as f:
    pickle.dump(data, f)
print(f"[INFO] Saved {len(known_names)} encodings to {encodings_file}")

# STEP 2: Real-time face recognition
print("[INFO] Starting webcam...")
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb_frame)
    encodings = face_recognition.face_encodings(rgb_frame, boxes)

    for encoding, box in zip(encodings, boxes):
        matches = face_recognition.compare_faces(known_encodings, encoding)
        name = "Unknown"

        if True in matches:
            matched_idxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            for i in matched_idxs:
                counts[known_names[i]] = counts.get(known_names[i], 0) + 1
            name = max(counts, key=counts.get)

        top, right, bottom, left = box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
