# encode_faces.py
import os
import pickle
import face_recognition

# Absolute path to your dataset
DATASET_DIR = "/home/muhammed/face-recognition-ai/dataset"
ENCODINGS_FILE = "/home/muhammed/face-recognition-ai/encodings/known_faces.pkl"

# Dictionary to store encodings
known_faces = {"names": [], "encodings": []}

# Loop through each employee folder
for employee_name in os.listdir(DATASET_DIR):
    employee_folder = os.path.join(DATASET_DIR, employee_name)
    if not os.path.isdir(employee_folder):
        continue

    # Loop through each image in the employee folder
    for image_name in os.listdir(employee_folder):
        image_path = os.path.join(employee_folder, image_name)
        image = face_recognition.load_image_file(image_path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_faces["names"].append(employee_name)
            known_faces["encodings"].append(encodings[0])

# Create encodings folder if it doesn't exist
os.makedirs(os.path.dirname(ENCODINGS_FILE), exist_ok=True)

# Save encodings
with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(known_faces, f)

print(f"✅ Saved {len(known_faces['names'])} encodings to {ENCODINGS_FILE}")

