Face Recognition Attendance System

Face Recognition Attendance System is an AI-powered project that automates employee attendance tracking using real-time facial recognition. It allows organizations to record attendance efficiently, accurately, and securely, eliminating the need for manual sign-ins.

Features

✅ Real-Time Face Recognition: Captures employee faces via webcam and marks attendance automatically.

✅ Employee Database Management: Add new employees easily by uploading their photos; the system automatically encodes their faces.

✅ Attendance Logs: Attendance records are stored in a CSV file (attendance.csv) with timestamps.

✅ Expandable: Add more employees without changing the core system.

✅ Lightweight & Fast: Uses Python, OpenCV, and face_recognition library for quick and efficient recognition.

Technologies Used

Python 3.12

OpenCV – for image and video processing

face_recognition – for facial detection and encoding

NumPy & Pandas – for data handling and storing attendance logs

Installation

Clone the repository:

git clone https://github.com/your-username/face-recognition-attendance.git
cd face-recognition-attendance


Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows


Install dependencies:

pip install -r requirements.txt


Set up your dataset:

mkdir -p dataset/EmployeeName
cp /path/to/photos/*.jpg dataset/EmployeeName/

Usage

Encode Employee Faces:

python encode_faces.py --dataset dataset


This will generate encodings/known_faces.pkl for all employees.

Run Real-Time Attendance System:

python real_time_attendance.py


The webcam will open, detect employee faces, and automatically mark attendance.

Attendance is saved in attendance.csv.

Add New Employees:

Add photos to a new folder in dataset/.

Re-run encode_faces.py to update face encodings.

Directory Structure
face-recognition-attendance/
│
├── dataset/                  # Employee images
│   └── EmployeeName/
│       ├── photo1.jpg
│       ├── photo2.jpg
│
├── encodings/                # Encoded face data
│   └── known_faces.pkl
│
├── encode_faces.py           # Script to encode face images
├── real_time_attendance.py   # Script for live attendance using webcam
├── attendance.csv            # Generated attendance logs
├── requirements.txt          # Python dependencies
└── README.md

Contributing

Fork the repository.

Create a new branch: git checkout -b feature-name

Make your changes and commit: git commit -m "Description of feature"

Push to your branch: git push origin feature-name

Open a Pull Request.

License

This project is licensed under the MIT License – see the LICENSE file for details.
