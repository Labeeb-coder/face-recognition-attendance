# setup_lfw_faces.py
import os
from sklearn.datasets import fetch_lfw_people
from pathlib import Path
from PIL import Image

# Configuration
output_dir = Path("known_faces")
min_images = 10  # Only includes people with at least this many images
num_people = 50  # Target number of people to include

print("Downloading LFW dataset... (this may take several minutes)")
lfw = fetch_lfw_people(min_faces_per_person=min_images, resize=0.4, color=True)

print(f"Dataset contains {len(lfw.target_names)} people with ≥ {min_images} face(s).")

output_dir.mkdir(parents=True, exist_ok=True)

added = 0
for person_idx, person_name in enumerate(lfw.target_names):
    if added >= num_people:
        break
    person_label = person_name.replace(" ", "_")
    person_dir = output_dir / person_label
    person_dir.mkdir(exist_ok=True)
    
    # Find all images indices belonging to this person
    indices = [i for i, t in enumerate(lfw.target) if t == person_idx]
    if not indices:
        continue

    # Save one image per person
    img = (lfw.images[indices[0]] * 255).astype("uint8")
    img_pil = Image.fromarray(img)
    img_pil.save(person_dir / "image1.jpg")
    
    added += 1
    print(f"Saved sample for: {person_label}")

print(f"Downloaded {added} people into '{output_dir}'. Now running encode...")

# Optionally auto-run encoding script
import subprocess
subprocess.run(["python", "src/encode_faces.py"])
