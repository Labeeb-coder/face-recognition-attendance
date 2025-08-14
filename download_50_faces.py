import os
import urllib.request

# List of 50 people from LFW dataset (2 images each)
people = [
    "George_W_Bush", "Tony_Blair", "Colin_Powell", "Donald_Rumsfeld", "Gerhard_Schroeder",
    "Ariel_Sharon", "Hugo_Chavez", "Jacques_Chirac", "John_Ashcroft", "Junichiro_Koizumi",
    "Laura_Bush", "Lorne_Michaels", "Lyndon_Johnson", "Mahmoud_Abbas", "Megawati_Sukarnoputri",
    "Michael_Bloomberg", "Michael_Jackson", "Nelson_Mandela", "Paul_Bremer", "Pervez_Musharraf",
    "Recep_Tayyip_Erdogan", "Richard_Cheney", "Robert_Mugabe", "Roh_Moo-hyun", "Rudolph_Giuliani",
    "Saddam_Hussein", "Silvio_Berlusconi", "Tiger_Woods", "Tom_Cruise", "Vladimir_Putin",
    "Yasser_Arafat", "Bill_Clinton", "Al_Gore", "Condoleezza_Rice", "Kofi_Annan",
    "Arnold_Schwarzenegger", "Gwyneth_Paltrow", "Julia_Roberts", "Matt_Damon", "Angelina_Jolie",
    "Brad_Pitt", "Catherine_Zeta-Jones", "Charlize_Theron", "Halle_Berry", "Jennifer_Aniston",
    "Johnny_Depp", "Leonardo_DiCaprio", "Nicole_Kidman", "Sandra_Bullock", "Will_Smith"
]

# Create known_faces directory
base_dir = "known_faces"
os.makedirs(base_dir, exist_ok=True)

# Download each image
for person in people:
    person_dir = os.path.join(base_dir, person)
    os.makedirs(person_dir, exist_ok=True)
    for idx in range(1, 3):  # Download first 2 images for each person
        url = f"https://vis-www.cs.umass.edu/lfw/images/{person}/{person}_{idx:04d}.jpg"
        file_path = os.path.join(person_dir, f"image{idx}.jpg")
        try:
            print(f"Downloading {person} - image {idx}...")
            urllib.request.urlretrieve(url, file_path)
        except Exception as e:
            print(f"❌ Failed to download {person} image {idx}: {e}")

print("✅ Download complete! 50 people added to known_faces/")
