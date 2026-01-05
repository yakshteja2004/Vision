import os

path = r"C:\Users\YAKSH TEJA\OneDrive\Desktop\Vision\vision_backend\eye_dataset\train"
print("Path exists:", os.path.exists(path))
print("Contents:", os.listdir(path) if os.path.exists(path) else "Not found")
