import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import json

IMG_SIZE     = (224, 224)
HEALTHY_DIR  = "data/thermal/healthy"
ABNORMAL_DIR = "data/thermal/abnormal"
OUTPUT_DIR   = "data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_images(folder, label):
    images, labels = [], []
    files = [f for f in os.listdir(folder)
             if f.lower().endswith(('.jpg','.jpeg','.png'))]
    print(f"Loading {len(files)} images from {folder}...")
    for fname in files:
        try:
            img = Image.open(os.path.join(folder, fname)).convert('RGB')
            img = img.resize(IMG_SIZE)
            images.append(np.array(img) / 255.0)
            labels.append(label)
        except Exception as e:
            print(f"Skipped {fname}: {e}")
    return images, labels

print("=" * 50)
print("PREPROCESSING WITH OVERSAMPLING")
print("=" * 50)

healthy_imgs,  healthy_labels  = load_images(HEALTHY_DIR,  0)
abnormal_imgs, abnormal_labels = load_images(ABNORMAL_DIR, 1)

print(f"\nBefore balancing:")
print(f"  Healthy  : {len(healthy_imgs)}")
print(f"  Abnormal : {len(abnormal_imgs)}")

# Oversample abnormal to match healthy
abnormal_arr = np.array(abnormal_imgs)
n_needed = len(healthy_imgs) - len(abnormal_imgs)
indices  = np.random.choice(len(abnormal_arr), n_needed, replace=True)
extra    = abnormal_arr[indices]

abnormal_balanced = list(abnormal_arr) + list(extra)
abnormal_labels_balanced = [1] * len(abnormal_balanced)

print(f"\nAfter balancing:")
print(f"  Healthy  : {len(healthy_imgs)}")
print(f"  Abnormal : {len(abnormal_balanced)}")

X = np.array(healthy_imgs + abnormal_balanced, dtype=np.float32)
y = np.array(healthy_labels + abnormal_labels_balanced)

print(f"\nTotal: {len(X)} images")

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"\nData Split:")
print(f"  Train : {len(X_train)}")
print(f"  Val   : {len(X_val)}")
print(f"  Test  : {len(X_test)}")

np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
np.save(f"{OUTPUT_DIR}/X_val.npy",   X_val)
np.save(f"{OUTPUT_DIR}/X_test.npy",  X_test)
np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
np.save(f"{OUTPUT_DIR}/y_val.npy",   y_val)
np.save(f"{OUTPUT_DIR}/y_test.npy",  y_test)

info = {
    "total":    int(len(X)),
    "healthy":  int(sum(y==0)),
    "abnormal": int(sum(y==1)),
    "train":    int(len(X_train)),
    "val":      int(len(X_val)),
    "test":     int(len(X_test)),
    "img_size": list(IMG_SIZE)
}
with open(f"{OUTPUT_DIR}/dataset_info.json", "w") as f:
    json.dump(info, f, indent=2)

print(f"\n Balanced dataset saved to {OUTPUT_DIR}/")