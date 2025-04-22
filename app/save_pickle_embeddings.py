import os
import glob
import pickle
import torch
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
import torchvision.transforms as transforms

# -------------------------------
# Paths and device
# -------------------------------
data_root = "../faces"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_path = "../models/best_real_classifier.pth"

# -------------------------------
# Load FaceNet embedder
# -------------------------------
class FaceNetIdentity(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.embedder = InceptionResnetV1(pretrained='vggface2', classify=False)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        emb = self.embedder(x)
        return self.classifier(emb)

model = FaceNetIdentity(num_classes=9)
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()
facenet_embedder = model.embedder  # use only the embedder part

# -------------------------------
# Face preprocessing (160x160)
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def preprocess_pil(pil_img):
    return transform(pil_img).unsqueeze(0)

# -------------------------------
# Face detector
# -------------------------------
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)

# -------------------------------
# Embedding Lists
# -------------------------------
real_embs, real_labels = [], []
fake_embs, fake_origins, fake_swaps = [], [], []

print("Building real embeddings...")
for name in sorted(os.listdir(data_root)):
    real_dir = os.path.join(data_root, name, "real")
    if not os.path.isdir(real_dir):
        continue
    for img_path in tqdm(glob.glob(os.path.join(real_dir, "*"))):
        try:
            img = Image.open(img_path).convert("RGB")
            face = mtcnn(img)
            if face is None:
                continue
            emb = facenet_embedder(face.unsqueeze(0))[0]
            real_embs.append(emb)
            real_labels.append(name)
        except:
            continue

print("Building fake embeddings...")
for name in sorted(os.listdir(data_root)):
    fake_base = os.path.join(data_root, name, "fake")
    if not os.path.isdir(fake_base):
        continue
    for orig_name in os.listdir(fake_base):
        fake_dir = os.path.join(fake_base, orig_name)
        if not os.path.isdir(fake_dir):
            continue
        for img_path in tqdm(glob.glob(os.path.join(fake_dir, "*"))):
            try:
                img = Image.open(img_path).convert("RGB")
                face = mtcnn(img)
                if face is None:
                    continue
                emb = facenet_embedder(face.unsqueeze(0))[0]
                fake_embs.append(emb)
                fake_origins.append(orig_name)
                fake_swaps.append(name)
            except:
                continue

# -------------------------------
# Save to pickle files
# -------------------------------
os.makedirs("models", exist_ok=True)
with open("../models/real_embeddings.pkl", "wb") as f:
    pickle.dump((real_embs, real_labels), f)

with open("../models/fake_embeddings.pkl", "wb") as f:
    pickle.dump((fake_embs, fake_origins, fake_swaps), f)

print("✅ Embedding databases saved.")
