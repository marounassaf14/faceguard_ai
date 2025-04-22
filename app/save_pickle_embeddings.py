import os
import glob
import pickle
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
from facenet_pytorch import InceptionResnetV1, MTCNN
import torchvision.transforms as transforms

# -------------------------------
# Paths and device
# -------------------------------
data_root = "faces"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_path = "models/best_real_classifier.pth"

# -------------------------------
# Load FaceNet embedder
# -------------------------------
class FaceNetEmbedder(torch.nn.Module):
    def __init__(self, model_path, device='cpu'):
        super().__init__()
        self.device = device
        self.embedder = InceptionResnetV1(pretrained=None, classify=False).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        filtered_ckpt = {k: v for k, v in checkpoint.items() if not k.startswith('classifier.')}
        self.embedder.load_state_dict(filtered_ckpt)
        self.embedder.eval()

    def forward(self, face_img):
        with torch.no_grad():
            return self.embedder(face_img.to(self.device)).cpu().numpy()

facenet_embedder = FaceNetEmbedder(weights_path, device=device)

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
with open("models/real_embeddings.pkl", "wb") as f:
    pickle.dump((real_embs, real_labels), f)

with open("models/fake_embeddings.pkl", "wb") as f:
    pickle.dump((fake_embs, fake_origins, fake_swaps), f)

print("✅ Embedding databases saved.")
