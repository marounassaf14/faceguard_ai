import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mtcnn import MTCNN
from keras.models import load_model
from collections import Counter
from app.utils import preprocess_xception, preprocess_facenet
from facenet_pytorch import InceptionResnetV1
from PIL import Image

# Paths
xception_model_path = "models/xception_finetuned.h5"
facenet_weights_path = "models/best_real_classifier.pth"
video_output_path = "inference_Data/processed_output.mp4"

# -----------------------------
# Load models
# -----------------------------

# Load Xception (Keras) model
xception_model = load_model(xception_model_path)

# Load MTCNN detector
detector = MTCNN()

# Load PyTorch FaceNet embedder
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
            emb = self.embedder(face_img.to(self.device))
            return emb.cpu().numpy()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
facenet_embedder = FaceNetEmbedder(facenet_weights_path, device=device)

# -----------------------------
# Load Real/Fake Embedding Databases
# -----------------------------

# You should precompute and save these in practice.
# Here we assume they're in memory for simplicity.

import pickle
with open("models/real_embeddings.pkl", "rb") as f:
    real_embs, real_labels = pickle.load(f)

with open("models/fake_embeddings.pkl", "rb") as f:
    fake_embs, fake_origins, fake_swaps = pickle.load(f)

# Convert to tensors for cosine similarity
real_embs_tensor = torch.tensor(np.stack(real_embs)).to(device)
fake_embs_tensor = torch.tensor(np.stack(fake_embs)).to(device)

# -----------------------------
# Matching function
# -----------------------------

def match_embedding(embedding, is_real=True):
    emb = torch.tensor(embedding).unsqueeze(0).to(device)
    if is_real:
        sims = F.cosine_similarity(emb, real_embs_tensor, dim=1)
        idx = torch.argmax(sims).item()
        return real_labels[idx], float(sims[idx])
    else:
        sims = F.cosine_similarity(emb, fake_embs_tensor, dim=1)
        idx = torch.argmax(sims).item()
        return (fake_origins[idx], fake_swaps[idx], float(sims[idx]))

# -----------------------------
# Deepfake detection function
# -----------------------------

def detect_deepfake(video_path):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

    os.makedirs("inference_Data", exist_ok=True)
    out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    fake_flags = []
    source_names = []
    swapped_names = []
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detector.detect_faces(frame)
        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            cropped = frame[y:y+h, x:x+w]

            if cropped.size == 0:
                continue

            # -------- Xception: real/fake --------
            xcep_input = preprocess_xception(cropped)
            xcep_pred = xception_model.predict(xcep_input, verbose=0)[0][0]
            is_fake = xcep_pred > 0.5
            confidence = round(float(xcep_pred if is_fake else 1 - xcep_pred), 2)

            # -------- FaceNet: identity --------
            face_img = Image.fromarray(cropped).convert("RGB")
            face_tensor = preprocess_facenet(face_img).to(device)
            emb = facenet_embedder(face_tensor)[0]

            if is_fake:
                source, target, sim = match_embedding(emb, is_real=False)
                label = f"Fake {confidence*100:.1f}%\n{source} → {target}"
                color = (0, 0, 255)
                fake_flags.append(True)
                source_names.append(source)
                swapped_names.append(target)
            else:
                identity, sim = match_embedding(emb, is_real=True)
                label = f"Real {confidence*100:.1f}%\n{identity}"
                color = (0, 255, 0)
                fake_flags.append(False)
                source_names.append(identity)

            # Draw
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            for i, line in enumerate(label.split("\n")):
                cv2.putText(frame, line, (x, y - 10 - i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frames.append(frame)

    cap.release()

    # Summary
    if sum(fake_flags) > sum([not f for f in fake_flags]):
        most_src = Counter(source_names).most_common(1)[0][0]
        most_trg = Counter(swapped_names).most_common(1)[0][0]
        summary_label = f"🔴 Deepfake: {most_src} → {most_trg}"
    else:
        most_src = Counter(source_names).most_common(1)[0][0]
        summary_label = f"🟢 Real: {most_src}"

    for frame in frames:
        cv2.putText(frame, summary_label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 255, 255), 3)
        out.write(frame)

    out.release()
    return video_output_path
