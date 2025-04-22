import os
from collections import Counter
from mtcnn import MTCNN
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from app.utils import preprocess_xception, preprocess_facenet

# Load detector once
detector = MTCNN()

def match_embedding(embedding, is_real, real_embs_tensor, fake_embs_tensor, real_labels, fake_origins, fake_swaps, device):
    # Ensure embedding is a NumPy array with correct shape
    if isinstance(embedding, torch.Tensor):
        embedding = embedding.detach().cpu().numpy()
    if len(embedding.shape) == 1:
        embedding = embedding.reshape(1, -1)

    emb = torch.tensor(embedding, dtype=torch.float32).to(device)

    if is_real:
        sims = F.cosine_similarity(emb, real_embs_tensor, dim=1)
        if sims.numel() == 0:
            return "Unknown", 0.0
        idx = torch.argmax(sims).item()
        idx = min(idx, len(real_labels) - 1)  # avoid out-of-bounds
        return real_labels[idx], float(sims[idx])
    else:
        sims = F.cosine_similarity(emb, fake_embs_tensor, dim=1)
        if sims.numel() == 0:
            return "Unknown", "Unknown", 0.0
        idx = torch.argmax(sims).item()
        idx = min(idx, len(fake_origins) - 1)
        return fake_origins[idx], fake_swaps[idx], float(sims[idx])


def detect_deepfake(
    video_path,
    facenet_embedder,
    real_embs_tensor, real_labels,
    fake_embs_tensor, fake_origins, fake_swaps,
    xception_model,
    device,
    video_output_path="inference_Data/processed_output.mp4"
):
    cap = cv2.VideoCapture(video_path)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    os.makedirs("inference_Data", exist_ok=True)
    out = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    fake_flags, source_names, swapped_names, frames = [], [], [], []

    while True:
        ret, frame = cap.read()
        if not ret: break
        faces = detector.detect_faces(frame)
        for face in faces:
            x, y, w, h = face['box']
            x, y = max(0, x), max(0, y)
            cropped = frame[y:y+h, x:x+w]
            if cropped.size == 0: continue

            xcep_input = preprocess_xception(cropped)
            xcep_pred = xception_model.predict(xcep_input, verbose=0)[0][0]
            is_fake = xcep_pred > 0.5
            confidence = round(float(xcep_pred if is_fake else 1 - xcep_pred), 2)

            face_img = Image.fromarray(cropped).convert("RGB")
            face_tensor = preprocess_facenet(face_img).to(device)
            embedding = facenet_embedder(face_tensor)

            if is_fake:
                source, target, sim = match_embedding(embedding, False, real_embs_tensor, fake_embs_tensor, real_labels, fake_origins, fake_swaps, device)
                label = f"Fake {confidence*100:.1f}%\nTarget: {source} | Source: {target}"
                color = (0, 0, 255)
                fake_flags.append(True)
                source_names.append(source)
                swapped_names.append(target)
            else:
                identity, sim = match_embedding(embedding, True, real_embs_tensor, fake_embs_tensor, real_labels, fake_origins, fake_swaps, device)
                label = f"Real {confidence*100:.1f}%\n{identity}"
                color = (0, 255, 0)
                fake_flags.append(False)
                source_names.append(identity)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            for i, line in enumerate(label.split("\n")):
                cv2.putText(frame, line, (x, y - 10 - i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        frames.append(frame)

    cap.release()

    summary_label = ""
    if sum(fake_flags) > sum([not f for f in fake_flags]):
        most_src = Counter(source_names).most_common(1)[0][0]
        most_trg = Counter(swapped_names).most_common(1)[0][0]
        summary_label = f"Deepfake   Target: {most_src} | Source: {most_trg}"
    elif source_names:
        most_src = Counter(source_names).most_common(1)[0][0]
        summary_label = f"Real: {most_src}"

    for frame in frames:
        # Calculate text size
        text_size, _ = cv2.getTextSize(summary_label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)
        text_width = text_size[0]

        # Centered x coordinate
        x_center = int((frame.shape[1] - text_width) / 2)
        y_pos = 40  # Top margin

        cv2.putText(frame, summary_label, (x_center, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 255, 255), 3)
        out.write(frame)

    out.release()
    return video_output_path


def detect_image_deepfake(
    image_path,
    facenet_embedder,
    real_embs_tensor, real_labels,
    fake_embs_tensor, fake_origins, fake_swaps,
    xception_model,
    device
):
    try:
        # Load and preprocess image
        face_img = Image.open(image_path).convert("RGB")
        xcep_input = preprocess_xception(np.array(face_img))
        xcep_pred = xception_model.predict(xcep_input, verbose=0)[0][0]
        is_fake = xcep_pred > 0.5
        confidence = round(float(xcep_pred if is_fake else 1 - xcep_pred), 2)

        face_tensor = preprocess_facenet(face_img).to(device)
        embedding = facenet_embedder(face_tensor)

        if is_fake:
            source, target, sim = match_embedding(
                embedding, False,
                real_embs_tensor, fake_embs_tensor,
                real_labels, fake_origins, fake_swaps,
                device
            )
            return None, f"🟥 Fake ({confidence*100:.1f}%)\nTarget: {source} | Source: {target}"
        else:
            identity, sim = match_embedding(
                embedding, True,
                real_embs_tensor, fake_embs_tensor,
                real_labels, fake_origins, fake_swaps,
                device
            )
            return None, f"🟩 Real ({confidence*100:.1f}%)\nIdentity: {identity}"

    except Exception as e:
        return None, f"❌ Error: {str(e)}"

