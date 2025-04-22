import pickle
import torch
import numpy as np
from facenet_pytorch import InceptionResnetV1
from keras.models import load_model

# ------------------------
# DEVICE
# ------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------
# XCEPTION
# ------------------------
xception_model = load_model("models/xception_finetuned.h5", compile=False)

# ------------------------
# FACENET
# ------------------------
class FaceNetEmbedder(torch.nn.Module):
    def __init__(self, model_path, device='cpu'):
        super().__init__()
        self.device = device
        self.embedder = InceptionResnetV1(pretrained=None, classify=False).to(device)
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint.get("state_dict", checkpoint)
        filtered = {}
        for k, v in state_dict.items():
            if k.startswith("embedder."):
                k = k[len("embedder."):]
            if not (k.startswith("classifier") or k.startswith("logits")):
                filtered[k] = v
        self.embedder.load_state_dict(filtered, strict=True)
        self.embedder.eval()

    def forward(self, face_img):
        with torch.no_grad():
            return self.embedder(face_img.to(self.device)).cpu().numpy()

facenet_embedder = FaceNetEmbedder("models/best_real_classifier.pth", device=device)

# ------------------------
# LOAD EMBEDDINGS
# ------------------------
with open("models/real_embeddings.pkl", "rb") as f:
    real_embs, real_labels = pickle.load(f)
real_embs = [e.detach().cpu().numpy() if isinstance(e, torch.Tensor) else e for e in real_embs]
real_embs_tensor = torch.tensor(np.stack(real_embs)).to(device)

with open("models/fake_embeddings.pkl", "rb") as f:
    fake_embs, fake_origins, fake_swaps = pickle.load(f)
fake_embs = [e.detach().cpu().numpy() if isinstance(e, torch.Tensor) else e for e in fake_embs]
fake_embs_tensor = torch.tensor(np.stack(fake_embs)).to(device)

print("Real emb tensor shape:", real_embs_tensor.shape)
print("Fake emb tensor shape:", fake_embs_tensor.shape)
print("Len real_labels:", len(real_labels))
print("Len fake_origins:", len(fake_origins))
