# 🛡️ FaceGuard AI – Identity & Deepfake Protection

FaceGuard AI is an all-in-one deepfake detection and identity verification tool. It enables users to generate deepfakes using SimSwap, detect real vs fake faces using Xception, and attribute identities using FaceNet.

---

## 🚀 Features

- 📹 Upload or record a video via web UI
- 🔼 Extract frames automatically
- 👤 Detect, crop and save real faces
- 🦢 Generate deepfakes using [SimSwap](https://github.com/neuralchen/SimSwap)
- 🧐 Detect deepfakes using a fine-tuned **Xception** model
- 🧬 Recognize real identities using **FaceNet**
- 🕵️ If fake, detect both source (host) and swapped-in identity
- 💾 Generate and save `.pkl` files with FaceNet embeddings
- 💻 Unified Gradio UI with **two options**:
  - `Generate Dataset`: Extract + crop faces
  - `Run Inference`: Detect fake/real, show identity

---

## 🧠 Models Used

| Model          | Purpose                         | Path                            |
|----------------|----------------------------------|----------------------------------|
| `Xception`     | Real vs. fake classification     | `models/Xception_finetuned.h5`  |
| `FaceNet`      | Identity recognition             | `models/best_real_classifier.pth` |
| `.pkl files`   | Real/fake embeddings database    | `models/real_embeddings.pkl`, `models/fake_embeddings.pkl` |

---

## 🧱 Folder Structure

```
faceguard_ai/
👉🔍 app/
│   ├── ui.py                     # 🔥 Main Gradio interface
│   ├── deepfake_detector.py      # Xception + FaceNet inference logic
│   ├── save_pickle_embeddings.py # Save .pkl identity databases
│   ├── simswap_runner.py         # Run SimSwap
│   ├── frame_extractor.py        # Extract frames
│   └── face_cropper.py           # Crop faces using MTCNN
├── models/                       # Fine-tuned models + identity DBs
│   ├── Xception_finetuned.h5
│   ├── best_real_classifier.pth
│   ├── real_embeddings.pkl
│   └── fake_embeddings.pkl
├── faces/                        # Real and fake face images
│   └── <username>/
│       ├── real/
│       └── fake/<source>/
├── frames/                       # Extracted video frames
├── simswap_new/                  # SimSwap repository
│   ├── checkpoints/
│   ├── arcface_model/
│   └── videos/<user>/
├── run_app.py                    # Launch UI
├── requirements-faceguard.txt
├── requirements-simswap.txt
└── README.md
```

---

## 📦 Setup

### 1. Create environments:

#### 🟩 FaceGuard:
```bash
conda create -n faceguard python=3.9
conda activate faceguard
pip install -r requirements-faceguard.txt
```

#### 🟦 SimSwap:
```bash
conda create -n simswap python=3.9
conda activate simswap
pip install -r requirements-simswap.txt
```

> 💡 FFmpeg must be installed system-wide and available in your PATH.

---

## ▶️ Run the UI

```bash
conda activate faceguard
python run_app.py
```

Then open:
```
http://127.0.0.1:7860
```

---

## 🔍 Real vs. Fake Detection Pipeline

- 📸 Extracted frames are passed to **Xception**
- ✅ If **real**: use **FaceNet** to predict identity
- ❌ If **fake**:
  - Use FaceNet to find:
    - 👤 The original host face
    - 🤵 The swapped-in identity

You must first run:
```bash
python app/save_pickle_embeddings.py
```
to save the FaceNet embedding databases.

---

## 💾 Model Downloads Required

### SimSwap Checkpoints
- [Checkpoints Folder](https://drive.google.com/drive/folders/1jV6_0FIMPC53FZ2HzZNJZGMe55bbu17R)
  - Unzip `checkpoints.zip` into:
    ```bash
    simswap_new/checkpoints/
    ```
  - Place `arcface_checkpoint.tar` into:
    ```bash
    simswap_new/arcface_model/
    ```

### Face Parsing
- [Download model](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view)
- Place in:
  ```bash
  simswap_new/parsing_model/checkpoint/
  ```

### InsightFace (Preprocessing)
- [Download model](https://drive.google.com/file/d/1goH5lO8BAhTpRhpBeXqWEcGkxiiLlgx9/view)
- Unzip into:
  ```bash
  simswap_new/insightface_func/models/
  ```
---

## 🧙 Credits

- SimSwap: [neuralchen](https://github.com/neuralchen/SimSwap)
- Xception: Binary classification model
- FaceNet: Identity recognition (via `facenet-pytorch`)
- MTCNN: Face detector
- Gradio: UI

---

## 🔒 Disclaimer

This project is for academic use only. Use deepfake tools ethically and responsibly.

