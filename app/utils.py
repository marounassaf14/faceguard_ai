import torch
from facenet_pytorch import InceptionResnetV1
import torchvision.transforms as transforms
import tensorflow as tf

# -------------------------------
# Load FaceNet model (PyTorch)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
facenet_model = InceptionResnetV1(pretrained='vggface2', classify=False).to(device)
facenet_model.eval()

# -------------------------------
# XCEPTION PREPROCESSING
# -------------------------------
def preprocess_xception(image_np):
    """
    Prepares a face image for Xception input (299x299 with preprocessing).
    """
    img = tf.convert_to_tensor(image_np, dtype=tf.uint8)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.xception.preprocess_input(img)
    return tf.expand_dims(img, axis=0)  # [1, 299, 299, 3]

# -------------------------------
# FACENET PREPROCESSING
# -------------------------------
def preprocess_facenet(img_pil):
    transform = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])
    return transform(img_pil).unsqueeze(0).to(device)  # [1, 3, 160, 160]