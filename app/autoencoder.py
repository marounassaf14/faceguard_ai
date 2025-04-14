import os
import numpy as np
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm

IMG_SIZE = 128  # size for resizing cropped faces

def load_faces(folder):
    images = []
    for file in sorted(os.listdir(folder)):
        if file.endswith((".jpg", ".png")):
            img = load_img(os.path.join(folder, file), target_size=(IMG_SIZE, IMG_SIZE))
            img = img_to_array(img) / 255.0
            images.append(img)
    return np.array(images)

def build_autoencoder():
    input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

def generate_deepfakes(input_folder, output_folder, epochs=10):
    os.makedirs(output_folder, exist_ok=True)

    # Load data
    print("📥 Loading cropped faces...")
    data = load_faces(input_folder)
    print(f"✅ Loaded {len(data)} face images.")

    # Build and train the model
    model = build_autoencoder()
    print("⚙️ Training autoencoder...")
    model.fit(data, data, epochs=epochs, batch_size=8, shuffle=True, verbose=1)

    # Predict (generate fakes)
    print("🧪 Generating deepfake-style reconstructions...")
    reconstructions = model.predict(data)

    for i, img in enumerate(reconstructions):
        img = (img * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_folder, f"deepfake_{i:04d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    print(f"✅ Deepfake-style images saved to: {output_folder}")
