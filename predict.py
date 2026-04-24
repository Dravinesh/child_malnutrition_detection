import io
import os

import gdown
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "model/best_malnutrition_model_finetuned.keras"
GDRIVE_FILE_ID = "18S_5_9yEjMJsj5HPnRMyCb-hlyGGerrM"

CLASS_LABELS = ["healthy", "mild", "moderate", "severe"]
IMG_SIZE = (224, 224)
NORMALIZE_MODE = "efficientnet"


def ensure_model_exists():
    os.makedirs("model", exist_ok=True)

    if os.path.exists(MODEL_PATH):
        print(f"Model already exists: {MODEL_PATH}")
        return

    if GDRIVE_FILE_ID == "YOUR_GOOGLE_DRIVE_FILE_ID":
        raise ValueError("Set GDRIVE_FILE_ID in predict.py before deploying.")

    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    print(f"Downloading model from Google Drive: {url}")
    gdown.download(url, MODEL_PATH, quiet=False)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model download failed: {MODEL_PATH}")


ensure_model_exists()

print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")
print(f"Input shape  : {model.input_shape}")
print(f"Output shape : {model.output_shape}")
print(f"Classes      : {CLASS_LABELS}")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)

    if NORMALIZE_MODE == "divide":
        img_array = img_array / 255.0
    elif NORMALIZE_MODE == "efficientnet":
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict(image_bytes: bytes) -> dict:
    img_array = preprocess_image(image_bytes)

    predictions = model.predict(img_array, verbose=0)
    scores = predictions[0]

    predicted_index = int(np.argmax(scores))
    predicted_label = CLASS_LABELS[predicted_index]
    confidence = float(scores[predicted_index])

    all_scores = {
        CLASS_LABELS[i]: round(float(scores[i]), 4)
        for i in range(len(CLASS_LABELS))
    }

    return {
        "classification": predicted_label,
        "confidence": round(confidence, 4),
        "all_scores": all_scores,
    }
