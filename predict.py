import numpy as np
from PIL import Image
import tensorflow as tf
import io
import os
import gdown

# ─────────────────────────────────────────
# ⚠️ CONFIGURE THESE
# ─────────────────────────────────────────

# Paste your Google Drive file ID here
# Get it from your shareable link:
# https://drive.google.com/file/d/  THIS_PART  /view?usp=sharing
GDRIVE_FILE_ID = "1viicllzL-UBaQR6HyulMTgEyRJNKDWIA"

# Local path where model will be saved after download
MODEL_PATH = "model/best_malnutrition_model_finetuned.keras"

# Class labels — must match your training order
CLASS_LABELS = ["healthy", "mild", "moderate", "severe"]

# Input image size
IMG_SIZE = (224, 224)

# Normalization mode:
# "divide"       → image / 255.0  (if you normalized manually)
# "efficientnet" → EfficientNet's built-in preprocess_input
NORMALIZE_MODE = "divide"

# ─────────────────────────────────────────
# DOWNLOAD MODEL FROM GOOGLE DRIVE
# Only downloads if not already present
# ─────────────────────────────────────────
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found locally. Downloading from Google Drive...")
        os.makedirs("model", exist_ok=True)
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
        print("✅ Model downloaded successfully!")
    else:
        print("✅ Model already exists locally, skipping download.")

download_model()

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")
print(f"   Input shape  : {model.input_shape}")
print(f"   Output shape : {model.output_shape}")
print(f"   Classes      : {CLASS_LABELS}")


# ─────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────
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


# ─────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────
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
