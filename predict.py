import numpy as np
from PIL import Image
import tensorflow as tf
import io
import os
import zipfile
import gdown

# ─────────────────────────────────────────
# ⚠️ PASTE YOUR GOOGLE DRIVE FILE ID HERE
# Get it from your shareable link:
# https://drive.google.com/file/d/  THIS_PART  /view?usp=sharing
# ─────────────────────────────────────────
GDRIVE_FILE_ID = "1UneK9S_Arlb3lvRtOmo-bIbswjJPzbv6"

MODEL_ZIP_PATH = "model/nutriscan_savedmodel.zip"
MODEL_PATH     = "model/nutriscan_savedmodel"
CLASS_LABELS   = ["healthy", "mild", "moderate", "severe"]
IMG_SIZE       = (224, 224)


# ─────────────────────────────────────────
# DOWNLOAD & EXTRACT MODEL
# ─────────────────────────────────────────
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Model not found. Downloading from Google Drive...")
        os.makedirs("model", exist_ok=True)
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_ZIP_PATH, quiet=False)
        print("Extracting model...")
        with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as z:
            z.extractall("model/")
        os.remove(MODEL_ZIP_PATH)
        print("✅ Model downloaded and extracted!")
    else:
        print("✅ Model already exists, skipping download.")

download_model()


# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
print(f"Loading model from: {MODEL_PATH}")
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]
print("✅ Model loaded successfully!")


# ─────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> tf.Tensor:
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)
    return tf.constant(img_array)


# ─────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────
def predict(image_bytes: bytes) -> dict:
    img_tensor = preprocess_image(image_bytes)

    # Run inference using SavedModel signature
    output = infer(img_tensor)

    # Get the output tensor (key may vary — we take the first output)
    output_key = list(output.keys())[0]
    scores = output[output_key].numpy()[0]  # shape: (4,)

    predicted_index = int(np.argmax(scores))
    predicted_label = CLASS_LABELS[predicted_index]
    confidence      = float(scores[predicted_index])

    all_scores = {
        CLASS_LABELS[i]: round(float(scores[i]), 4)
        for i in range(len(CLASS_LABELS))
    }

    return {
        "classification": predicted_label,
        "confidence":     round(confidence, 4),
        "all_scores":     all_scores,
    }
