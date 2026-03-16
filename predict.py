import numpy as np
from PIL import Image
import tensorflow as tf
import io
import os
import zipfile
import gdown

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
GDRIVE_FILE_ID     = "1viicllzL-UBaQR6HyulMTgEyRJNKDWIA"
MODEL_ZIP_PATH     = "model/nutriscan_savedmodel.zip"
MODEL_PATH         = "model/nutriscan_savedmodel"
CLASS_LABELS       = ["healthy", "mild", "moderate", "severe"]
IMG_SIZE           = (224, 224)
CONFIDENCE_THRESHOLD = 0.60  # Reject predictions below 60% confidence


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
        print("Model downloaded and extracted!")
    else:
        print("Model already exists, skipping download.")

download_model()


# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
print(f"Loading model from: {MODEL_PATH}")
model = tf.saved_model.load(MODEL_PATH)
infer = model.signatures["serving_default"]
print("Model loaded successfully!")


# ─────────────────────────────────────────
# IMAGE QUALITY CHECK
# ─────────────────────────────────────────
def check_image_quality(image_bytes: bytes) -> bool:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    if img.width < 100 or img.height < 100:
        return False
    arr = np.array(img)
    if arr.mean() < 20:
        return False
    return True


# ─────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> tf.Tensor:
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    img_array = np.array(img, dtype=np.float32)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return tf.constant(img_array)


# ─────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────
def predict(image_bytes: bytes) -> dict:
    # Step 1 — Check image quality
    if not check_image_quality(image_bytes):
        return {
            "classification": "uncertain",
            "confidence": 0.0,
            "all_scores": {},
            "message": "Image quality too low. Please retake in good lighting."
        }

    # Step 2 — Preprocess
    img_tensor = preprocess_image(image_bytes)

    # Step 3 — Run inference
    output = infer(img_tensor)
    output_key = list(output.keys())[0]
    scores = output[output_key].numpy()[0]

    # Step 4 — Get prediction
    predicted_index = int(np.argmax(scores))
    predicted_label = CLASS_LABELS[predicted_index]
    confidence = float(scores[predicted_index])

    all_scores = {
        CLASS_LABELS[i]: round(float(scores[i]), 4)
        for i in range(len(CLASS_LABELS))
    }

    # Step 5 — Reject low confidence predictions
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "classification": "uncertain",
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
            "message": f"Low confidence ({round(confidence*100, 1)}%). Please use a clearer image."
        }

    return {
        "classification": predicted_label,
        "confidence": round(confidence, 4),
        "all_scores": all_scores,
        "message": "Success"
    }
