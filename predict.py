
GDRIVE_FILE_ID     = "1UneK9S_Arlb3lvRtOmo-bIbswjJPzbv6"
import numpy as np
from PIL import Image, ImageEnhance
import tensorflow as tf
import io
import os
import zipfile
import gdown
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
GDRIVE_FILE_ID       = "1UneK9S_Arlb3lvRtOmo-bIbswjJPzbv6"  # ← keep your existing ID
MODEL_ZIP_PATH       = "model/nutriscan_savedmodel.zip"
MODEL_PATH           = "model/nutriscan_savedmodel"
CLASS_LABELS         = ["healthy", "mild", "moderate", "severe"]
IMG_SIZE             = (224, 224)
CONFIDENCE_THRESHOLD = 0.55   # slightly relaxed for real-world images


# ─────────────────────────────────────────
# DOWNLOAD & EXTRACT MODEL
# ─────────────────────────────────────────
def download_model():
    if not os.path.exists(MODEL_PATH):
        logger.info("Downloading model from Google Drive...")
        os.makedirs("model", exist_ok=True)
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_ZIP_PATH, quiet=False)
        logger.info("Extracting model...")
        with zipfile.ZipFile(MODEL_ZIP_PATH, "r") as z:
            z.extractall("model/")
        os.remove(MODEL_ZIP_PATH)
        logger.info("Model ready!")
    else:
        logger.info("Model already exists locally.")

download_model()


# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────
logger.info(f"Loading model from: {MODEL_PATH}")
model     = tf.saved_model.load(MODEL_PATH)
infer     = model.signatures["serving_default"]
OUTPUT_KEY = list(infer.structured_outputs.keys())[0]
logger.info(f"Model loaded! Output key: {OUTPUT_KEY}")


# ─────────────────────────────────────────
# IMAGE QUALITY CHECK
# ─────────────────────────────────────────
def check_image_quality(image_bytes: bytes) -> tuple:
    """
    Returns (is_ok: bool, reason: str)
    Rejects images that are too small, too dark, or completely blank.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return False, f"Cannot open image: {e}"

    if img.width < 80 or img.height < 80:
        return False, "Image too small (min 80x80)"

    arr = np.array(img, dtype=np.float32)
    mean_brightness = arr.mean()

    if mean_brightness < 15:
        return False, "Image too dark"

    if mean_brightness > 245:
        return False, "Image too bright / blank"

    return True, "ok"


# ─────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────
def preprocess_image(image_bytes: bytes) -> tf.Tensor:
    """
    Robust preprocessing pipeline:
    1. Open with Pillow
    2. Convert to RGB (handles RGBA, grayscale, CMYK etc.)
    3. Auto-correct orientation (EXIF)
    4. Resize using LANCZOS (high quality)
    5. Normalize using EfficientNet's preprocess_input
       (scales to [-1, 1] range — matches training)
    6. Add batch dimension
    """
    img = Image.open(io.BytesIO(image_bytes))

    # Fix EXIF orientation — phones often send rotated images
    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # Convert to RGB
    img = img.convert("RGB")

    # Resize with high quality
    img = img.resize(IMG_SIZE, Image.LANCZOS)

    # To numpy
    img_array = np.array(img, dtype=np.float32)

    # ✅ EfficientNet preprocessing — MUST match training
    # This scales pixels from [0,255] → [-1, 1]
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    # Add batch dim → (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    return tf.constant(img_array, dtype=tf.float32)


# ─────────────────────────────────────────
# PREDICTION WITH ENSEMBLE
# ─────────────────────────────────────────
def run_inference(img_tensor: tf.Tensor) -> np.ndarray:
    """Run a single inference pass and return raw scores."""
    output = infer(img_tensor)
    scores = output[OUTPUT_KEY].numpy()[0]
    return scores


def predict_with_tta(image_bytes: bytes) -> np.ndarray:
    """
    Test-Time Augmentation (TTA):
    Run inference on 3 versions of the image and average scores.
    This reduces random classification on borderline images.

    Versions:
    1. Original image
    2. Slightly brightened image
    3. Slightly contrasted image
    """
    img = Image.open(io.BytesIO(image_bytes))

    try:
        from PIL import ImageOps
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    img = img.convert("RGB").resize(IMG_SIZE, Image.LANCZOS)

    all_scores = []

    # Version 1 — original
    arr1 = np.array(img, dtype=np.float32)
    arr1 = tf.keras.applications.efficientnet.preprocess_input(arr1.copy())
    t1   = tf.constant(np.expand_dims(arr1, 0), dtype=tf.float32)
    all_scores.append(run_inference(t1))

    # Version 2 — slightly brighter
    try:
        img2 = ImageEnhance.Brightness(img).enhance(1.15)
        arr2 = np.array(img2, dtype=np.float32)
        arr2 = tf.keras.applications.efficientnet.preprocess_input(arr2.copy())
        t2   = tf.constant(np.expand_dims(arr2, 0), dtype=tf.float32)
        all_scores.append(run_inference(t2))
    except Exception:
        pass

    # Version 3 — slightly higher contrast
    try:
        img3 = ImageEnhance.Contrast(img).enhance(1.15)
        arr3 = np.array(img3, dtype=np.float32)
        arr3 = tf.keras.applications.efficientnet.preprocess_input(arr3.copy())
        t3   = tf.constant(np.expand_dims(arr3, 0), dtype=tf.float32)
        all_scores.append(run_inference(t3))
    except Exception:
        pass

    # Average scores across all augmentations
    averaged = np.mean(all_scores, axis=0)
    return averaged


# ─────────────────────────────────────────
# MAIN PREDICT FUNCTION
# ─────────────────────────────────────────
def predict(image_bytes: bytes) -> dict:
    """
    Full prediction pipeline:
    1. Quality check
    2. TTA inference (3 augmented versions averaged)
    3. Confidence threshold check
    4. Return result
    """

    # Step 1 — Quality check
    is_ok, reason = check_image_quality(image_bytes)
    if not is_ok:
        logger.warning(f"Image quality check failed: {reason}")
        return {
            "classification": "uncertain",
            "confidence":     0.0,
            "all_scores":     {},
            "message":        f"Image quality issue: {reason}. Please retake the photo."
        }

    # Step 2 — TTA inference
    try:
        scores = predict_with_tta(image_bytes)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return {
            "classification": "uncertain",
            "confidence":     0.0,
            "all_scores":     {},
            "message":        f"Prediction failed: {str(e)}"
        }

    # Step 3 — Get top prediction
    predicted_index = int(np.argmax(scores))
    predicted_label = CLASS_LABELS[predicted_index]
    confidence      = float(scores[predicted_index])

    all_scores = {
        CLASS_LABELS[i]: round(float(scores[i]), 4)
        for i in range(len(CLASS_LABELS))
    }

    logger.info(f"Prediction: {predicted_label} ({confidence:.2%}) | All: {all_scores}")

    # Step 4 — Confidence threshold
    if confidence < CONFIDENCE_THRESHOLD:
        logger.warning(f"Low confidence: {confidence:.2%}")
        return {
            "classification": "uncertain",
            "confidence":     round(confidence, 4),
            "all_scores":     all_scores,
            "message":        f"Low confidence ({round(confidence*100,1)}%). Use a clearer, well-lit photo."
        }

    return {
        "classification": predicted_label,
        "confidence":     round(confidence, 4),
        "all_scores":     all_scores,
        "message":        "success"
    }
