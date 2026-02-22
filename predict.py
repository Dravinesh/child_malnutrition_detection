import numpy as np
from PIL import Image
import tensorflow as tf
import io

# ─────────────────────────────────────────
# ⚠️  CONFIGURE THESE TO MATCH YOUR MODEL
# ─────────────────────────────────────────

# 1. Path to your saved .h5 model file
MODEL_PATH = "model/best_malnutrition_model_finetuned.keras"

# 2. Class labels — ORDER MUST MATCH YOUR TRAINING
#    Check how you encoded labels during training (e.g. LabelEncoder, to_categorical)
#    Index 0 = first class, Index 1 = second class, etc.
CLASS_LABELS = ["healthy", "mild", "moderate", "severe"]

# 3. Input image size — EfficientNetB0 default is 224x224
#    Change only if you trained with a different size
IMG_SIZE = (224, 224)

# 4. Normalization mode
#    "divide"   → divides pixels by 255.0  (use if you normalized manually)
#    "efficientnet" → uses EfficientNet's built-in preprocess_input
#                     (use if you used tf.keras.applications.efficientnet.preprocess_input during training)
NORMALIZE_MODE = "divide"

# ─────────────────────────────────────────
# LOAD MODEL — runs once when server starts
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
    """
    Convert raw image bytes → preprocessed numpy array for the model.

    Steps:
      1. Open with Pillow (handles JPEG, PNG, RGBA, grayscale, etc.)
      2. Convert to RGB (ensures 3 channels always)
      3. Resize to IMG_SIZE
      4. Normalize pixel values
      5. Add batch dimension → shape: (1, H, W, 3)
    """
    img = Image.open(io.BytesIO(image_bytes))
    img = img.convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)

    if NORMALIZE_MODE == "divide":
        img_array = img_array / 255.0
    elif NORMALIZE_MODE == "efficientnet":
        img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)
    return img_array


# ─────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────
def predict(image_bytes: bytes) -> dict:
    """
    Run inference on image bytes.

    Returns:
      {
        "classification": "mild",          # predicted class name
        "confidence": 0.9231,              # probability of predicted class
        "all_scores": {                    # probabilities for all classes
          "healthy":  0.03,
          "mild":     0.92,
          "moderate": 0.04,
          "severe":   0.01
        }
      }
    """
    img_array = preprocess_image(image_bytes)

    # Run model — returns shape (1, num_classes)
    predictions = model.predict(img_array, verbose=0)
    scores = predictions[0]  # shape: (num_classes,)

    # Get top prediction
    predicted_index = int(np.argmax(scores))
    predicted_label = CLASS_LABELS[predicted_index]
    confidence      = float(scores[predicted_index])

    # All class scores as a readable dict
    all_scores = {
        CLASS_LABELS[i]: round(float(scores[i]), 4)
        for i in range(len(CLASS_LABELS))
    }

    return {
        "classification": predicted_label,
        "confidence":     round(confidence, 4),
        "all_scores":     all_scores,
    }
