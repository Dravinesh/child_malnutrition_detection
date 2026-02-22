import gdown, zipfile, os

GDRIVE_FILE_ID = "1UneK9S_Arlb3lvRtOmo-bIbswjJPzbv6"
MODEL_ZIP_PATH = "model/nutriscan_savedmodel.zip"
MODEL_PATH = "model/nutriscan_savedmodel"

def download_model():
    if not os.path.exists(MODEL_PATH):
        os.makedirs("model", exist_ok=True)
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_ZIP_PATH, quiet=False)
        print("Extracting model...")
        with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as z:
            z.extractall("model/")
        os.remove(MODEL_ZIP_PATH)
        print("✅ Model ready!")

download_model()

model = tf.keras.models.load_model(MODEL_PATH)

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
