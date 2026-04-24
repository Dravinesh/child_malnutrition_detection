from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
import logging
from predict import predict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────
app = FastAPI(
    title="NutriScan API",
    description="Child malnutrition detection using EfficientNetB0",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES  = {"image/jpeg", "image/png", "image/webp", "image/jpg", "image/heic", "image/heif"}
MAX_FILE_SIZE  = 10 * 1024 * 1024  # 10 MB (increased for phone photos)


# ─────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "NutriScan API is running ✅", "version": "2.0.0"}

@app.get("/health")
async def health():
    return {"status": "ok"}


# ─────────────────────────────────────────
# PREDICTION ENDPOINT
# ─────────────────────────────────────────
@app.post("/predict")
async def predict_malnutrition(image: UploadFile = File(...)):

    # 1. Read bytes
    image_bytes = await image.read()

    # 2. Size check
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 10MB.")

    # 3. Content type check (relaxed — accept anything image-like)
    content_type = image.content_type or ""
    if not content_type.startswith("image/") and not content_type == "application/octet-stream":
        # Try to proceed anyway — PIL will reject truly invalid files
        logger.warning(f"Unusual content type: {content_type}, proceeding anyway")

    # 4. Run prediction
    try:
        result = predict(image_bytes)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    logger.info(f"Result: {result}")
    return JSONResponse(content=result)


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
