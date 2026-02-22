from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from predict import predict

# ─────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────
app = FastAPI(
    title="NutriScan API",
    description="Child malnutrition detection using EfficientNetB0",
    version="1.0.0",
)

# ─────────────────────────────────────────
# CORS — allows React app to call this API
# ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
MAX_FILE_SIZE  = 5 * 1024 * 1024  # 5 MB


# ─────────────────────────────────────────
# HEALTH CHECK
# Visit http://localhost:8000/ to confirm server is running
# ─────────────────────────────────────────
@app.get("/")
async def root():
    return {"status": "NutriScan API is running ✅"}


# ─────────────────────────────────────────
# PREDICTION ENDPOINT
# POST /predict  →  receives image, returns classification
# ─────────────────────────────────────────
@app.post("/predict")
async def predict_malnutrition(image: UploadFile = File(...)):

    # 1. Validate file type
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{image.content_type}'. Upload a JPEG or PNG image.",
        )

    # 2. Read bytes
    image_bytes = await image.read()

    # 3. Reject empty or oversized files
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum size is 5 MB.")

    # 4. Run model prediction
    try:
        result = predict(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # 5. Return result
    # Example response:
    # {
    #   "classification": "mild",
    #   "confidence": 0.9231,
    #   "all_scores": { "healthy": 0.03, "mild": 0.92, "moderate": 0.04, "severe": 0.01 }
    # }
    return JSONResponse(content=result)


# ─────────────────────────────────────────
# ENTRY POINT
# Run:  python main.py
# ─────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
