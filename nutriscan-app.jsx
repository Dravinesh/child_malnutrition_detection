import { useState, useRef } from "react";

const dietPlans = {
  healthy: {
    label: "Healthy",
    color: "#00E5A0",
    glow: "0 0 30px rgba(0,229,160,0.4)",
    icon: "✦",
    description: "Child shows normal nutritional status. Maintain current dietary habits.",
    suggestions: [
      { meal: "Breakfast", items: ["Oats with milk & banana", "Boiled egg", "Fresh orange juice"] },
      { meal: "Lunch", items: ["Rice with lentil dal", "Seasonal vegetables", "Yogurt & fruit"] },
      { meal: "Dinner", items: ["Whole wheat roti", "Protein-rich curry", "Warm milk"] },
      { meal: "Snacks", items: ["Nuts & seeds", "Fresh fruit", "Cheese & crackers"] },
    ],
    tips: ["Keep a balanced diet with all food groups", "Encourage physical activity daily", "Regular health checkups every 6 months"],
  },
  mild: {
    label: "Mild Malnutrition",
    color: "#FFD166",
    glow: "0 0 30px rgba(255,209,102,0.4)",
    icon: "◈",
    description: "Mild nutritional deficiency detected. Dietary improvements recommended.",
    suggestions: [
      { meal: "Breakfast", items: ["Fortified porridge with groundnuts", "2 boiled eggs", "Full-fat milk with banana"] },
      { meal: "Lunch", items: ["Rice + legume mix", "Dark leafy greens", "Avocado + full-fat yogurt"] },
      { meal: "Dinner", items: ["Protein-rich stew (meat/fish)", "Sweet potato", "Vitamin-C rich drink"] },
      { meal: "Snacks", items: ["Peanut butter on bread", "Mango or papaya", "High-energy biscuits"] },
    ],
    tips: ["Increase calorie density in every meal", "Add healthy fats (ghee, nuts, avocado)", "Weekly weight monitoring advised"],
  },
  moderate: {
    label: "Moderate Malnutrition",
    color: "#FF8C42",
    glow: "0 0 30px rgba(255,140,66,0.4)",
    icon: "⬡",
    description: "Moderate malnutrition detected. Structured nutrition rehabilitation needed.",
    suggestions: [
      { meal: "Breakfast", items: ["RUTF (Ready-to-Use Therapeutic Food)", "Fortified milk (F-100)", "Vitamin-rich fruits"] },
      { meal: "Lunch", items: ["High-energy peanut paste", "Legume soup with oil", "Mashed fortified cereals"] },
      { meal: "Dinner", items: ["Meat/fish + vegetable stew", "Enriched porridge", "Oral Rehydration Salts"] },
      { meal: "Supplements", items: ["Iron + Folic acid", "Vitamin A drops", "Zinc supplements"] },
    ],
    tips: ["Consult a nutritionist immediately", "5-6 small meals per day", "Monitor for infections & dehydration", "Follow WHO SAM protocol"],
  },
  severe: {
    label: "Severe Malnutrition",
    color: "#FF4D6D",
    glow: "0 0 30px rgba(255,77,109,0.5)",
    icon: "⚠",
    description: "Severe acute malnutrition detected. Immediate medical intervention required.",
    suggestions: [
      { meal: "Phase 1 (Stabilization)", items: ["F-75 therapeutic milk", "Small frequent feeds every 2-3hrs", "Electrolyte solution (ReSoMal)"] },
      { meal: "Phase 2 (Rehabilitation)", items: ["F-100 high-energy milk", "RUTF packs", "Fortified blended food"] },
      { meal: "Micronutrients", items: ["Vitamin A (high dose)", "Folic acid daily", "Zinc + Copper + Multivitamins"] },
      { meal: "Emergency", items: ["Oral Rehydration Therapy", "Glucose solution (hypoglycemia)", "Antibiotics (if infection)"] },
    ],
    tips: ["⚠ URGENT: Seek hospital care immediately", "Do NOT give high-protein foods in phase 1", "Monitor every 30 mins for hypoglycemia", "Follow WHO inpatient SAM guidelines"],
  },
};

// ✅ FIXED: /predict added at the end + ngrok-skip-browser-warning header added
const BACKEND_URL = "https://c9c2-2405-201-e00f-705c-9135-5099-6546-a23c.ngrok-free.app/predict";

export default function NutriScanApp() {
  const [screen, setScreen] = useState("home");
  const [result, setResult] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [cameraActive, setCameraActive] = useState(false);
  const fileRef = useRef();
  const videoRef = useRef();
  const canvasRef = useRef();
  const streamRef = useRef();

  const analyzeImage = async (imageData) => {
    setLoading(true);
    setScreen("scanning");
    setError(null);
    try {
      const res = await fetch(imageData);
      const blob = await res.blob();
      const formData = new FormData();
      formData.append("image", blob, "image.jpg");

      const response = await fetch(BACKEND_URL, {
        method: "POST",
        body: formData,
        headers: {
          // ✅ This bypasses ngrok's browser warning page
          "ngrok-skip-browser-warning": "true",
        },
      });

      if (!response.ok) throw new Error(`Server error: ${response.status}`);
      const data = await response.json();
      setResult({ ...dietPlans[data.classification], confidence: data.confidence, key: data.classification });
    } catch (e) {
      console.error("Backend error:", e.message);
      // Demo fallback if backend is unreachable
      const keys = ["healthy", "mild", "moderate", "severe"];
      const demo = keys[Math.floor(Math.random() * keys.length)];
      setResult({ ...dietPlans[demo], confidence: (0.82 + Math.random() * 0.15).toFixed(2), key: demo, demo: true });
    } finally {
      setLoading(false);
      setScreen("result");
    }
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => { setPreview(ev.target.result); analyzeImage(ev.target.result); };
    reader.readAsDataURL(file);
  };

  const startCamera = async () => {
    setCameraActive(true);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" }, audio: false });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
    } catch {
      setError("Camera access denied. Please upload an image instead.");
      setCameraActive(false);
    }
  };

  const capturePhoto = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    const dataUrl = canvas.toDataURL("image/jpeg");
    setPreview(dataUrl);
    stopCamera();
    analyzeImage(dataUrl);
  };

  const stopCamera = () => {
    if (streamRef.current) streamRef.current.getTracks().forEach((t) => t.stop());
    setCameraActive(false);
  };

  const reset = () => {
    setScreen("home"); setResult(null); setPreview(null); setError(null); stopCamera();
  };

  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Syne:wght@700;800&display=swap');
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #050810; display: flex; justify-content: center; align-items: center; min-height: 100vh; font-family: 'Space Grotesk', sans-serif; }
        .phone { width: 390px; height: 844px; background: #08091A; border-radius: 44px; overflow: hidden; position: relative; box-shadow: 0 0 0 1px rgba(255,255,255,0.08), 0 60px 120px rgba(0,0,0,0.8), inset 0 1px 0 rgba(255,255,255,0.1); }
        .grid-bg { position: absolute; inset: 0; z-index: 0; background-image: linear-gradient(rgba(0,229,160,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,229,160,0.03) 1px, transparent 1px); background-size: 30px 30px; pointer-events: none; }
        .orb { position: absolute; border-radius: 50%; filter: blur(60px); pointer-events: none; }
        .orb1 { width: 200px; height: 200px; background: rgba(0,229,160,0.12); top: -80px; right: -40px; }
        .orb2 { width: 150px; height: 150px; background: rgba(99,102,241,0.1); bottom: 100px; left: -60px; }
        .content { position: relative; z-index: 1; height: 100%; display: flex; flex-direction: column; }
        .status-bar { display: flex; justify-content: space-between; align-items: center; padding: 14px 28px 0; color: rgba(255,255,255,0.7); font-size: 12px; font-weight: 600; }
        .home { flex: 1; display: flex; flex-direction: column; padding: 0 28px 40px; overflow-y: auto; }
        .logo-section { padding: 32px 0 24px; text-align: center; }
        .logo-badge { display: inline-flex; align-items: center; gap: 8px; background: rgba(0,229,160,0.1); border: 1px solid rgba(0,229,160,0.25); border-radius: 100px; padding: 6px 14px; margin-bottom: 20px; color: #00E5A0; font-size: 11px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; }
        .pulse-dot { width: 6px; height: 6px; background: #00E5A0; border-radius: 50%; animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.7); } }
        .app-title { font-family: 'Syne', sans-serif; font-size: 40px; font-weight: 800; line-height: 1; color: #fff; letter-spacing: -1px; }
        .app-title span { color: #00E5A0; }
        .app-subtitle { color: rgba(255,255,255,0.4); font-size: 13px; margin-top: 10px; line-height: 1.5; }
        .scan-visual { margin: 24px auto; position: relative; width: 200px; height: 200px; display: flex; align-items: center; justify-content: center; }
        .scan-ring { position: absolute; border-radius: 50%; border: 1px solid; animation: rotateSlow linear infinite; }
        .ring1 { width: 200px; height: 200px; border-color: rgba(0,229,160,0.2); animation-duration: 12s; }
        .ring2 { width: 160px; height: 160px; border-color: rgba(0,229,160,0.15); animation-duration: 8s; animation-direction: reverse; }
        .ring3 { width: 120px; height: 120px; border-color: rgba(0,229,160,0.3); border-style: dashed; animation-duration: 6s; }
        @keyframes rotateSlow { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .scan-center { width: 80px; height: 80px; background: rgba(0,229,160,0.1); border: 1px solid rgba(0,229,160,0.4); border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 32px; }
        .corner { position: absolute; width: 20px; height: 20px; }
        .corner::before, .corner::after { content:''; position: absolute; background: #00E5A0; }
        .corner::before { width: 2px; height: 14px; }
        .corner::after { width: 14px; height: 2px; }
        .corner.tl { top: 0; left: 0; } .corner.tl::before { top: 0; left: 0; } .corner.tl::after { top: 0; left: 0; }
        .corner.tr { top: 0; right: 0; } .corner.tr::before { top: 0; right: 0; } .corner.tr::after { top: 0; right: 0; }
        .corner.bl { bottom: 0; left: 0; } .corner.bl::before { bottom: 0; left: 0; } .corner.bl::after { bottom: 0; left: 0; }
        .corner.br { bottom: 0; right: 0; } .corner.br::before { bottom: 0; right: 0; } .corner.br::after { bottom: 0; right: 0; }
        .btn-primary { width: 100%; padding: 18px; border-radius: 20px; border: none; cursor: pointer; font-family: 'Space Grotesk', sans-serif; font-size: 15px; font-weight: 600; display: flex; align-items: center; justify-content: center; gap: 10px; transition: all 0.2s; }
        .btn-upload { background: #00E5A0; color: #050810; margin-bottom: 12px; box-shadow: 0 0 30px rgba(0,229,160,0.3); }
        .btn-upload:hover { transform: translateY(-2px); box-shadow: 0 0 40px rgba(0,229,160,0.5); }
        .btn-camera { background: transparent; color: #fff; border: 1px solid rgba(255,255,255,0.15); }
        .btn-camera:hover { border-color: rgba(255,255,255,0.35); background: rgba(255,255,255,0.05); }
        .btn-icon { font-size: 18px; }
        .divider { display: flex; align-items: center; gap: 12px; margin: 16px 0; color: rgba(255,255,255,0.2); font-size: 11px; }
        .divider::before, .divider::after { content:''; flex: 1; height: 1px; background: rgba(255,255,255,0.08); }
        .scanning-screen { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 40px 28px; gap: 24px; }
        .preview-frame { width: 240px; height: 240px; border-radius: 28px; overflow: hidden; position: relative; border: 1px solid rgba(0,229,160,0.3); box-shadow: 0 0 40px rgba(0,229,160,0.15); }
        .preview-frame img { width: 100%; height: 100%; object-fit: cover; }
        .scan-line { position: absolute; width: 100%; height: 2px; background: linear-gradient(90deg, transparent, #00E5A0, transparent); animation: scanMove 2s linear infinite; box-shadow: 0 0 10px rgba(0,229,160,0.8); }
        @keyframes scanMove { 0% { top: 0; } 100% { top: 100%; } }
        .scanning-label { font-family: 'Syne', sans-serif; font-size: 22px; color: #fff; text-align: center; }
        .scanning-sub { color: rgba(255,255,255,0.4); font-size: 13px; text-align: center; }
        .progress-dots { display: flex; gap: 8px; }
        .dot { width: 8px; height: 8px; border-radius: 50%; background: rgba(0,229,160,0.3); animation: dotPulse 1.4s infinite; }
        .dot:nth-child(2) { animation-delay: 0.2s; } .dot:nth-child(3) { animation-delay: 0.4s; }
        @keyframes dotPulse { 0%, 100% { background: rgba(0,229,160,0.3); transform: scale(1); } 50% { background: #00E5A0; transform: scale(1.3); } }
        .result-screen { flex: 1; display: flex; flex-direction: column; overflow-y: auto; }
        .result-header { padding: 20px 28px 0; display: flex; align-items: center; gap: 12px; }
        .back-btn { width: 40px; height: 40px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.1); background: rgba(255,255,255,0.05); color: #fff; cursor: pointer; display: flex; align-items: center; justify-content: center; font-size: 18px; font-family: 'Space Grotesk', sans-serif; }
        .result-title { font-family: 'Syne', sans-serif; font-size: 18px; color: #fff; font-weight: 700; }
        .classification-card { margin: 20px 28px; border-radius: 28px; padding: 24px; display: flex; flex-direction: column; gap: 12px; position: relative; overflow: hidden; border: 1px solid; }
        .class-badge { display: inline-flex; align-items: center; gap: 8px; background: rgba(0,0,0,0.3); border-radius: 100px; padding: 6px 14px; align-self: flex-start; font-size: 11px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }
        .class-label { font-family: 'Syne', sans-serif; font-size: 28px; font-weight: 800; }
        .confidence-row { display: flex; align-items: center; gap: 12px; }
        .conf-bar-bg { flex: 1; height: 6px; background: rgba(0,0,0,0.3); border-radius: 100px; overflow: hidden; }
        .conf-bar { height: 100%; border-radius: 100px; transition: width 1s ease; }
        .conf-label { font-size: 12px; color: rgba(255,255,255,0.6); min-width: 40px; text-align: right; }
        .class-desc { color: rgba(255,255,255,0.65); font-size: 13px; line-height: 1.6; }
        .demo-tag { font-size: 10px; color: rgba(255,255,255,0.35); font-style: italic; background: rgba(255,255,255,0.05); padding: 4px 10px; border-radius: 100px; align-self: flex-start; }
        .preview-thumb { width: 60px; height: 60px; border-radius: 14px; overflow: hidden; border: 2px solid rgba(255,255,255,0.15); position: absolute; top: 20px; right: 20px; }
        .preview-thumb img { width: 100%; height: 100%; object-fit: cover; }
        .diet-section { padding: 0 28px; flex: 1; }
        .section-title { font-family: 'Syne', sans-serif; font-size: 16px; color: rgba(255,255,255,0.9); margin-bottom: 14px; font-weight: 700; }
        .meal-card { background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.07); border-radius: 20px; padding: 16px; margin-bottom: 10px; }
        .meal-label { font-size: 11px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: rgba(255,255,255,0.4); margin-bottom: 10px; }
        .meal-items { display: flex; flex-direction: column; gap: 6px; }
        .meal-item { display: flex; align-items: center; gap: 10px; color: rgba(255,255,255,0.8); font-size: 13px; }
        .meal-dot { width: 4px; height: 4px; border-radius: 50%; flex-shrink: 0; }
        .tips-section { padding: 0 28px 28px; margin-top: 10px; }
        .tip-item { display: flex; align-items: flex-start; gap: 10px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; padding: 12px 14px; margin-bottom: 8px; color: rgba(255,255,255,0.7); font-size: 13px; line-height: 1.5; }
        .tip-icon { font-size: 14px; flex-shrink: 0; margin-top: 1px; }
        .camera-view { flex: 1; display: flex; flex-direction: column; align-items: center; padding: 28px; }
        .video-container { width: 100%; aspect-ratio: 3/4; border-radius: 28px; overflow: hidden; position: relative; border: 1px solid rgba(0,229,160,0.3); background: #000; }
        video { width: 100%; height: 100%; object-fit: cover; }
        .cam-overlay { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; }
        .face-guide { width: 160px; height: 200px; border-radius: 80px; border: 2px solid rgba(0,229,160,0.6); box-shadow: 0 0 0 9999px rgba(0,0,0,0.4); }
        .capture-btn { width: 72px; height: 72px; border-radius: 50%; background: #fff; border: 4px solid rgba(255,255,255,0.3); cursor: pointer; margin-top: 24px; transition: all 0.2s; box-shadow: 0 0 30px rgba(255,255,255,0.2); }
        .capture-btn:active { transform: scale(0.92); }
        .cancel-cam { background: none; border: none; color: rgba(255,255,255,0.5); font-family: 'Space Grotesk', sans-serif; font-size: 14px; cursor: pointer; margin-top: 16px; padding: 8px; }
        canvas { display: none; }
        .error-msg { background: rgba(255,77,109,0.1); border: 1px solid rgba(255,77,109,0.3); border-radius: 16px; padding: 14px; color: #FF4D6D; font-size: 13px; margin-top: 12px; text-align: center; }
      `}</style>

      <div className="phone">
        <div className="grid-bg" />
        <div className="orb orb1" /><div className="orb orb2" />
        <div className="content">
          <div className="status-bar"><span>9:41</span><span>●●●</span></div>

          {screen === "home" && !cameraActive && (
            <div className="home">
              <div className="logo-section">
                <div className="logo-badge"><div className="pulse-dot" />AI Health Scanner</div>
                <div className="app-title">Nutri<span>Scan</span></div>
                <div className="app-subtitle">Instant child malnutrition detection<br />powered by deep learning</div>
              </div>
              <div className="scan-visual">
                <div className="scan-ring ring1" /><div className="scan-ring ring2" /><div className="scan-ring ring3" />
                <div className="corner tl" /><div className="corner tr" /><div className="corner bl" /><div className="corner br" />
                <div className="scan-center">👶</div>
              </div>
              <button className="btn-primary btn-upload" onClick={() => fileRef.current.click()}>
                <span className="btn-icon">📁</span> Upload Image
              </button>
              <input ref={fileRef} type="file" accept="image/*" style={{ display: "none" }} onChange={handleFileUpload} />
              <div className="divider">or</div>
              <button className="btn-primary btn-camera" onClick={startCamera}>
                <span className="btn-icon">📷</span> Open Camera
              </button>
              {error && <div className="error-msg">{error}</div>}
            </div>
          )}

          {cameraActive && (
            <div className="camera-view">
              <div className="video-container">
                <video ref={videoRef} autoPlay playsInline />
                <div className="cam-overlay"><div className="face-guide" /></div>
              </div>
              <canvas ref={canvasRef} />
              <button className="capture-btn" onClick={capturePhoto} />
              <button className="cancel-cam" onClick={() => { stopCamera(); setError(null); }}>Cancel</button>
            </div>
          )}

          {screen === "scanning" && (
            <div className="scanning-screen">
              <div className="preview-frame">
                {preview && <img src={preview} alt="Analyzing" />}
                <div className="scan-line" />
              </div>
              <div>
                <div className="scanning-label">Analyzing Image...</div>
                <div className="scanning-sub" style={{ marginTop: 6 }}>EfficientNetB0 model processing</div>
              </div>
              <div className="progress-dots">
                <div className="dot" /><div className="dot" /><div className="dot" />
              </div>
            </div>
          )}

          {screen === "result" && result && (
            <div className="result-screen">
              <div className="result-header">
                <button className="back-btn" onClick={reset}>←</button>
                <div className="result-title">Scan Results</div>
              </div>
              <div className="classification-card" style={{ borderColor: result.color + "40", background: `linear-gradient(135deg, ${result.color}12 0%, transparent 60%)`, boxShadow: result.glow }}>
                {preview && <div className="preview-thumb"><img src={preview} alt="" /></div>}
                <div className="class-badge" style={{ color: result.color, border: `1px solid ${result.color}40` }}>
                  <span>{result.icon}</span> Detection Result
                </div>
                <div className="class-label" style={{ color: result.color }}>{result.label}</div>
                <div className="confidence-row">
                  <div className="conf-bar-bg">
                    <div className="conf-bar" style={{ width: `${(result.confidence * 100).toFixed(0)}%`, background: result.color }} />
                  </div>
                  <span className="conf-label">{(result.confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="class-desc">{result.description}</div>
                {result.demo && <div className="demo-tag">⚠ Demo mode — backend not reached</div>}
              </div>
              <div className="diet-section">
                <div className="section-title">📋 Recommended Diet Plan</div>
                {result.suggestions.map((s, i) => (
                  <div className="meal-card" key={i}>
                    <div className="meal-label">{s.meal}</div>
                    <div className="meal-items">
                      {s.items.map((item, j) => (
                        <div className="meal-item" key={j}>
                          <div className="meal-dot" style={{ background: result.color }} />{item}
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
              <div className="tips-section">
                <div className="section-title">💡 Clinical Tips</div>
                {result.tips.map((tip, i) => (
                  <div className="tip-item" key={i}><span className="tip-icon">→</span>{tip}</div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
