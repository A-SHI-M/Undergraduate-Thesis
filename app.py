import io
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from Anomaly_Detection.pipeline.prediction_pipeline import PredictionPipeline
from Anomaly_Detection.constant import IMG_SIZE

app = Flask(__name__)

# Cache loaded pipelines so each model is only loaded once per server session
_pipelines: dict = {}

# Default threshold — adjust per model after evaluating on a validation set
_THRESHOLDS = {
    "fc_ae":    0.01,
    "cnn_ae":   0.01,
    "vae":      0.01,
    "beta_vae": 0.01,
    "cvae":     0.01,
    "vqvae":    0.01,
    "bigan":    0.05,
}


def _get_pipeline(model_type: str) -> PredictionPipeline:
    if model_type not in _pipelines:
        pipeline = PredictionPipeline(
            model_type=model_type,
            threshold=_THRESHOLDS.get(model_type, 0.01),
        )
        pipeline.load()
        _pipelines[model_type] = pipeline
    return _pipelines[model_type]


def _preprocess(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert("L").resize(IMG_SIZE)
    return np.array(img, dtype=np.float32) / 255.0


# ── UI route ──────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# ── API routes ────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    model_type = request.form.get("model_type", "bigan")
    if model_type not in _THRESHOLDS:
        return jsonify({"error": f"Unknown model_type '{model_type}'"}), 400

    try:
        image = _preprocess(request.files["image"].read())
    except Exception as e:
        return jsonify({"error": f"Could not decode image: {e}"}), 400

    try:
        result = _get_pipeline(model_type).predict_single(image)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    return jsonify(result)


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "No images field in request"}), 400

    model_type = request.form.get("model_type", "bigan")
    images = []
    for f in files:
        try:
            images.append(_preprocess(f.read()))
        except Exception:
            continue

    if not images:
        return jsonify({"error": "No valid images could be decoded"}), 400

    try:
        scores, preds = _get_pipeline(model_type).predict(np.stack(images))
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    return jsonify([
        {"anomaly_score": float(s), "is_anomaly": bool(p)}
        for s, p in zip(scores, preds)
    ])


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
