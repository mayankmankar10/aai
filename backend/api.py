"""
backend/api.py
==============
Flask REST API — serves the CapsNet Traffic Sign Agent to the React frontend.
"""

import sys
import os
import tempfile
from pathlib import Path

# Allow imports from project root
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from agent.agent_pipeline import agent_pipeline
from model.capsnet_model import GTSRB_LABELS

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}


def _allowed(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/api/health", methods=["GET"])
def health():
    """Health-check endpoint."""
    return jsonify({"status": "ok", "service": "CapsNet Traffic Sign Agent"})


@app.route("/api/labels", methods=["GET"])
def labels():
    """Return all 43 GTSRB class labels."""
    return jsonify({"labels": GTSRB_LABELS})


@app.route("/api/classify", methods=["POST"])
def classify():
    """
    Accept an image upload, run the agent pipeline, return structured JSON.

    Expects multipart/form-data with field name 'image'.
    Optional field 'model_path' to override the default .h5 location.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not _allowed(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    # Save to temp file for OpenCV processing
    suffix = Path(secure_filename(file.filename)).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp)
        tmp_path = tmp.name

    try:
        model_path = request.form.get("model_path", "").strip() or None
        response = agent_pipeline(tmp_path, model_path=model_path)

        payload = {
            "success": response.success,
            "class_index": response.class_index,
            "label": response.label,
            "confidence": response.confidence,
            "confidence_assessment": response.confidence_assessment,
            "category": response.category,
            "explanation": response.explanation,
            "safety_tip": response.safety_tip,
            "reasoning_trace": response.reasoning_trace,
            "error_message": response.error_message,
        }
        return jsonify(payload)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  CapsNet Traffic Sign Agent — Flask API")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True)
