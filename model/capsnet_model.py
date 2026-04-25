"""
model/capsnet_model.py
======================
Model Layer — CapsNet Traffic Sign Classifier
Responsibilities:
  - Load a pre-trained CapsNet model from a .h5 file
  - Preprocess input images (BGR→RGB, resize, normalize)
  - Run inference and return (class_index, confidence, label)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GTSRB class label mapping — all 43 classes
# ---------------------------------------------------------------------------
GTSRB_LABELS: dict[int, str] = {
    0:  "Speed limit (20km/h)",
    1:  "Speed limit (30km/h)",
    2:  "Speed limit (50km/h)",
    3:  "Speed limit (60km/h)",
    4:  "Speed limit (70km/h)",
    5:  "Speed limit (80km/h)",
    6:  "End of speed limit (80km/h)",
    7:  "Speed limit (100km/h)",
    8:  "Speed limit (120km/h)",
    9:  "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons",
}

# Default model path relative to project root
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "capsnet_model.h5"


# ---------------------------------------------------------------------------
# Model loader (singleton-style lazy loading)
# ---------------------------------------------------------------------------
_model = None  # cached model instance


def load_model(model_path: Optional[str] = None):
    """
    Load the pre-trained CapsNet model from a .h5 file.

    Parameters
    ----------
    model_path : str, optional
        Explicit path to the .h5 file.  If omitted, the default path
        ``traffic_sign_capsnet.h5`` at the project root is used.

    Returns
    -------
    keras Model
        The loaded model ready for inference.

    Raises
    ------
    FileNotFoundError
        If the model file cannot be located.
    RuntimeError
        If TensorFlow / Keras fails to load the model.
    """
    global _model
    if _model is not None:
        return _model  # already loaded — reuse

    target = Path(model_path) if model_path else _DEFAULT_MODEL_PATH

    if not target.exists():
        raise FileNotFoundError(
            f"CapsNet model not found at '{target}'.\n"
            "Please place your trained model file at that path and retry."
        )

    try:
        # Import TF lazily so the module is importable even without GPU
        import tensorflow as tf  # noqa: PLC0415

        logger.info("Loading CapsNet model from: %s", target)
        _model = tf.keras.models.load_model(str(target), compile=False)
        logger.info("Model loaded successfully.")
        return _model
    except Exception as exc:
        raise RuntimeError(f"Failed to load model: {exc}") from exc


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------
_IMG_SIZE = (32, 32)


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Read and pre-process a traffic sign image for CapsNet inference.

    Pipeline
    --------
    1. Read with OpenCV (BGR)
    2. Convert BGR → RGB
    3. Resize to 32 × 32
    4. Normalise pixel values to [0, 1]
    5. Add batch dimension → shape (1, 32, 32, 3)

    Parameters
    ----------
    image_path : str
        Absolute or relative path to the image file.

    Returns
    -------
    np.ndarray
        Pre-processed image array of shape (1, 32, 32, 3).

    Raises
    ------
    FileNotFoundError
        If the image file does not exist.
    ValueError
        If OpenCV cannot decode the file as an image.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: '{image_path}'")

    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise ValueError(
            f"OpenCV could not decode '{image_path}'. "
            "Ensure it is a valid image (PNG/JPG/BMP etc.)."
        )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, _IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_normalised = img_resized.astype(np.float32) / 255.0
    img_batch = np.expand_dims(img_normalised, axis=0)  # (1, 32, 32, 3)

    logger.debug("Image pre-processed: %s → shape %s", image_path, img_batch.shape)
    return img_batch


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_image(image_path: str, model_path: Optional[str] = None) -> Tuple[int, float, str]:
    """
    End-to-end prediction for a single traffic sign image.

    Parameters
    ----------
    image_path : str
        Path to the input image.
    model_path : str, optional
        Path to the CapsNet .h5 file.  Uses project-root default if omitted.

    Returns
    -------
    (class_index, confidence, label) : Tuple[int, float, str]
        class_index  — integer class index (0–42)
        confidence   — probability score for the predicted class (0.0–1.0)
        label        — human-readable GTSRB class name

    Raises
    ------
    FileNotFoundError, ValueError, RuntimeError
        Propagated from ``load_model`` or ``preprocess_image``.
    """
    model = load_model(model_path)
    img_batch = preprocess_image(image_path)

    logger.info("Running inference on: %s", image_path)
    raw_output = model.predict(img_batch, verbose=0)

    # CapsNet output can be:
    #   a) A single softmax/sigmoid array  → shape (1, 43)
    #   b) A list/tuple of outputs where the first is the class capsule norms
    if isinstance(raw_output, (list, tuple)):
        logits = raw_output[0]  # first output = digit capsule lengths
    else:
        logits = raw_output

    # Flatten to 1-D probability vector
    probs = np.squeeze(logits)          # (43,)
    if probs.ndim == 0:
        probs = np.array([probs])

    class_index = int(np.argmax(probs))
    confidence = float(probs[class_index])
    label = GTSRB_LABELS.get(class_index, f"Unknown class ({class_index})")

    logger.info("Predicted class %d ('%s') with confidence %.4f", class_index, label, confidence)
    return class_index, confidence, label


# ---------------------------------------------------------------------------
# Convenience helper
# ---------------------------------------------------------------------------

def index_to_label(class_index: int) -> str:
    """Return the human-readable label for a given class index."""
    return GTSRB_LABELS.get(class_index, f"Unknown class ({class_index})")
