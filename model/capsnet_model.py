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

# Default model path — same directory as this file (model/capsnet_model.keras)
_DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "capsnet_model.keras"


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
        Explicit path to the .h5 or .keras file.  If omitted, the default
        ``model/capsnet_model.keras`` (same folder as this script) is used.

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
            "Place your trained model file (capsnet_model.keras or .h5) at that path and retry."
        )

    try:
        # Import TF lazily so the module is importable even without GPU
        import tensorflow as tf  # noqa: PLC0415
        import keras
        from keras import layers

        @keras.saving.register_keras_serializable()
        def squash(vectors, axis=-1):
            """Squashing activation — keeps direction, limits magnitude to (0,1)."""
            s_sq   = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
            scale  = s_sq / (1.0 + s_sq) / tf.sqrt(s_sq + tf.keras.backend.epsilon())
            return scale * vectors

        @keras.saving.register_keras_serializable()
        class CapsuleLayer(layers.Layer):
            """Digit Capsule layer with dynamic routing."""
            def __init__(self, num_capsules, dim_capsules, routing_iters=3, **kw):
                super().__init__(**kw)
                self.num_capsules  = num_capsules
                self.dim_capsules  = dim_capsules
                self.routing_iters = routing_iters

            def build(self, input_shape):
                self.in_caps = input_shape[1]
                self.in_dim  = input_shape[2]
                self.W = self.add_weight(
                    name        = "routing_weights",
                    shape       = (self.in_caps, self.num_capsules, self.in_dim, self.dim_capsules),
                    initializer = "glorot_uniform",
                    trainable   = True,
                )
                super().build(input_shape)

            def call(self, u):
                B  = tf.shape(u)[0]
                # u shape: (B, in_caps, in_dim)
                # W shape: (in_caps, num_capsules, in_dim, dim_capsules)
                # u_hat shape: (B, in_caps, num_capsules, dim_capsules)
                u_hat = tf.einsum('bic,ijcd->bijd', u, self.W)

                b = tf.zeros([B, self.in_caps, self.num_capsules, 1])
                for i in range(self.routing_iters):
                    c    = tf.nn.softmax(b, axis=2)                            
                    s    = tf.reduce_sum(c * u_hat, axis=1, keepdims=True)     
                    v    = squash(s, axis=-1)                                  
                    if i < self.routing_iters - 1:
                        b += tf.reduce_sum(u_hat * v, axis=-1, keepdims=True)  
                return tf.squeeze(v, axis=1)                                   
            
            def get_config(self):
                config = super().get_config()
                config.update({
                    "num_capsules": self.num_capsules,
                    "dim_capsules": self.dim_capsules,
                    "routing_iters": self.routing_iters
                })
                return config

        @keras.saving.register_keras_serializable()
        class Length(layers.Layer):
            """Computes the L2 norm of each capsule vector."""
            def call(self, x):
                return tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1))

        @keras.saving.register_keras_serializable()
        def margin_loss(y_true, y_pred, m_plus=0.9, m_minus=0.1, lam=0.5):
            """Margin loss for CapsNet."""
            L = (y_true * tf.square(tf.maximum(0.0, m_plus - y_pred))
                 + lam * (1.0 - y_true) * tf.square(tf.maximum(0.0, y_pred - m_minus)))
            return tf.reduce_mean(tf.reduce_sum(L, axis=1))

        logger.info("Building CapsNet architecture...")
        
        # 1. Inputs
        inp = keras.Input(shape=(_IMG_SIZE[0], _IMG_SIZE[1], 3), name="input_layer")
        
        # 2. Initial Conv2D
        x = layers.Conv2D(96, 5, strides=1, activation="relu", padding="valid", name="conv2d")(inp)
        x = layers.Dropout(0.3, name="dropout")(x)
        
        # 3. Primary Capsules (Conv2D -> Reshape -> Squash)
        x = layers.Conv2D(96, 5, strides=2, activation="relu", padding="valid", name="conv2d_1")(x)
        x = layers.Dropout(0.3, name="dropout_1")(x)
        primary_caps = layers.Reshape((-1, 8), name="reshape")(x)
        primary_caps = layers.Lambda(squash, name="lambda")(primary_caps)
        
        # 4. Digit Capsules
        digit_caps = CapsuleLayer(
            num_capsules=43, dim_capsules=16, routing_iters=3, name="capsule_layer"
        )(primary_caps)
        
        # 5. Length (Norm)
        def length_norm(x):
            return tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1))
            
        output = layers.Lambda(length_norm, name="lambda_1")(digit_caps)
        
        _model = keras.Model(inputs=inp, outputs=output, name="CapsNet_GTSRB")
        
        logger.info("Loading weights from: %s", target)
        _model.load_weights(str(target))
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
