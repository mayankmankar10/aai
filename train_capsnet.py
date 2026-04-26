"""
train_capsnet.py
================
Downloads the GTSRB dataset, builds a CapsNet, trains it, and saves
the model to  traffic_sign_capsnet.h5  in the project root.

Usage
-----
    python train_capsnet.py

No manual downloads required — tensorflow_datasets handles everything.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import tensorflow_datasets as tfds

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
IMG_SIZE    = 32
NUM_CLASSES = 43
BATCH_SIZE  = 64
EPOCHS      = 15
SAVE_PATH   = "traffic_sign_capsnet.h5"

print("=" * 60)
print("  CapsNet — GTSRB Traffic Sign Trainer")
print("=" * 60)
print(f"  TensorFlow : {tf.__version__}")
print(f"  Image size : {IMG_SIZE}x{IMG_SIZE}")
print(f"  Classes    : {NUM_CLASSES}")
print(f"  Epochs     : {EPOCHS}")
print(f"  Batch size : {BATCH_SIZE}")
print(f"  Save path  : {SAVE_PATH}")
print("=" * 60)

# ---------------------------------------------------------------------------
# 1. Load & preprocess GTSRB via tensorflow_datasets
# ---------------------------------------------------------------------------
print("\n[1/5] Downloading / loading GTSRB dataset …")

def preprocess(sample):
    img   = tf.image.resize(tf.cast(sample["image"], tf.float32), [IMG_SIZE, IMG_SIZE]) / 255.0
    label = tf.one_hot(sample["label"], NUM_CLASSES)
    return img, label

(ds_train_raw, ds_test_raw), info = tfds.load(
    "german_credit_numeric",          # fallback handled below
    split=["train", "test"],
    with_info=True,
    as_supervised=False,
)

# ── use the correct dataset name ──────────────────────────────────────────
try:
    (ds_train_raw, ds_test_raw), info = tfds.load(
        "gtsrb",
        split=["train", "test"],
        with_info=True,
        as_supervised=False,
    )
    print(f"  Dataset info: {info.splits}")
except Exception:
    # tfds name variant
    (ds_train_raw, ds_test_raw), info = tfds.load(
        "german_traffic_sign",
        split=["train", "test"],
        with_info=True,
        as_supervised=False,
    )
    print(f"  Dataset info: {info.splits}")

ds_train = (
    ds_train_raw
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(10_000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
ds_test = (
    ds_test_raw
    .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

print("  Dataset ready.\n")

# ---------------------------------------------------------------------------
# 2. CapsNet building blocks
# ---------------------------------------------------------------------------
print("[2/5] Building CapsNet architecture …")

def squash(vectors, axis=-1):
    """Squashing activation — keeps direction, limits magnitude to (0,1)."""
    s_sq   = tf.reduce_sum(tf.square(vectors), axis=axis, keepdims=True)
    scale  = s_sq / (1.0 + s_sq) / tf.sqrt(s_sq + keras.backend.epsilon())
    return scale * vectors


class PrimaryCaps(layers.Layer):
    """
    Primary Capsule layer.
    Applies a Conv2D and reshapes into (batch, num_capsules, dim_capsule).
    """
    def __init__(self, num_capsules, dim_capsule, conv_filters, kernel_size, strides, **kw):
        super().__init__(**kw)
        self.num_capsules = num_capsules
        self.dim_capsule  = dim_capsule
        self.conv = layers.Conv2D(
            filters     = conv_capsules * dim_capsule,
            kernel_size = kernel_size,
            strides     = strides,
            padding     = "valid",
            activation  = "relu",
        )

    def build(self, input_shape):
        self.conv.build(input_shape)
        super().build(input_shape)

    def call(self, x):
        out  = self.conv(x)                                           # (B, H, W, caps*dim)
        B    = tf.shape(out)[0]
        out  = tf.reshape(out, [B, -1, self.dim_capsule])             # (B, total_caps, dim)
        return squash(out)


class DigitCaps(layers.Layer):
    """
    Digit Capsule layer with dynamic routing (routing iterations = 3).
    """
    def __init__(self, num_capsules, dim_capsule, routing_iters=3, **kw):
        super().__init__(**kw)
        self.num_capsules  = num_capsules
        self.dim_capsule   = dim_capsule
        self.routing_iters = routing_iters

    def build(self, input_shape):
        # input_shape: (batch, in_caps, in_dim)
        self.in_caps = input_shape[1]
        self.in_dim  = input_shape[2]
        self.W = self.add_weight(
            name        = "routing_weights",
            shape       = (1, self.in_caps, self.num_capsules, self.dim_capsule, self.in_dim),
            initializer = "glorot_uniform",
            trainable   = True,
        )
        super().build(input_shape)

    def call(self, u):
        # u: (B, in_caps, in_dim)
        B  = tf.shape(u)[0]
        u_ = tf.expand_dims(tf.expand_dims(u, 2), 4)                  # (B, in_caps, 1, in_dim, 1)
        W  = tf.tile(self.W, [B, 1, 1, 1, 1])                         # (B, in_caps, out_caps, out_dim, in_dim)
        u_hat = tf.squeeze(tf.matmul(W, u_), axis=-1)                  # (B, in_caps, out_caps, out_dim)

        # Dynamic routing
        b = tf.zeros([B, self.in_caps, self.num_capsules, 1])
        for i in range(self.routing_iters):
            c    = tf.nn.softmax(b, axis=2)                            # (B, in_caps, out_caps, 1)
            s    = tf.reduce_sum(c * u_hat, axis=1, keepdims=True)     # (B, 1, out_caps, out_dim)
            v    = squash(s, axis=-1)                                  # squash over out_dim
            if i < self.routing_iters - 1:
                b += tf.reduce_sum(u_hat * v, axis=-1, keepdims=True)  # agreement update

        return tf.squeeze(v, axis=1)                                   # (B, out_caps, out_dim)


class CapsuleNorm(layers.Layer):
    """Computes the L2 norm of each capsule vector → class probabilities."""
    def call(self, x):
        return tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1))           # (B, num_capsules)


# ── Hyper-parameters ─────────────────────────────────────────────────────
conv_capsules = 32          # number of primary capsule types
primary_dim   = 8           # dimensions per primary capsule
digit_dim     = 16          # dimensions per digit capsule

# ── Build model ───────────────────────────────────────────────────────────
inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")

# Initial convolution to extract features
x = layers.Conv2D(256, 9, activation="relu", padding="valid", name="conv1")(inp)

# Primary capsules
# After Conv(256, 9, valid) on 32x32: output is 24x24x256
# PrimaryCaps conv(9,2,valid): output is 8x8
primary_out = layers.Conv2D(
    filters     = conv_capsules * primary_dim,
    kernel_size = 9,
    strides     = 2,
    padding     = "valid",
    activation  = "relu",
    name        = "primary_conv",
)(x)
# Reshape to capsules
B_   = tf.shape(primary_out)[0]
primary_caps = layers.Reshape((-1, primary_dim), name="primary_caps")(primary_out)
primary_caps = layers.Lambda(squash, name="primary_squash")(primary_caps)

# Digit capsules  (one per class)
digit_caps = DigitCaps(
    num_capsules  = NUM_CLASSES,
    dim_capsule   = digit_dim,
    routing_iters = 3,
    name          = "digit_caps",
)(primary_caps)

# Output: capsule L2 norms  →  (B, 43)  probability-like scores
output = CapsuleNorm(name="capsule_norm")(digit_caps)

model = Model(inputs=inp, outputs=output, name="CapsNet_GTSRB")
model.summary()
print()

# ---------------------------------------------------------------------------
# 3. Compile
# ---------------------------------------------------------------------------
print("[3/5] Compiling …")

# Margin loss — standard CapsNet objective
def margin_loss(y_true, y_pred, m_plus=0.9, m_minus=0.1, lam=0.5):
    L = (y_true * tf.square(tf.maximum(0.0, m_plus - y_pred))
         + lam * (1.0 - y_true) * tf.square(tf.maximum(0.0, y_pred - m_minus)))
    return tf.reduce_mean(tf.reduce_sum(L, axis=1))

model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=1e-3),
    loss      = margin_loss,
    metrics   = [
        keras.metrics.CategoricalAccuracy(name="accuracy"),
    ],
)
print("  Done.\n")

# ---------------------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------------------
print("[4/5] Training …")

callbacks = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=3, verbose=1, min_lr=1e-6
    ),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ModelCheckpoint(
        filepath          = SAVE_PATH,
        monitor           = "val_accuracy",
        save_best_only    = True,
        save_weights_only = False,
        verbose           = 1,
    ),
]

history = model.fit(
    ds_train,
    epochs          = EPOCHS,
    validation_data = ds_test,
    callbacks       = callbacks,
)

# ---------------------------------------------------------------------------
# 5. Final save & evaluation
# ---------------------------------------------------------------------------
print("\n[5/5] Saving final model …")
model.save(SAVE_PATH)

# Evaluate
print("\n--- Final Evaluation on Test Set ---")
results = model.evaluate(ds_test, verbose=1)
for name, val in zip(model.metrics_names, results):
    print(f"  {name}: {val:.4f}")

print(f"\n✅ Model saved to: {os.path.abspath(SAVE_PATH)}")
print("   You can now run the Flask backend and React frontend.")
