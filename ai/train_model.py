"""
Origami Transfer Learning with MobileNetV2
========================================
Implements Transfer Learning on MobileNetV2 (ImageNet weights) for origami
category classification. Builds, compiles, and trains the model.

Usage:
    python ai/train_model.py
"""

import sys
import os
import json
import re

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

tf.get_logger().setLevel("ERROR")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from ai.data_generator import load_dataframe, OrigamiDataGenerator  # noqa: E402

# Configuration constants
IMG_SIZE        = (224, 224)
BATCH_SIZE      = 32
EPOCHS          = int(os.getenv("ORIGAMI_EPOCHS", "50"))
DROPOUT_RATE    = 0.5
LEARNING_RATE   = 5e-5
VAL_SPLIT       = 0.20
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_model.keras")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "origami_model.h5")
PLOT_SAVE_PATH  = os.path.join(BASE_DIR, "training_history.png")
LABEL_MAP_PATH  = os.path.join(BASE_DIR, "label_map.json")

MIN_SAMPLES_PER_CLASS = int(os.getenv("ORIGAMI_MIN_SAMPLES_PER_CLASS", "3"))
MAX_CLASSES = int(os.getenv("ORIGAMI_MAX_CLASSES", "50"))
RESET_ARTIFACTS = os.getenv("ORIGAMI_RESET_ARTIFACTS", "1") == "1"

NOISY_EXACT_LABELS = {
    "index",
    "pg 1",
    "pg1",
}

NOISY_CATEGORY_HEADER_HINTS = {
    "animals",
    "birds",
    "boxes",
    "butterflies",
    "dinosaurs",
    "holiday",
    "dollar",
    "origami",
    "fish",
    "flowers",
    "hearts",
    "insects",
    "modular",
    "airplane",
    "cranes",
    "polyhedra",
    "reptile",
    "star",
    "wars",
    "vehicles",
}


# ─ Model Architecture

def build_model(num_classes: int) -> tf.keras.Model:
    """
    Builds a Transfer Learning model based on MobileNetV2 (include_top=False).

    Architecture:
        MobileNetV2 (frozen) → GlobalAveragePooling2D
        → Dense(256, relu) → Dropout(0.4)
        → Dense(num_classes, softmax)

    Parameters
    ----------
    num_classes : int
        Number of origami categories.

    Returns
    -------
    keras.Model
        Compiled model ready for training.
    """
    # ─ Base model (frozen)────────
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,          # Without ImageNet classification head
        weights="imagenet",
    )
    base_model.trainable = False

    # ─ Classification head────
    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3), name="image_input")

    # Data augmentation layers
    x = tf.keras.layers.RandomRotation(0.15, name="augment_rotate")(inputs)
    x = tf.keras.layers.RandomZoom(0.15, name="augment_zoom")(x)
    x = tf.keras.layers.RandomFlip("horizontal", name="augment_flip")(x)

    # MobileNetV2 preprocessing: [0,1] -> [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(256, activation="relu", name="fc1")(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE, name="dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="origami_mobilenetv2")

    # ─ Compile───────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ─ Train/Validation split

def split_dataframe_by_label(df: pd.DataFrame, val_split: float = VAL_SPLIT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data per-label to ensure validation set doesn't have classes unseen in training.
    Single-sample classes are kept in the training set.
    """
    train_parts = []
    val_parts = []

    for _, group in df.groupby("label", sort=False):
        group = group.sample(frac=1.0, random_state=42)
        sample_count = len(group)

        if sample_count == 1:
            train_parts.append(group)
            continue

        val_count = max(1, int(round(sample_count * val_split)))
        val_count = min(val_count, sample_count - 1)

        val_parts.append(group.iloc[:val_count])
        train_parts.append(group.iloc[val_count:])

    train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return train_df, val_df


def reset_training_artifacts() -> None:
    """Remove previous model artifacts for a clean retrain run."""
    paths_to_remove = [CHECKPOINT_PATH, MODEL_SAVE_PATH, PLOT_SAVE_PATH, LABEL_MAP_PATH]

    for path in paths_to_remove:
        if os.path.exists(path):
            try:
                os.remove(path)
                print(f"    Removed old artifact: {path}")
            except OSError as exc:
                print(f"    Could not remove {path}: {exc}")


def is_noisy_label(label: str) -> bool:
    """Detect obvious non-class labels coming from scraped navigation/page text."""
    normalized = str(label).strip().lower()
    if not normalized:
        return True

    if normalized in NOISY_EXACT_LABELS:
        return True

    if re.fullmatch(r"pg\s*\d+", normalized):
        return True

    tokens = [token for token in re.split(r"\W+", normalized) if token]
    if len(tokens) >= 8:
        hint_hits = sum(1 for token in tokens if token in NOISY_CATEGORY_HEADER_HINTS)
        if hint_hits >= 5:
            return True

    return False


def remove_noisy_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that belong to clearly noisy labels before class balancing."""
    noisy_mask = df["label"].map(is_noisy_label)
    noisy_count = int(noisy_mask.sum())

    if noisy_count == 0:
        return df

    noisy_labels = sorted(df.loc[noisy_mask, "label"].astype(str).unique().tolist())
    print(f"    Removed noisy labels: {len(noisy_labels)} class(es), {noisy_count} image(s)")
    for label in noisy_labels[:10]:
        print(f"      - {label}")
    if len(noisy_labels) > 10:
        print(f"      ... and {len(noisy_labels) - 10} more")

    return df.loc[~noisy_mask].copy().reset_index(drop=True)


def reduce_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only classes with enough samples and then keep the most frequent classes.
    This helps the model focus on stronger classes and improves stability.
    """
    df = remove_noisy_labels(df)

    counts = df["label"].value_counts()
    filtered_counts = counts[counts >= MIN_SAMPLES_PER_CLASS]

    if filtered_counts.empty:
        raise ValueError(
            "No classes left after filtering. "
            f"Lower ORIGAMI_MIN_SAMPLES_PER_CLASS (current: {MIN_SAMPLES_PER_CLASS})."
        )

    selected_labels = filtered_counts.index.tolist()
    if MAX_CLASSES > 0:
        selected_labels = selected_labels[:MAX_CLASSES]

    reduced_df = df[df["label"].isin(selected_labels)].copy().reset_index(drop=True)
    return reduced_df


def save_label_map(label_map: dict[str, int], save_path: str = LABEL_MAP_PATH) -> None:
    """Persist class mapping so inference uses the exact training index order."""
    # Save as index -> label for simple lookup at prediction time.
    index_to_label = {str(index): label for label, index in label_map.items()}
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(index_to_label, f, ensure_ascii=False, indent=2)
    print(f"    Label map saved: {save_path}")


# ─ Callbacks

def build_callbacks(checkpoint_path: str = CHECKPOINT_PATH) -> list:
    """
    EarlyStopping + ModelCheckpoint callback siyahisini qaytarir.
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    return [early_stop, checkpoint]


# ─ Training plots

def plot_training_history(history: tf.keras.callbacks.History, plot_path: str = PLOT_SAVE_PATH) -> None:
    """Accuracy ve loss qrafiklerini matplotlib ile cizer ve yadda saxlayir."""
    history_data = history.history
    epochs_range = range(1, len(history_data["loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history_data["accuracy"], label="Train Accuracy")
    plt.plot(epochs_range, history_data["val_accuracy"], label="Val Accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history_data["loss"], label="Train Loss")
    plt.plot(epochs_range, history_data["val_loss"], label="Val Loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"    Training plots saved: {plot_path}")


# ─ Main training workflow

def main():
    print("=" * 60)
    print("  Origami Transfer Learning - MobileNetV2")
    print("=" * 60)

    # ─ Data───────────────
    print("\n[1] Loading data from DB...")
    df = load_dataframe()
    print(f"    Raw data: {len(df)} images, {df['label'].nunique()} classes")

    if RESET_ARTIFACTS:
        print("\n[1.1] Resetting previous training artifacts...")
        reset_training_artifacts()

    print("\n[1.2] Reducing class set...")
    raw_classes = df["label"].nunique()
    df = reduce_classes(df)
    print(
        "    Reduced data: "
        f"{len(df)} images, {df['label'].nunique()} classes "
        f"(from {raw_classes})"
    )

    train_df, val_df = split_dataframe_by_label(df, val_split=VAL_SPLIT)
    print(f"    Train: {len(train_df)} | Val: {len(val_df)}")

    # ─ Generators───────────
    print("\n[2] Building data generators...")
    label_map = {label: idx for idx, label in enumerate(sorted(df["label"].unique()))}
    train_gen = OrigamiDataGenerator(
        train_df,
        batch_size=BATCH_SIZE,
        shuffle=True,
        label_map=label_map,
        one_hot_labels=True,
    )
    val_gen = OrigamiDataGenerator(
        val_df,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_map=label_map,
        one_hot_labels=True,
    )

    num_classes = len(label_map)
    print(f"    Num classes: {num_classes}")

    # ─ Model──────────────
    print("\n[3] Building model...")
    model = build_model(num_classes)
    model.summary(line_length=80)

    # ─ Training───────────────
    print("\n[4] Starting model.fit()...")
    callbacks = build_callbacks()

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n[5] Saving final model...")
    model.save(MODEL_SAVE_PATH)
    print(f"    Final model saved: {MODEL_SAVE_PATH}")

    save_label_map(label_map)

    print("\n[6] Plotting training history...")
    plot_training_history(history)

    return model, label_map, history


# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
