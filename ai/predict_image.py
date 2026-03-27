"""Run top-3 predictions on a local origami image using the trained model."""

import argparse
import os
import json
import sys

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from ai.image_preprocessing import preprocess_rgb_image_like_training

MODEL_PATH = os.path.join(BASE_DIR, "origami_model.h5")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "label_map.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict an origami image with a trained MobileNetV2 model.")
    parser.add_argument("image_path", nargs="?", help="Path to a local .jpg or .png file.")
    parser.add_argument("--image", dest="image_flag", help="Path to a local .jpg or .png file.")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to the trained .h5 model file.")
    return parser.parse_args()


def resolve_image_path(args: argparse.Namespace) -> str:
    image_path = args.image_flag or args.image_path

    if image_path:
        return image_path.strip().strip('"')

    entered_path = input("Enter the image path (.jpg or .png): ").strip().strip('"')
    if not entered_path:
        return ""
    return entered_path


def load_model(model_path: str) -> tf.keras.Model:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return tf.keras.models.load_model(model_path)


def load_labels_dict() -> dict[int, str]:
    if os.path.exists(LABEL_MAP_PATH):
        with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
            raw_map = json.load(f)
        return {int(idx): str(label) for idx, label in raw_map.items()}
    print(f"[WARN] Label map not found: {LABEL_MAP_PATH}. Falling back to class indices.")
    return {}


def load_and_preprocess_image(image_path: str) -> tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Could not read image: {image_path}")

    original_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    normalized = preprocess_rgb_image_like_training(original_rgb, img_size=IMG_SIZE)
    batch = np.expand_dims(normalized, axis=0)
    return original_rgb, batch


def get_top_predictions(
    predictions: np.ndarray,
    index_to_label: dict[int, str],
    top_k: int = 3,
) -> list[tuple[int, str, float]]:
    scores = predictions[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]

    results = []
    for class_index in top_indices:
        label = index_to_label.get(class_index, f"class_{class_index}")
        confidence = float(scores[class_index]) * 100.0
        results.append((int(class_index), label, confidence))
    return results


def draw_prediction(original_rgb: np.ndarray, best_label: str, best_confidence: float) -> np.ndarray:
    annotated_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
    text = f"{best_label.upper()} - {best_confidence:.1f}%"

    cv2.rectangle(annotated_bgr, (0, 0), (annotated_bgr.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(
        annotated_bgr,
        text,
        (12, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    args = parse_args()
    image_path = resolve_image_path(args)

    if not image_path:
        print("No image path was provided.")
        print("Example:")
        print("  python ai/predict_image.py --image path/to/image.jpg")
        print("  python ai/predict_image.py path/to/image.jpg")
        return

    model = load_model(args.model)
    index_to_label = load_labels_dict()
    original_rgb, image_batch = load_and_preprocess_image(image_path)
    predictions = model.predict(image_batch, verbose=0)
    top_predictions = get_top_predictions(predictions, index_to_label=index_to_label, top_k=3)

    print("Top 3 predictions:")
    for _, label, confidence in top_predictions:
        print(f"- {label}: {confidence:.2f}%")

    best_index, best_label, best_confidence = top_predictions[0]
    print(f"\nBest match: {best_label} (class index: {best_index})")

    annotated_rgb = draw_prediction(original_rgb, best_label, best_confidence)
    plt.figure(figsize=(10, 8))
    plt.imshow(annotated_rgb)
    plt.title("Origami prediction")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()