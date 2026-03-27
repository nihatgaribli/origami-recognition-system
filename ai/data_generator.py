"""
Origami Image Data Generator
==============================
Loads Cloudinary image URLs and labels from the database (images JOIN models).
Leverages Cloudinary's transformation API (w_224,h_224,c_fill) for optimized
image preprocessing and provides a custom TensorFlow Sequence-based data generator
for efficient training.
"""

import sys
import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import re
import math
import time

import numpy as np
import pandas as pd
import requests
import cv2

import tensorflow as tf
from sqlalchemy import create_engine

# Add DB folder to path for config imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, ROOT_DIR)

from visualization._db_config import (  # noqa: E402
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB,
    POSTGRES_USER, POSTGRES_PASSWORD,
)

# Configuration constants
IMG_SIZE     = (224, 224)          # (height, width)
BATCH_SIZE   = 32
SHUFFLE      = True
REQUEST_CONNECT_TIMEOUT = float(os.getenv("ORIGAMI_REQUEST_CONNECT_TIMEOUT", "4"))
REQUEST_READ_TIMEOUT = float(os.getenv("ORIGAMI_REQUEST_READ_TIMEOUT", "8"))
REQUEST_RETRIES = int(os.getenv("ORIGAMI_REQUEST_RETRIES", "2"))
RETRY_BACKOFF_SEC = float(os.getenv("ORIGAMI_RETRY_BACKOFF_SEC", "0.75"))

# Cloudinary transformation parameters for optimized image delivery
CLD_TRANSFORM = "w_224,h_224,c_fill"

_HTTP_SESSION: requests.Session | None = None
_FAILED_URLS: set[str] = set()
_FAILED_URLS_WARNED: set[str] = set()


def get_http_session() -> requests.Session:
    global _HTTP_SESSION
    if _HTTP_SESSION is None:
        s = requests.Session()
        s.headers.update({"User-Agent": "origami-trainer/1.0"})
        _HTTP_SESSION = s
    return _HTTP_SESSION


# ─ Load DataFrame from database

def load_dataframe() -> pd.DataFrame:
    """
    Fetches cloudinary_url and model_name_original from images JOIN models.
    Returns only rows where cloudinary_url is not null or empty.
    """
    query = """
        SELECT
            i.cloudinary_url,
            m.model_name_original AS label
        FROM images i
        INNER JOIN models m ON i.model_id = m.model_id
        WHERE i.cloudinary_url IS NOT NULL
          AND i.cloudinary_url <> ''
          AND m.model_name_original IS NOT NULL
          AND TRIM(m.model_name_original) <> ''
            AND POSITION('facebook' IN LOWER(m.model_name_original)) = 0
            AND POSITION('free origami instructions' IN LOWER(m.model_name_original)) = 0
            AND POSITION('more origami instructions' IN LOWER(m.model_name_original)) = 0
            AND POSITION('see more origami instructions' IN LOWER(m.model_name_original)) = 0
    """
    engine = create_engine(
        f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
        f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
        connect_args={"sslmode": "require", "connect_timeout": 30},
    )
    df = pd.read_sql_query(query, engine)
    engine.dispose()

    df = df.dropna(subset=["cloudinary_url", "label"]).reset_index(drop=True)
    print(f"[DB] {len(df)} rows loaded, {df['label'].nunique()} unique labels.")
    return df


# ─ Attach Cloudinary transformation to URL

def add_cloudinary_transform(url: str, transform: str = CLD_TRANSFORM) -> str:
    """
    Cloudinary URL-inə `/upload/<transform>/` hissəsini əlavə edir.

    Nümunə:
        https://res.cloudinary.com/demo/image/upload/sample.jpg
        →
        https://res.cloudinary.com/demo/image/upload/w_224,h_224,c_fill/sample.jpg
    """
    # Artıq transformasiya varsa dəyişdirme
    if transform in url:
        return url
    # /upload/ sonrasına transformasiya yerləşdir
    return re.sub(r"(/upload/)", rf"\1{transform}/", url, count=1)


# ─ Fetch single image from URL

def fetch_image(url: str) -> np.ndarray | None:
    """
    Cloudinary URL-dən şəkili requests ilə yükləyir,
    OpenCV ilə dekod edir, (224, 224, 3) ölçüsinə çevirir
    Fetches and normalizes images in [0, 1] range.

    Xəta halında None qaytarır.
    """
    transformed_url = add_cloudinary_transform(url)

    # Skip URLs already known to fail; this prevents repeated stalls each epoch.
    if transformed_url in _FAILED_URLS:
        return None

    session = get_http_session()

    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            response = session.get(
                transformed_url,
                timeout=(REQUEST_CONNECT_TIMEOUT, REQUEST_READ_TIMEOUT),
            )
            response.raise_for_status()

            # Bytes → NumPy array → OpenCV image
            arr = np.frombuffer(response.content, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if img_bgr is None:
                raise ValueError(f"OpenCV şəkili dekod edə bilmədi: {transformed_url}")

            # BGR → RGB
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Resize (backup - Cloudinary should have done this already)
            img_resized = cv2.resize(img_rgb, (IMG_SIZE[1], IMG_SIZE[0]),
                                     interpolation=cv2.INTER_AREA)

            # Normalize: [0, 255] → [0.0, 1.0]
            img_norm = img_resized.astype(np.float32) / 255.0
            return img_norm

        except Exception as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            # For permanent client-side failures, do not retry.
            if status_code in {400, 401, 403, 404, 410, 422}:
                _FAILED_URLS.add(transformed_url)
                if transformed_url not in _FAILED_URLS_WARNED:
                    _FAILED_URLS_WARNED.add(transformed_url)
                    print(f"[WARN] Image unavailable ({status_code}) ({url[:60]}...)")
                return None

            if attempt < REQUEST_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)
                continue

            _FAILED_URLS.add(transformed_url)
            if transformed_url not in _FAILED_URLS_WARNED:
                _FAILED_URLS_WARNED.add(transformed_url)
                print(f"[WARN] Image load failed ({url[:60]}...): {exc}")
                return None

            return None


# ─ Custom Keras Data Generator using keras.utils.Sequence

class OrigamiDataGenerator(tf.keras.utils.Sequence):
    """
    Keras Sequence əsaslı data generator.

    Hər epoch-da batch-ləri:
      - Cloudinary URL-lərindən şəkilləri yükləyir (requests + OpenCV)
      - (224, 224, 3) float32 tensoru qaytarır
      - Label-ləri integer index-ə çevirir

    Parameters
    ----------
    df : pd.DataFrame
        'cloudinary_url' və 'label' sütunları olan DataFrame.
    batch_size : int
        Mini-batch ölçüsü.
    shuffle : bool
        Hər epoch sonunda datasetə shuffle tətbiq edilsinmi.
    label_map : dict | None
        {label_str: int_index} lüğəti.  None olduqda df-dən avtomatik qurulur.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        batch_size: int = BATCH_SIZE,
        shuffle: bool = SHUFFLE,
        label_map: dict | None = None,
        one_hot_labels: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.df         = df.copy().reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.one_hot_labels = one_hot_labels

        # Encode label to one-hot vector
        if label_map is None:
            unique_labels = sorted(self.df["label"].unique())
            self.label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
        else:
            self.label_map = label_map

        self.num_classes = len(self.label_map)
        self.indices     = np.arange(len(self.df))

        if self.shuffle:
            np.random.shuffle(self.indices)

    # Keras Sequence interface methods

    def __len__(self) -> int:
        """Hər epoch üçün batch sayı."""
        return math.ceil(len(self.df) / self.batch_size)

    def __getitem__(self, batch_idx: int):
        """batch_idx-ci batch-i qaytarır: (X, y)."""
        start = batch_idx * self.batch_size
        end   = min(start + self.batch_size, len(self.df))
        batch_indices = self.indices[start:end]

        batch_rows = self.df.iloc[batch_indices]

        images: list[np.ndarray] = []
        labels: list[int] = []
        target_batch_size = len(batch_rows)

        for _, row in batch_rows.iterrows():
            img = fetch_image(row["cloudinary_url"])
            if img is None:
                continue
            images.append(img)
            labels.append(self.label_map[row["label"]])

        refill_attempts = 0
        max_refill_attempts = max(20, target_batch_size * 4)
        while len(images) < target_batch_size and refill_attempts < max_refill_attempts:
            refill_attempts += 1
            random_idx = np.random.randint(0, len(self.df))
            row = self.df.iloc[random_idx]
            img = fetch_image(row["cloudinary_url"])
            if img is None:
                continue
            images.append(img)
            labels.append(self.label_map[row["label"]])

        if not images:
            raise RuntimeError(
                "No valid images could be loaded for this batch. "
                "Check Cloudinary URLs and network access."
            )

        X = np.stack(images).astype(np.float32)
        y = np.array(labels, dtype=np.int64)

        if self.one_hot_labels:
            y = tf.keras.utils.to_categorical(y, num_classes=self.num_classes)

        return X, y

    def on_epoch_end(self):
        """Epoch sonunda, shuffle aktifdirsə, sıranı qarışdır."""
        if self.shuffle:
            np.random.shuffle(self.indices)

    # Utility methods

    def class_names(self) -> list[str]:
        """İndex sırasına görə sıralanmış label adlarını qaytarır."""
        return [k for k, _ in sorted(self.label_map.items(), key=lambda x: x[1])]


# ─ High-performance tf.data.Dataset variant (alternative approach)

def build_tf_dataset(
    df: pd.DataFrame,
    label_map: dict,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = SHUFFLE,
    buffer_size: int = 500,
) -> tf.data.Dataset:
    """
    tf.data.Dataset əsaslı pipeline.
    Python funksiyaları tf.py_function ilə sarılır.

    Returns
    -------
    tf.data.Dataset
        (image_tensor[batch, 224, 224, 3], label_tensor[batch]) cütlərini verir.
    """
    urls   = df["cloudinary_url"].tolist()
    labels = [label_map[lbl] for lbl in df["label"].tolist()]

    url_ds   = tf.data.Dataset.from_tensor_slices(urls)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds       = tf.data.Dataset.zip((url_ds, label_ds))

    if shuffle:
        ds = ds.shuffle(buffer_size=buffer_size)

    def _load(url_tensor, label_tensor):
        url = url_tensor.numpy().decode("utf-8")
        img = fetch_image(url)
        if img is None:
            img = np.zeros((*IMG_SIZE, 3), dtype=np.float32)
        lbl = np.int64(label_tensor.numpy())           # Cast int32 → int64
        return img, lbl

    def _tf_load(url, label):
        img, lbl = tf.py_function(
            func=_load,
            inp=[url, label],
            Tout=[tf.float32, tf.int64],
        )
        img.set_shape([*IMG_SIZE, 3])
        lbl.set_shape([])
        return img, lbl

    ds = ds.map(_tf_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ─ Quick test routine (run: python data_generator.py)

if __name__ == "__main__":
    print("-- Loading data from DB --")
    df = load_dataframe()
    print(df.head())

    # ── Sequence generator testi ──────────────────────────────────────────────
    print("\n-- Sequence Generator test --")
    gen = OrigamiDataGenerator(df, batch_size=4, shuffle=True)
    print(f"Num classes  : {gen.num_classes}")
    print(f"Num batches  : {len(gen)}")
    print(f"Class names  : {gen.class_names()[:5]} ...")

    X_batch, y_batch = gen[0]
    print(f"First batch X: {X_batch.shape}, dtype={X_batch.dtype}")
    print(f"First batch y: {y_batch}")

    # ── tf.data.Dataset testi ─────────────────────────────────────────────────
    print("\n-- tf.data.Dataset test --")
    dataset = build_tf_dataset(df, gen.label_map, batch_size=4, shuffle=False)
    for imgs, lbls in dataset.take(1):
        print(f"tf.data batch  X: {imgs.shape}, dtype={imgs.dtype}")
        print(f"tf.data batch  y: {lbls.numpy()}")

    print("\nAll tests passed!")
