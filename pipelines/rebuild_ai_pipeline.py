"""
Post-scrape pipeline for ORC -> AI training data -> model retraining.

Usage examples:
    py -3 pipelines/rebuild_ai_pipeline.py
    py -3 pipelines/rebuild_ai_pipeline.py --skip-train
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import uuid
from typing import Any

import psycopg2
from psycopg2.extras import RealDictCursor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scrapers import orc_scraping

SPAM_TOKENS = (
    "contact us",
    "reporting broken links",
    "site map",
    "back to ",
    "buy ",
    "book review",
    "privacy",
    "terms",
    "disclaimer",
)


def normalize_name(value: str) -> str:
    s = (value or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def is_spam_model_name(name: str) -> bool:
    low = (name or "").strip().lower()
    if not low:
        return True
    return any(token in low for token in SPAM_TOKENS)


def get_conn():
    return psycopg2.connect(
        host=orc_scraping.POSTGRES_HOST,
        port=orc_scraping.POSTGRES_PORT,
        database=orc_scraping.POSTGRES_DB,
        user=orc_scraping.POSTGRES_USER,
        password=orc_scraping.POSTGRES_PASSWORD,
        sslmode="require",
        connect_timeout=30,
    )


def ensure_orc_creator(cur) -> str:
    name_original = "Origami Resource Center"
    name_normalized = normalize_name(name_original)

    cur.execute(
        """
        INSERT INTO creators (creator_id, name_original, name_normalized)
        VALUES (%s, %s, %s)
        ON CONFLICT (name_normalized) DO NOTHING
        """,
        (str(uuid.uuid4()), name_original, name_normalized),
    )

    cur.execute("SELECT creator_id FROM creators WHERE name_normalized = %s", (name_normalized,))
    row = cur.fetchone()
    if not row:
        raise RuntimeError("Could not resolve ORC creator_id")
    return str(row["creator_id"])


def select_orc_rows(cur) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT
            id,
            model_name,
            model_name_base,
            category,
            source_page_url,
            diagram_url,
            image_url,
            cloudinary_url,
            creator_expanded,
            creator_raw
        FROM orc_models
        ORDER BY id
        """
    )
    rows = cur.fetchall()

    filtered: list[dict[str, Any]] = []
    for row in rows:
        name = (row["model_name"] or "").strip()
        if is_spam_model_name(name):
            continue
        filtered.append(row)

    return filtered


def find_or_insert_model(cur, row: dict[str, Any], creator_id: str) -> tuple[str, bool]:
    model_name_original = (row.get("model_name") or "").strip()
    model_name_normalized = normalize_name(row.get("model_name_base") or model_name_original)
    source_url = row.get("diagram_url") or row.get("source_page_url")
    paper_shape = (row.get("category") or "").replace("_", " ")[:100] or None

    # models.source_url has a global uniqueness constraint by normalized URL.
    # Reuse that model first to avoid insert-time UniqueViolation.
    if source_url:
        cur.execute(
            """
            SELECT model_id
            FROM models
            WHERE lower(regexp_replace(TRIM(BOTH FROM source_url), '/+$', ''))
                  = lower(regexp_replace(TRIM(BOTH FROM %s), '/+$', ''))
            LIMIT 1
            """,
            (source_url,),
        )
        by_source = cur.fetchone()
        if by_source:
            return str(by_source["model_id"]), False

    cur.execute(
        """
        SELECT model_id
        FROM models
        WHERE creator_id = %s
          AND model_name_normalized = %s
          AND COALESCE(source_url, '') = COALESCE(%s, '')
        LIMIT 1
        """,
        (creator_id, model_name_normalized, source_url),
    )
    existing = cur.fetchone()
    if existing:
        return str(existing["model_id"]), False

    model_id = str(uuid.uuid4())
    cur.execute(
        """
        INSERT INTO models (
            model_id,
            creator_id,
            model_name_original,
            model_name_normalized,
            paper_shape,
            pieces,
            uses_cutting,
            uses_glue,
            difficulty,
            source_url
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            model_id,
            creator_id,
            model_name_original,
            model_name_normalized,
            paper_shape,
            None,
            False,
            False,
            2,
            source_url,
        ),
    )
    return model_id, True


def ensure_image_for_model(cur, model_id: str, row: dict[str, Any]) -> bool:
    cloud_url = (row.get("cloudinary_url") or "").strip()
    raw_url = (row.get("image_url") or row.get("diagram_url") or "").strip()

    if not cloud_url:
        return False

    cur.execute(
        """
        SELECT image_id
        FROM images
        WHERE model_id = %s
          AND COALESCE(cloudinary_url, '') = %s
        LIMIT 1
        """,
        (model_id, cloud_url),
    )
    if cur.fetchone():
        return False

    image_id = str(uuid.uuid4())
    cur.execute(
        """
        INSERT INTO images (
            image_id,
            model_id,
            url,
            cloudinary_url,
            is_primary
        )
        VALUES (%s, %s, %s, %s, %s)
        """,
        (image_id, model_id, raw_url or None, cloud_url, True),
    )
    return True


def sync_orc_to_ai_tables() -> tuple[int, int, int]:
    """
    Returns:
        (models_inserted, images_inserted, processed_rows)
    """
    conn = get_conn()
    models_inserted = 0
    images_inserted = 0
    processed = 0
    batch_size = 50

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            creator_id = ensure_orc_creator(cur)
            rows = select_orc_rows(cur)
            total_rows = len(rows)
            print(f"    Total ORC rows to process: {total_rows}")

            for row in rows:
                model_id, inserted_model = find_or_insert_model(cur, row, creator_id)
                if inserted_model:
                    models_inserted += 1

                inserted_img = ensure_image_for_model(cur, model_id, row)
                if inserted_img:
                    images_inserted += 1

                processed += 1
                if processed % batch_size == 0:
                    conn.commit()
                    print(
                        f"    Progress: {processed}/{total_rows} | "
                        f"models+ {models_inserted} | images+ {images_inserted}"
                    )

            # Final progress line for non-exact batch endings
            if processed % batch_size != 0:
                print(
                    f"    Progress: {processed}/{total_rows} | "
                    f"models+ {models_inserted} | images+ {images_inserted}"
                )

        conn.commit()
        return models_inserted, images_inserted, len(rows)
    finally:
        conn.close()


def run_train() -> int:
    cmd = [sys.executable, os.path.join("ai", "train_model.py")]
    return subprocess.call(cmd, cwd=ROOT_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-scrape ORC sync + AI retrain pipeline")
    parser.add_argument("--skip-train", action="store_true", help="Skip model retraining phase")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 70)
    print("AI PIPELINE: ORC post-scrape sync -> retrain")
    print("=" * 70)

    print("\n[1/2] Syncing ORC data into models/images...")
    models_inserted, images_inserted, processed_rows = sync_orc_to_ai_tables()
    print(f"    Processed ORC rows: {processed_rows}")
    print(f"    Models inserted   : {models_inserted}")
    print(f"    Images inserted   : {images_inserted}")

    if not args.skip_train:
        print("\n[2/2] Retraining AI model...")
        rc = run_train()
        if rc != 0:
            print(f"[ERROR] Training failed with exit code {rc}")
            return rc
    else:
        print("\n[2/2] Training skipped.")

    print("\n[DONE] Pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
