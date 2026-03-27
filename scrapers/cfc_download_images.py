"""
CFC Origami Image Uploader (Cloudinary)
========================================
Uploads images directly to Cloudinary from cfc_diagrams/cfc_books tables
(does not store locally). Returns cloudinary_url back to database.

Usage:
    python cfc_download_images.py              # Upload all diagram images
    python cfc_download_images.py --limit 50   # Upload only first 50 diagrams
    python cfc_download_images.py --books      # Also upload book images
"""

import os
import psycopg2
import psycopg2.extras
import requests
from bs4 import BeautifulSoup
import time
import random
import re
import sys
from urllib.parse import urljoin, urlparse

import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
from dotenv import load_dotenv

load_dotenv()


# ===================== CONFIG =====================

BASE_URL = "https://cfcorigami.com"

# ===================== PostgreSQL CONFIG =====================

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "db-origami-nur-3364.c.aivencloud.com")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "19924"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "defaultdb")
POSTGRES_USER = os.getenv("POSTGRES_USER", "avnadmin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "your_password_here")


def get_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
        sslmode="require",
        connect_timeout=30,
    )


# ===================== CLOUDINARY CONFIG =====================

cloudinary.config(
    cloud_name="djig2omyz",
    api_key="521356887126377",
    api_secret="eRuxRbyTuK4osGgHF5Zgoj0WsNI",
    secure=True
)

REQUEST_DELAY_MIN = 1.0
REQUEST_DELAY_MAX = 2.5
MAX_RETRIES = 3
TIMEOUT = 20

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]


# ===================== HELPERS =====================

def get_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    })
    return session


def delay():
    time.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))


def fetch_page(session, url):
    """HTML sehifesini al."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            session.headers["User-Agent"] = random.choice(USER_AGENTS)
            delay()
            resp = session.get(url, timeout=TIMEOUT)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "html.parser")
            elif resp.status_code == 404:
                return None
            else:
                print(f"    [{resp.status_code}] {url} (attempt {attempt})")
                time.sleep(3 * attempt)
        except Exception as e:
            print(f"    [ERROR] {url}: {e} (attempt {attempt})")
            time.sleep(3 * attempt)
    return None


def upload_to_cloudinary(image_source, public_id, folder="origami"):
    """
    Sekili Cloudinary-ye yukle.
    image_source: lokal fayl yolu ve ya URL ola biler.
    public_id: Cloudinary-deki unikal ad.
    folder: Cloudinary-deki qovluq adi.
    """
    try:
        result = cloudinary.uploader.upload(
            image_source,
            public_id=public_id,
            folder=folder,
            overwrite=False,
            resource_type="image"
        )
        secure_url = result.get("secure_url", "")
        print(f"    [CLOUD] Uploaded -> {secure_url}")
        return secure_url
    except Exception as e:
        print(f"    [CLOUD_ERR] Upload ugursuz: {e}")
        return None


def slug_from_url(url):
    """URL-den slug cixar."""
    path = urlparse(url).path.rstrip("/")
    return path.split("/")[-1] if path else ""


def update_cloudinary_url(table, record_id, cloud_url, image_url=None):
    """Bazadaki cloudinary_url (ve image_url) sahesini yenile."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        if image_url:
            cur.execute(f"UPDATE {table} SET cloudinary_url = %s, image_url = %s WHERE id = %s",
                        (cloud_url, image_url, record_id))
        else:
            cur.execute(f"UPDATE {table} SET cloudinary_url = %s WHERE id = %s",
                        (cloud_url, record_id))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"    [DB_ERR] {e}")
    finally:
        cur.close()
        conn.close()


def update_image_url(table, record_id, image_url):
    """Bazadaki image_url sahesini yenile."""
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(f"UPDATE {table} SET image_url = %s WHERE id = %s", (image_url, record_id))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"    [DB_ERR] {e}")
    finally:
        cur.close()
        conn.close()


# ===================== IMAGE EXTRACTION =====================

def find_diagram_image(soup, diagram_url):
    """
    Detail sehifesinden esas sekil URL-ini tap.
    Bir nece farkli CSS selector ile calisir (Drupal site).
    """
    # 1. Preferred field: field--name-field-image
    img_field = soup.select_one(".field--name-field-image img, .field--name-field-diagram-image img")
    if img_field and img_field.get("src"):
        return urljoin(BASE_URL, img_field["src"])

    # 2. First large image within article
    article = soup.select_one("article, .node__content, .node--type-diagram")
    if article:
        for img in article.find_all("img"):
            src = img.get("src", "")
            if not src:
                continue
            if any(skip in src.lower() for skip in ["logo", "icon", "avatar", "flag", "spinner"]):
                continue
            w = img.get("width", "")
            h = img.get("height", "")
            try:
                if w and int(w) < 50:
                    continue
                if h and int(h) < 50:
                    continue
            except ValueError:
                pass
            return urljoin(BASE_URL, src)

    # 3. og:image meta tag
    og_img = soup.find("meta", property="og:image")
    if og_img and og_img.get("content"):
        return urljoin(BASE_URL, og_img["content"])

    # 4. Find suitable image from all img tags
    for img in soup.find_all("img"):
        src = img.get("src", "")
        if not src:
            continue
        if any(skip in src.lower() for skip in ["logo", "icon", "avatar", "flag", "spinner", "pixel"]):
            continue
        if "/files/" in src or "/styles/" in src or "/diagram" in src:
            return urljoin(BASE_URL, src)

    return None


def find_book_image(soup):
    """Kitab sehifesinden sekil URL-ini tap."""
    img_field = soup.select_one(".field--name-field-image img, .field--name-field-book-cover img, .field--name-field-cover img")
    if img_field and img_field.get("src"):
        return urljoin(BASE_URL, img_field["src"])

    article = soup.select_one("article, .node__content")
    if article:
        img = article.find("img")
        if img and img.get("src"):
            return urljoin(BASE_URL, img["src"])

    og_img = soup.find("meta", property="og:image")
    if og_img and og_img.get("content"):
        return urljoin(BASE_URL, og_img["content"])

    return None


# ===================== MAIN LOGIC =====================

def upload_diagram_images(limit=None):
    """PostgreSQL-deki diaqramlarin sekillerini Cloudinary-ye yukle."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        if limit:
            cur.execute("""
                SELECT id, title, url, image_url, cloudinary_url
                FROM cfc_diagrams
                ORDER BY id
                LIMIT %s
            """, (limit,))
        else:
            cur.execute("""
                SELECT id, title, url, image_url, cloudinary_url
                FROM cfc_diagrams
                ORDER BY id
            """)
        diagrams = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    if not diagrams:
        print("[INFO] Heç bir diaqram tapilmadi.")
        return

    session = get_session()
    total = len(diagrams)
    uploaded = 0
    skipped = 0
    failed = 0

    print(f"\n{'=' * 60}")
    print(f"  DIAGRAM IMAGES -> Cloudinary ({total} diagrams)")
    print(f"{'=' * 60}\n")

    for i, diagram in enumerate(diagrams):
        record_id = diagram["id"]
        url = diagram.get("url", "") or ""
        title = diagram.get("title", "Unknown")
        slug = slug_from_url(url) if url else ""

        # Artiq Cloudinary-de varsa kec
        if diagram.get("cloudinary_url"):
            skipped += 1
            continue

        print(f"[{i + 1}/{total}] {title}")

        if not url:
            print("    [SKIP] URL yoxdur.")
            failed += 1
            continue

        # If image_url already in database, use it
        image_url = diagram.get("image_url") or ""

        if not image_url:
            # Fetch detail page
            soup = fetch_page(session, url)
            if not soup:
                print("    [FAIL] Sehife yuklenilmedi.")
                failed += 1
                continue

            image_url = find_diagram_image(soup, url)
            if not image_url:
                print("    [NO_IMG] Sekil tapilmadi.")
                failed += 1
                continue

            # Write found image_url back to database
            update_image_url("cfc_diagrams", record_id, image_url)

        # Upload directly to Cloudinary from URL
        cloud_public_id = slug or f"diagram_{record_id}"
        cloud_url = upload_to_cloudinary(image_url, cloud_public_id, folder="origami/diagrams")
        if cloud_url:
            update_cloudinary_url("cfc_diagrams", record_id, cloud_url)
            uploaded += 1
        else:
            print(f"    [FAIL] Cloudinary upload ugursuz: {image_url}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  RESULTS:")
    print(f"  Uploaded   : {uploaded}")
    print(f"  Skipped    : {skipped} (already on Cloudinary)")
    print(f"  Failed     : {failed}")
    print(f"  Total      : {total}")
    print(f"{'=' * 60}")


def upload_book_images(limit=None):
    """PostgreSQL-deki kitablarin sekillerini Cloudinary-ye yukle."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        if limit:
            cur.execute("""
                SELECT id, title, url, image_url, cloudinary_url
                FROM cfc_books
                ORDER BY id
                LIMIT %s
            """, (limit,))
        else:
            cur.execute("""
                SELECT id, title, url, image_url, cloudinary_url
                FROM cfc_books
                ORDER BY id
            """)
        books = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    if not books:
        print("[INFO] Heç bir kitab tapilmadi.")
        return

    session = get_session()
    total = len(books)
    uploaded = 0
    skipped = 0
    failed = 0

    print(f"\n{'=' * 60}")
    print(f"  BOOK IMAGES -> Cloudinary ({total} books)")
    print(f"{'=' * 60}\n")

    for i, book in enumerate(books):
        record_id = book["id"]
        url = book.get("url", "") or ""
        title = book.get("title", "Unknown")

        # Artiq Cloudinary-de varsa kec
        if book.get("cloudinary_url"):
            skipped += 1
            continue

        print(f"[{i + 1}/{total}] {title}")

        if not url:
            failed += 1
            continue

        image_url = book.get("image_url") or ""

        if not image_url:
            soup = fetch_page(session, url)
            if not soup:
                failed += 1
                continue

            image_url = find_book_image(soup)
            if not image_url:
                print("    [NO_IMG] Sekil tapilmadi.")
                failed += 1
                continue

            update_image_url("cfc_books", record_id, image_url)

        slug = slug_from_url(url) or re.sub(r"[^\w]", "_", title)[:50]
        cloud_public_id = slug or f"book_{record_id}"
        cloud_url = upload_to_cloudinary(image_url, cloud_public_id, folder="origami/books")
        if cloud_url:
            update_cloudinary_url("cfc_books", record_id, cloud_url)
            uploaded += 1
        else:
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  BOOK IMAGE RESULTS:")
    print(f"  Uploaded   : {uploaded}")
    print(f"  Skipped    : {skipped}")
    print(f"  Failed     : {failed}")
    print(f"{'=' * 60}")


# ===================== MAIN =====================

def main():
    args = sys.argv[1:]

    limit = None
    if "--limit" in args:
        idx = args.index("--limit")
        if idx + 1 < len(args):
            try:
                limit = int(args[idx + 1])
            except ValueError:
                pass

    do_books = "--books" in args
    do_diagrams = "--books" not in args or "--diagrams" in args  # default: diagrams

    if "--all" in args:
        do_diagrams = True
        do_books = True

    print("=" * 60)
    print("  CFC ORIGAMI - CLOUDINARY UPLOADER")
    print(f"  Source : PostgreSQL ({POSTGRES_HOST})")
    print(f"  Cloud  : djig2omyz")
    if limit:
        print(f"  Limit  : {limit}")
    print("=" * 60)

    try:
        if do_diagrams:
            upload_diagram_images(limit)

        if do_books:
            upload_book_images(limit)

    except KeyboardInterrupt:
        print("\n\n[!] Interrupted.")


if __name__ == "__main__":
    main()
