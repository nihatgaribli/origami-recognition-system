"""
Origami Resource Center (ORC) scraper
=====================================
WordPress REST API + HTML fallback ile model siyahisini cixir,
PostgreSQL-de orc_models cedveline yazir ve sekilleri Cloudinary-ye yukleyir.

Istifade:
    python orc_scraping.py
    python orc_scraping.py --limit-pages 20
    python orc_scraping.py --no-cloudinary
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()
from typing import Any
from urllib.parse import parse_qs, urlparse

import cloudinary
import cloudinary.uploader
import psycopg2
import requests
from bs4 import BeautifulSoup, NavigableString, Tag


# ===================== CONFIG =====================

SITE_BASE = "https://origami-resource-center.com"
WP_API_BASE = f"{SITE_BASE}/wp-json/wp/v2"

REQUEST_DELAY_SEC = 2.0
MAX_RETRIES = 2
TIMEOUT = 60

HEADERS = {
    "User-Agent": "OrigamiDatasetBot/1.0 (research; contact=local-project)",
    "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.8",
    "Connection": "keep-alive",
}


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
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", "djig2omyz"),
    api_key=os.getenv("CLOUDINARY_API_KEY", "521356887126377"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET", "eRuxRbyTuK4osGgHF5Zgoj0WsNI"),
    secure=True,
)


def ensure_table(conn):
    """ORC ucun ayrica cedvel yaradir (yoxdursa)."""
    ddl = """
    CREATE TABLE IF NOT EXISTS orc_models (
        id BIGSERIAL PRIMARY KEY,
        model_name TEXT NOT NULL,
        model_name_base TEXT,
        variant_index INT,
        creator_raw TEXT,
        creator_expanded TEXT,
        creator_type TEXT DEFAULT 'unknown',
        category TEXT,
        subcategory TEXT,
        source_page_url TEXT NOT NULL,
        page_title TEXT,
        page_modified TIMESTAMPTZ,
        sitemap_lastmod TIMESTAMPTZ,
        diagram_url TEXT,
        diagram_type TEXT,
        diagram_is_hosted_on_orc BOOLEAN DEFAULT FALSE,
        diagram_is_archived BOOLEAN DEFAULT FALSE,
        is_dollar_bill BOOLEAN DEFAULT FALSE,
        image_url TEXT,
        cloudinary_url TEXT,
        scraped_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE INDEX IF NOT EXISTS idx_orc_models_category ON orc_models (category);
    CREATE INDEX IF NOT EXISTS idx_orc_models_creator ON orc_models (creator_raw);
    CREATE INDEX IF NOT EXISTS idx_orc_models_diagram_type ON orc_models (diagram_type);
    CREATE UNIQUE INDEX IF NOT EXISTS uq_orc_models_src_name_diagram
        ON orc_models (source_page_url, model_name, COALESCE(diagram_url, ''));
    """
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()


def get_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def request_with_retry(session: requests.Session, url: str, params: dict[str, Any] | None = None) -> requests.Response | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=TIMEOUT)
            if resp.status_code in (200, 201):
                return resp
            if resp.status_code == 404:
                return None
            print(f"    [HTTP {resp.status_code}] {url} (attempt {attempt})")
        except Exception as exc:
            print(f"    [REQ_ERR attempt {attempt}] {type(exc).__name__}: {exc}")

        if attempt < MAX_RETRIES:
            # Exponential backoff but capped
            delay = min(5.0 * (2 ** (attempt - 1)), 20.0)
            time.sleep(delay)

    return None


def parse_iso_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        fixed = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(fixed)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def fetch_page_sitemap(session: requests.Session) -> dict[str, datetime | None]:
    """page-sitemap.xml-den URL -> lastmod map qaytarir."""
    import xml.etree.ElementTree as ET

    url = f"{SITE_BASE}/page-sitemap.xml"
    resp = request_with_retry(session, url)
    if not resp:
        return {}

    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    root = ET.fromstring(resp.text)

    result: dict[str, datetime | None] = {}
    for node in root.findall("sm:url", ns):
        loc = node.find("sm:loc", ns)
        lastmod = node.find("sm:lastmod", ns)
        if loc is None or not loc.text:
            continue
        result[loc.text.strip()] = parse_iso_ts(lastmod.text.strip() if lastmod is not None and lastmod.text else None)
    return result


def fetch_all_wp_pages(session: requests.Session, limit_pages: int | None = None) -> list[dict[str, Any]]:
    """WP REST API pages endpoint-den butun sehifeleri oxuyur."""
    all_pages: list[dict[str, Any]] = []
    page_num = 1

    while True:
        if limit_pages is not None and len(all_pages) >= limit_pages:
            break

        resp = request_with_retry(
            session,
            f"{WP_API_BASE}/pages",
            params={
                "per_page": 50,
                "page": page_num,
                "_fields": "id,slug,title,content,link,modified",
            },
        )

        if not resp:
            break

        if resp.status_code == 400:
            break

        batch = resp.json()
        if not isinstance(batch, list) or not batch:
            break

        all_pages.extend(batch)
        total_pages = int(resp.headers.get("X-WP-TotalPages", 1))
        print(f"[WP] page {page_num}/{total_pages}, fetched {len(batch)} pages")

        if page_num >= total_pages:
            break
        page_num += 1
        time.sleep(REQUEST_DELAY_SEC)

    if limit_pages is not None:
        return all_pages[:limit_pages]
    return all_pages


def classify_diagram_url(url: str | None) -> tuple[str, bool, bool]:
    if not url:
        return "none", False, False

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()

    if "youtube.com" in host or "youtu.be" in host:
        return "youtube", False, False
    if path.endswith(".pdf"):
        hosted = "origami-resource-center.com" in host
        return "pdf", hosted, False
    if path.endswith((".gif", ".jpg", ".jpeg", ".png", ".webp")):
        hosted = "origami-resource-center.com" in host
        return "image_diagram", hosted, False
    if "web.archive.org" in host:
        return "archived_html", False, True
    return "html", False, False


def extract_creator(li_tag: Tag, a_tag: Tag) -> str | None:
    """li icinde <a>-dan sonra gelen parantezdeki creator adini cixarir."""
    text_after = a_tag.next_sibling
    if isinstance(text_after, NavigableString):
        match = re.search(r"\(([^)]+)\)", str(text_after))
        if match:
            return match.group(1).strip()

    # Backup: extract the '(...)' from end of all li text
    all_text = li_tag.get_text(" ", strip=True)
    match = re.search(r"\(([^)]+)\)\s*$", all_text)
    if match:
        return match.group(1).strip()

    return None


def normalize_creator(creator_raw: str | None) -> tuple[str | None, str]:
    if not creator_raw:
        return None, "unknown"

    lookup = {
        "AF Barbour": "Anita F. Barbour",
        "SS Cucek": "Stephan S. Cucek",
        "P Jackson": "Paul Jackson",
        "R Neale": "Robert Neale",
        "M Sonobe": "Mitsunobu Sonobe",
        "T Fuse": "Tomoko Fuse",
        "A Yoshizawa": "Akira Yoshizawa",
        "J Montroll": "John Montroll",
        "R Lang": "Robert J. Lang",
        "ORC": "Origami Resource Center",
        "Traditional": None,
    }

    creator_raw = creator_raw.strip()
    if creator_raw == "Traditional":
        return None, "traditional"
    if creator_raw == "ORC":
        return "Origami Resource Center", "orc"
    expanded = lookup.get(creator_raw, creator_raw)
    return expanded, "individual"


def get_nearest_heading(li_tag: Tag) -> str | None:
    for sibling in li_tag.find_previous_siblings():
        if isinstance(sibling, Tag) and sibling.name in ("h2", "h3", "h4"):
            text = sibling.get_text(" ", strip=True)
            if text:
                return text
    return None


def is_valid_image_url(url: str | None) -> bool:
    if not url:
        return False

    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        return False

    host = parsed.netloc.lower()
    path = parsed.path.lower()

    # Reklam/tracking domenlerini sekil kimi goturme
    blocked_hosts = (
        "amazon-adsystem.com",
        "doubleclick.net",
        "googleadservices.com",
        "googlesyndication.com",
    )
    if any(bh in host for bh in blocked_hosts):
        return False

    # Etibarlı image URL pattern-leri
    image_ext = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".avif")
    if path.endswith(image_ext):
        return True
    if "wp-content/uploads/" in path:
        return True

    return False


def get_nearest_image(li_tag: Tag) -> str | None:
    """Model entry-e en yaxin onceki sekil URL-ni tapir (category panel sekilleri)."""
    img = li_tag.find_previous("img")
    if not img:
        return None
    src = img.get("data-src") or img.get("src")
    if not src:
        return None
    src = src.strip()
    if src.startswith("//"):
        src = "https:" + src
    elif src.startswith("/"):
        src = SITE_BASE + src

    if not is_valid_image_url(src):
        return None

    return src


def should_skip_li(li_tag: Tag, link_text: str, href: str) -> bool:
    if not link_text or len(link_text) < 2:
        return True

    low_text = link_text.lower().strip()
    low_href = href.lower().strip()

    # Footer/menu/navigation URL pattern-leri
    footer_patterns = (
        "/wp-admin", "/wp-login", "#respond", "mailto:",
        "/contact-us/", "/report-broken-links/", "/buy-origami",
        "/privacy", "/terms", "/disclaimer", "/cookie"
    )
    if any(token in low_href for token in footer_patterns):
        return True

    # Spam/navigation text patterns
    skip_phrases = (
        "click here", "read more", "privacy policy", "cookie",
        "back to ", "reporting broken links", "buy ",
        "easy origami books", "site map", "view slide details"
    )
    if any(phrase in low_text for phrase in skip_phrases):
        return True

    # Exact nav words
    nav_words = {
        "home", "about", "contact", "contact us", "privacy",
        "terms", "disclaimer", "main menu", "sitemap", "search",
        "previous", "next", "all", "animals", "birds", "boxes",
    }
    if low_text in nav_words:
        return True

    if re.fullmatch(r"[0-9]+", low_text):
        return True

    # "book review" â†’ skip (booklar deyil origami)
    if low_text.startswith("book review") or low_text == "book review":
        return True

    # Ana sayt URL-u
    if low_href in (SITE_BASE, SITE_BASE + "/", "https://origami-resource-center.com", "https://origami-resource-center.com/"):
        return True

    return False


def normalize_href(href: str) -> str:
    href = (href or "").strip()
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    elif href.startswith("/"):
        href = SITE_BASE + href
    return href


def extract_model_name(li_tag: Tag, a_tag: Tag) -> str:
    """
    ORC list item-den model adini cixarir.

    Eyni sətirdə [here], [1], [2] tipli linklər ola bilər; bu hallarda
    model adı anchor mətni deyil, anchor-dan əvvəlki mətn olur.
    """
    anchor_text = html.unescape(a_tag.get_text(" ", strip=True)).strip()
    low_anchor = anchor_text.lower()

    generic_anchor = {
        "here", "or", "and", "together", "pg", "page", "details",
    }

    # 1) Anchor-dan əvvəl gələn mətn (məs: "Bird of Paradise: [here]")
    prefix_chunks = []
    for node in li_tag.contents:
        if node is a_tag:
            break
        if isinstance(node, NavigableString):
            txt = str(node).strip()
            if txt:
                prefix_chunks.append(txt)

    prefix = html.unescape(" ".join(prefix_chunks)).strip()
    prefix = re.sub(r"\s+", " ", prefix).strip(" :-,;/")

    # 2) Generic anchor olduqda prefix-i üstün tut.
    if prefix and (low_anchor in generic_anchor or re.fullmatch(r"\d+", low_anchor)):
        return prefix

    # 3) If prefix exists and ends with ':', model name is the prefix.
    if prefix and prefix.endswith(":"):
        return prefix.rstrip(":").strip()

    # 4) Standart halda anchor text model adıdır.
    candidate = anchor_text

    # 5) Anchor text yenə də generic çıxsa, li text-dən çıxar.
    if not candidate or low_anchor in generic_anchor or re.fullmatch(r"\d+", low_anchor):
        full_text = html.unescape(li_tag.get_text(" ", strip=True))
        # Son creator hissəsini çıxar: (...)
        full_text = re.sub(r"\s*\([^)]+\)\s*$", "", full_text)
        if ":" in full_text:
            candidate = full_text.split(":", 1)[0].strip()
        else:
            candidate = full_text.strip()

    candidate = re.sub(r"\s+", " ", candidate).strip(" :-,;/")
    return candidate


def parse_variant(model_name: str) -> tuple[str, int | None]:
    m = re.search(r"\s+(\d+)$", model_name)
    if not m:
        return model_name, None
    idx = int(m.group(1))
    base = re.sub(r"\s+\d+$", "", model_name).strip()
    return base, idx


def normalize_slug_to_category(slug: str) -> str:
    return slug.strip().lower().replace("-", "_")


def extract_models_from_content(
    wp_page: dict[str, Any],
    sitemap_lastmod: datetime | None,
) -> list[dict[str, Any]]:
    """WP content.rendered icinden model list elementlerini parse edir."""
    slug = (wp_page.get("slug") or "").strip()
    page_url = (wp_page.get("link") or "").strip()
    page_title = html.unescape((wp_page.get("title") or {}).get("rendered", "") or "")
    page_modified = parse_iso_ts(wp_page.get("modified"))

    raw_html = (wp_page.get("content") or {}).get("rendered", "") or ""
    if not raw_html:
        return []

    soup = BeautifulSoup(raw_html, "lxml")
    models: list[dict[str, Any]] = []

    for li in soup.find_all("li"):
        a = li.find("a", href=True)
        if not a:
            continue

        href = normalize_href(a.get("href", ""))
        model_name = extract_model_name(li, a)

        if should_skip_li(li, model_name, href):
            continue

        creator_raw = extract_creator(li, a)
        creator_expanded, creator_type = normalize_creator(creator_raw)
        subcategory = get_nearest_heading(li)

        is_dollar_bill = model_name.startswith("$")
        if is_dollar_bill:
            model_name = model_name.lstrip("$").strip()

        base_name, variant_index = parse_variant(model_name)
        diagram_type, is_hosted_on_orc, is_archived = classify_diagram_url(href)
        image_url = get_nearest_image(li)

        models.append(
            {
                "model_name": model_name,
                "model_name_base": base_name,
                "variant_index": variant_index,
                "creator_raw": creator_raw,
                "creator_expanded": creator_expanded,
                "creator_type": creator_type,
                "category": normalize_slug_to_category(slug),
                "subcategory": subcategory,
                "source_page_url": page_url,
                "page_title": page_title,
                "page_modified": page_modified,
                "sitemap_lastmod": sitemap_lastmod,
                "diagram_url": href,
                "diagram_type": diagram_type,
                "diagram_is_hosted_on_orc": is_hosted_on_orc,
                "diagram_is_archived": is_archived,
                "is_dollar_bill": is_dollar_bill,
                "image_url": image_url,
                "cloudinary_url": None,
                "scraped_at": datetime.now(timezone.utc),
            }
        )

    # Dublikatlari eyni sehife daxilinde azalt
    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in models:
        key = (
            row["source_page_url"],
            row["model_name"].strip().lower(),
            (row.get("diagram_url") or "").strip().lower(),
        )
        if key not in dedup:
            dedup[key] = row

    return list(dedup.values())


def extract_models_from_html_page(
    session: requests.Session,
    page_url: str,
    sitemap_lastmod: datetime | None,
) -> list[dict[str, Any]]:
    """REST API olmadigi halda birbasa HTML sehifeden modelleri cixarma fallback-i."""
    resp = request_with_retry(session, page_url)
    if not resp:
        return []

    soup = BeautifulSoup(resp.text, "lxml")
    content = soup.select_one("div.entry-content") or soup.select_one("div#content")
    # Yeni ORC layout-larinda entry-content olmaya biler; bu halda main content area-den parse edirik.
    search_root: Tag = content if content else soup

    title_el = soup.select_one("h1.entry-title")
    page_title = html.unescape(title_el.get_text(" ", strip=True)) if title_el else ""
    slug = page_url.rstrip("/").split("/")[-1]

    modified_meta = soup.find("meta", attrs={"property": "article:modified_time"})
    page_modified = parse_iso_ts(modified_meta.get("content") if modified_meta else None)

    models: list[dict[str, Any]] = []
    for li in search_root.find_all("li"):
        a = li.find("a", href=True)
        if not a:
            continue

        href = normalize_href(a.get("href", ""))
        model_name = extract_model_name(li, a)
        if should_skip_li(li, model_name, href):
            continue

        creator_raw = extract_creator(li, a)
        creator_expanded, creator_type = normalize_creator(creator_raw)
        subcategory = get_nearest_heading(li)

        is_dollar_bill = model_name.startswith("$")
        if is_dollar_bill:
            model_name = model_name.lstrip("$").strip()

        base_name, variant_index = parse_variant(model_name)
        diagram_type, is_hosted_on_orc, is_archived = classify_diagram_url(href)
        image_url = get_nearest_image(li)

        models.append(
            {
                "model_name": model_name,
                "model_name_base": base_name,
                "variant_index": variant_index,
                "creator_raw": creator_raw,
                "creator_expanded": creator_expanded,
                "creator_type": creator_type,
                "category": normalize_slug_to_category(slug),
                "subcategory": subcategory,
                "source_page_url": page_url,
                "page_title": page_title,
                "page_modified": page_modified,
                "sitemap_lastmod": sitemap_lastmod,
                "diagram_url": href,
                "diagram_type": diagram_type,
                "diagram_is_hosted_on_orc": is_hosted_on_orc,
                "diagram_is_archived": is_archived,
                "is_dollar_bill": is_dollar_bill,
                "image_url": image_url,
                "cloudinary_url": None,
                "scraped_at": datetime.now(timezone.utc),
            }
        )

    dedup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in models:
        key = (
            row["source_page_url"],
            row["model_name"].strip().lower(),
            (row.get("diagram_url") or "").strip().lower(),
        )
        if key not in dedup:
            dedup[key] = row
    return list(dedup.values())


def cloudinary_public_id(model_name: str, row_id_hint: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", model_name.strip().lower())
    safe = re.sub(r"_+", "_", safe).strip("_")
    if not safe:
        safe = "model"
    return f"{safe}_{row_id_hint}"


def upload_image_to_cloudinary(image_url: str, model_name: str) -> str | None:
    if not image_url:
        return None
    if not is_valid_image_url(image_url):
        print(f"    [CLOUD_SKIP] Non-image URL: {image_url}")
        return None
    try:
        hint = str(int(time.time() * 1000))[-8:]
        public_id = cloudinary_public_id(model_name, hint)
        result = cloudinary.uploader.upload(
            image_url,
            public_id=public_id,
            folder="origami/orc_models",
            overwrite=False,
            resource_type="image",
        )
        secure_url = result.get("secure_url")
        if secure_url:
            print(f"    [CLOUD] {secure_url}")
        return secure_url
    except Exception as exc:
        print(f"    [CLOUD_ERR] {image_url}: {exc}")
        return None


def insert_model(conn, row: dict[str, Any]) -> bool:
    sql = """
    INSERT INTO orc_models (
        model_name,
        model_name_base,
        variant_index,
        creator_raw,
        creator_expanded,
        creator_type,
        category,
        subcategory,
        source_page_url,
        page_title,
        page_modified,
        sitemap_lastmod,
        diagram_url,
        diagram_type,
        diagram_is_hosted_on_orc,
        diagram_is_archived,
        is_dollar_bill,
        image_url,
        cloudinary_url,
        scraped_at
    )
    VALUES (
        %(model_name)s,
        %(model_name_base)s,
        %(variant_index)s,
        %(creator_raw)s,
        %(creator_expanded)s,
        %(creator_type)s,
        %(category)s,
        %(subcategory)s,
        %(source_page_url)s,
        %(page_title)s,
        %(page_modified)s,
        %(sitemap_lastmod)s,
        %(diagram_url)s,
        %(diagram_type)s,
        %(diagram_is_hosted_on_orc)s,
        %(diagram_is_archived)s,
        %(is_dollar_bill)s,
        %(image_url)s,
        %(cloudinary_url)s,
        %(scraped_at)s
    )
    ON CONFLICT (source_page_url, model_name, COALESCE(diagram_url, '')) DO NOTHING;
    """
    with conn.cursor() as cur:
        cur.execute(sql, row)
        return cur.rowcount > 0


def run_scrape(
    limit_pages: int | None = None,
    use_cloudinary: bool = True,
    init_only: bool = False,
    html_only: bool = False,
    start_page: int = 1,
):
    session = get_session()
    conn = get_connection()
    ensure_table(conn)

    if init_only:
        conn.close()
        print("[INIT] orc_models table created/verified.")
        return

    sitemap_map = fetch_page_sitemap(session)
    print(f"[SITEMAP] URLs with lastmod: {len(sitemap_map)}")

    wp_pages: list[dict[str, Any]] = []
    if not html_only:
        wp_pages = fetch_all_wp_pages(session, limit_pages=limit_pages)
    print(f"[WP] total pages fetched: {len(wp_pages)}")

    saved = 0
    skipped = 0
    page_with_models = 0

    try:
        if wp_pages:
            for idx, page in enumerate(wp_pages, start=1):
                page_url = (page.get("link") or "").strip()
                sitemap_lastmod = sitemap_map.get(page_url)
                rows = extract_models_from_content(page, sitemap_lastmod)

                if not rows:
                    continue

                page_with_models += 1
                print(f"[{idx}/{len(wp_pages)}] {page.get('slug')} -> {len(rows)} models")

                for row in rows:
                    if use_cloudinary and row.get("image_url"):
                        cloud_url = upload_image_to_cloudinary(row["image_url"], row["model_name"])
                        row["cloudinary_url"] = cloud_url

                    ok = insert_model(conn, row)
                    if ok:
                        saved += 1
                    else:
                        skipped += 1

                conn.commit()
                time.sleep(REQUEST_DELAY_SEC)
        else:
            print("[FALLBACK] REST API response alinmadi, sitemap + HTML parsing ile davam edir.")
            urls = list(sitemap_map.keys())
            start_idx = max(start_page - 1, 0)
            if start_idx > 0:
                urls = urls[start_idx:]
            if limit_pages is not None:
                urls = urls[:limit_pages]

            for idx, url in enumerate(urls, start=start_idx + 1):
                rows = extract_models_from_html_page(session, url, sitemap_map.get(url))
                if not rows:
                    continue

                page_with_models += 1
                print(f"[{idx}/{len(urls)}] {url} -> {len(rows)} models")

                for row in rows:
                    if use_cloudinary and row.get("image_url"):
                        cloud_url = upload_image_to_cloudinary(row["image_url"], row["model_name"])
                        row["cloudinary_url"] = cloud_url

                    ok = insert_model(conn, row)
                    if ok:
                        saved += 1
                    else:
                        skipped += 1

                conn.commit()
                time.sleep(REQUEST_DELAY_SEC)

    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
    finally:
        conn.close()

    print("\n" + "=" * 60)
    print("ORC SCRAPE DONE")
    print(f"Pages with models: {page_with_models}")
    print(f"Saved rows       : {saved}")
    print(f"Skipped (dup)    : {skipped}")
    print("Target table     : orc_models")
    print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORC scraper -> PostgreSQL + Cloudinary")
    parser.add_argument("--limit-pages", type=int, default=None, help="Test ucun yalniz ilk N page")
    parser.add_argument("--no-cloudinary", action="store_true", help="Sekilleri Cloudinary-ye yukleme")
    parser.add_argument("--init-only", action="store_true", help="Yalniz cedveli yarat/yoxla")
    parser.add_argument("--html-only", action="store_true", help="WP REST API-ni kecib sitemap+HTML ile scrape et")
    parser.add_argument("--start-page", type=int, default=1, help="Fallback sitemap emeliyyatini bu page index-den baslat (1-based)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_scrape(
        limit_pages=args.limit_pages,
        use_cloudinary=not args.no_cloudinary,
        init_only=args.init_only,
        html_only=args.html_only,
        start_page=args.start_page,
    )

