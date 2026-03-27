"""
CFC Origami Scraper
====================
cfcorigami.com saytindan origami melumatlari (diaqramlar, kitablar, resurslar, calllar)
cekilib PostgreSQL bazasina yazilir.

Istifade:
    python cfc_scraping.py              # Butun kateqoriyalari scrape edir
    python cfc_scraping.py --diagrams   # Yalniz diaqramlari
    python cfc_scraping.py --books      # Yalniz kitablari
    python cfc_scraping.py --resources  # Yalniz resurslari
    python cfc_scraping.py --calls      # Yalniz call-lari
"""

import os
import psycopg2
import psycopg2.extras
import requests
from bs4 import BeautifulSoup
import json
import time
from dotenv import load_dotenv

load_dotenv()
import random
import re
import sys
from datetime import datetime
from urllib.parse import urljoin, urlparse
import cloudinary
import cloudinary.uploader


# ===================== CONFIG =====================

BASE_URL = "https://cfcorigami.com"

# Cloudinary settings
cloudinary.config(
    cloud_name="djig2omyz",
    api_key="521356887126377",
    api_secret="eRuxRbyTuK4osGgHF5Zgoj0WsNI",
    secure=True,
)

CFC_CLOUDINARY_ROOT = "origami/cfc_models"

# Scraping settings
REQUEST_DELAY_MIN = 1.5
REQUEST_DELAY_MAX = 3.5
MAX_RETRIES = 3
TIMEOUT = 15

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
]

# ===================== PostgreSQL CONFIG =====================

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "db-origami-nur-3364.c.aivencloud.com")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "19924"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "defaultdb")
POSTGRES_USER = os.getenv("POSTGRES_USER", "avnadmin")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "your_password_here")

# Shared connection - reuse single connection instead of opening new ones each time
_shared_conn = None


def get_connection():
    """Shared DB connection qaytarir. Qirilsa yeniden qosulur."""
    global _shared_conn
    try:
        if _shared_conn and not _shared_conn.closed:
            try:
                _shared_conn.cursor().execute("SELECT 1")
                return _shared_conn
            except Exception:
                try:
                    _shared_conn.close()
                except Exception:
                    pass
        _shared_conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            sslmode="require",
            connect_timeout=30,
        )
        return _shared_conn
    except Exception as e:
        print(f"    [DB_CONN] Connection failed: {e}")
        return None


def _reset_connection():
    """Reset connection so next get_connection() opens a fresh one."""
    global _shared_conn
    try:
        if _shared_conn:
            _shared_conn.close()
    except Exception:
        pass
    _shared_conn = None


def _db_execute_with_retry(operation_func, retries=3):
    """DB emeliyyatini retry mexanizmi ile icra edir."""
    for attempt in range(1, retries + 1):
        conn = get_connection()
        if not conn:
            if attempt < retries:
                time.sleep(2 * attempt)
                _reset_connection()
                continue
            return False
        cur = conn.cursor()
        try:
            result = operation_func(cur, conn)
            return result
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            _reset_connection()
            print(f"    [RETRY {attempt}/{retries}] Connection lost, reconnecting...")
            if attempt < retries:
                time.sleep(2 * attempt)
                continue
            print(f"    [DB_ERR] {e}")
            return False
        except Exception as e:
            try:
                conn.rollback()
            except Exception:
                pass
            print(f"    [DB_ERR] {e}")
            return False
        finally:
            try:
                cur.close()
            except Exception:
                pass
    return False


# ===================== DB HELPERS =====================

def get_existing_urls(table, column="url"):
    """Retrieve existing URLs from database."""
    def _op(cur, conn):
        cur.execute(f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL")
        return {row[0] for row in cur.fetchall()}
    result = _db_execute_with_retry(_op)
    return result if result else set()


def get_existing_titles(table):
    """Retrieve existing titles from database."""
    def _op(cur, conn):
        cur.execute(f"SELECT title FROM {table}")
        return {row[0] for row in cur.fetchall()}
    result = _db_execute_with_retry(_op)
    return result if result else set()


def insert_diagram(item):
    """Diaqrami cfc_diagrams cedveline yaz."""
    def _op(cur, conn):
        downloads_json = json.dumps(item.get("downloads", []), ensure_ascii=False) if item.get("downloads") else None
        cur.execute("""
            INSERT INTO cfc_diagrams (title, url, creator, language, description, difficulty, paper_size, category, image_url, downloads, scraped_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO NOTHING
        """, (
            item.get("title", ""),
            item.get("url"),
            item.get("creator", ""),
            item.get("language", ""),
            item.get("description", ""),
            item.get("difficulty", ""),
            item.get("paper_size", ""),
            item.get("category", ""),
            item.get("image_url", ""),
            downloads_json,
            datetime.now(),
        ))
        conn.commit()
        return cur.rowcount > 0
    return _db_execute_with_retry(_op)


def insert_book(item):
    """Kitabi cfc_books cedveline yaz."""
    def _op(cur, conn):
        cur.execute("""
            INSERT INTO cfc_books (title, url, author, published_date, image_url, scraped_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (title) DO NOTHING
        """, (
            item.get("title", ""),
            item.get("url", ""),
            item.get("author", ""),
            item.get("published_date", ""),
            item.get("image_url", ""),
            datetime.now(),
        ))
        conn.commit()
        return cur.rowcount > 0
    return _db_execute_with_retry(_op)


def insert_resource(item):
    """Resursu cfc_resources cedveline yaz."""
    def _op(cur, conn):
        links_json = json.dumps(item.get("resource_links", []), ensure_ascii=False) if item.get("resource_links") else None
        cur.execute("""
            INSERT INTO cfc_resources (title, url, updated_date, posted_on, summary, body, resource_links, scraped_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (url) DO NOTHING
        """, (
            item.get("title", ""),
            item.get("url"),
            item.get("updated_date", ""),
            item.get("posted_on", ""),
            item.get("summary", ""),
            item.get("body", ""),
            links_json,
            datetime.now(),
        ))
        conn.commit()
        return cur.rowcount > 0
    return _db_execute_with_retry(_op)


def insert_call(item):
    """Call-i cfc_calls cedveline yaz."""
    def _op(cur, conn):
        cur.execute("""
            INSERT INTO cfc_calls (title, url, posted_on, submission_deadline, summary, scraped_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (title) DO NOTHING
        """, (
            item.get("title", ""),
            item.get("url", ""),
            item.get("posted_on", ""),
            item.get("submission_deadline", ""),
            item.get("summary", ""),
            datetime.now(),
        ))
        conn.commit()
        return cur.rowcount > 0
    return _db_execute_with_retry(_op)


# ===================== HTTP SESSION =====================

class Scraper:
    """HTTP request-ler ve parsing ucun esas sinif."""

    def __init__(self):
        self.session = requests.Session()
        self._rotate_ua()
        self._warmup()

    def _warmup(self):
        """Session-u istilesdir: ana sehifeye request at ki cookie-ler alinsin."""
        try:
            self.session.get(BASE_URL, timeout=TIMEOUT)
            time.sleep(1)
        except Exception:
            pass

    def _rotate_ua(self):
        ua = random.choice(USER_AGENTS)
        self.session.headers.update({
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": BASE_URL + "/",
        })

    def _delay(self):
        time.sleep(random.uniform(REQUEST_DELAY_MIN, REQUEST_DELAY_MAX))

    def fetch(self, url, retries=MAX_RETRIES):
        """URL-den HTML al, retry ve error handling ile."""
        for attempt in range(1, retries + 1):
            try:
                self._rotate_ua()
                self._delay()
                resp = self.session.get(url, timeout=TIMEOUT)
                if resp.status_code == 200:
                    return BeautifulSoup(resp.text, "html.parser")
                elif resp.status_code == 406:
                    print(f"  [406] Blocked: {url} (attempt {attempt}/{retries})")
                    time.sleep(5 * attempt)
                elif resp.status_code == 404:
                    print(f"  [404] Not found: {url}")
                    return None
                else:
                    print(f"  [{resp.status_code}] {url} (attempt {attempt}/{retries})")
                    time.sleep(3 * attempt)
            except requests.exceptions.Timeout:
                print(f"  [TIMEOUT] {url} (attempt {attempt}/{retries})")
                time.sleep(5 * attempt)
            except requests.exceptions.ConnectionError:
                print(f"  [CONN_ERR] {url} (attempt {attempt}/{retries})")
                time.sleep(5 * attempt)
            except Exception as e:
                print(f"  [ERROR] {url}: {e}")
                time.sleep(3)

        print(f"  [FAILED] Could not fetch: {url}")
        return None


# ===================== SCRAPERS =====================

def scrape_diagrams(scraper):
    """
    Diagram Pool sehifelerini scrape et.
    URL: /diagram-pool  (pagination: ?page=0, ?page=1, ...)
    """
    print("\n" + "=" * 60)
    print("  DIAGRAMS - Scraping diagram pool...")
    print("=" * 60)

    existing_urls = get_existing_urls("cfc_diagrams")
    page = 0
    total_new = 0
    consecutive_empty = 0

    while True:
        if page == 0:
            url = f"{BASE_URL}/diagram-pool"
        else:
            url = f"{BASE_URL}/diagram-pool?page={page}"

        print(f"\n[Page {page + 1}] {url}")
        soup = scraper.fetch(url)
        if not soup:
            break

        items = _extract_diagram_list_items(soup)

        if not items:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                print("  No more diagram pages.")
                break
            page += 1
            continue

        consecutive_empty = 0
        page_new = 0

        for item in items:
            detail_url = item.get("url", "")
            if detail_url in existing_urls:
                continue

            # Fetch detail page
            if detail_url:
                detail_data = _scrape_diagram_detail(scraper, detail_url)
                if detail_data:
                    item.update(detail_data)

            if item.get("image_url"):
                item["image_url"] = _upload_cfc_image_to_cloudinary(
                    item.get("image_url"),
                    item.get("title", "diagram"),
                    "model",
                )

            if insert_diagram(item):
                page_new += 1
                total_new += 1
                print(f"    + {item.get('title', 'N/A')} | {item.get('creator', 'N/A')}")

        print(f"  Page {page + 1}: {page_new} new diagrams added.")
        page += 1

    print(f"\n  Total new diagrams: {total_new}")
    return total_new


def _extract_diagram_list_items(soup):
    """Diagram Pool list sehifesinden diaqram elementlerini cixar."""
    items = []
    root = _content_root(soup)

    rows = root.select(".views-row, .view-content .node, .node--type-diagram")
    if not rows:
        rows = root.select(".view-content > div, .view-content > li, .item-list li")

    if not rows:
        all_links = root.find_all("a", href=re.compile(r"^/diagram/|https?://[^\s]+/diagram/"))
        seen = set()
        for a in all_links:
            href = urljoin(BASE_URL, a.get("href", ""))
            if href not in seen and "/diagram-pool" not in href:
                seen.add(href)
                title = _clean_text(a.get_text(" ", strip=True) or _slug_to_title(href))
                items.append({
                    "title": title,
                    "url": href,
                    "creator": "",
                    "language": "",
                })
        return items

    for row in rows:
        title_el = row.select_one("h2, h3, h4, .views-field-title, .field--name-title")
        link_el = row.select_one("a[href*='/diagram/']") or (title_el.find("a") if title_el else None)

        if not link_el:
            continue

        href = urljoin(BASE_URL, link_el.get("href", ""))
        title = _clean_text(link_el.get_text(" ", strip=True) or (title_el.get_text(" ", strip=True) if title_el else ""))

        creator_el = row.select_one(".views-field-field-creator, .field--name-field-creator, .creator")
        creator = _clean_text(creator_el.get_text(" ", strip=True) if creator_el else "")

        lang_el = row.select_one(".views-field-field-language, .field--name-field-language, .language")
        language = _clean_text(lang_el.get_text(" ", strip=True) if lang_el else "")

        items.append({
            "title": title if title else _slug_to_title(href),
            "url": href,
            "creator": creator,
            "language": language,
        })

    return items


def _scrape_diagram_detail(scraper, url):
    """Diagram detail sehifesinden elave melumat al."""
    soup = scraper.fetch(url)
    if not soup:
        return {}

    data = {}

    # Title
    h1 = soup.find("h1")
    if h1:
        data["title"] = _clean_text(h1.get_text(" ", strip=True))

    # Body / description
    body = soup.select_one(".field--name-body, .node__content .field--type-text-with-summary, article .body")
    if body:
        data["description"] = body.get_text(strip=True)[:2000]

    # Metadata fields
    for field_name in ["creator", "language", "difficulty", "paper-size", "category"]:
        el = soup.select_one(f".field--name-field-{field_name}, .field--name-field-diagram-{field_name}")
        if el:
            label = el.select_one(".field__label")
            value = el.select_one(".field__item, .field__items")
            if value:
                key = field_name.replace("-", "_")
                data[key] = value.get_text(strip=True)

    # Image
    img = soup.select_one("article img, .field--name-field-image img, .node__content img")
    if img and img.get("src"):
        data["image_url"] = urljoin(BASE_URL, img["src"])

    # Download links
    downloads = []
    for a in soup.select("a[href$='.pdf'], a[href$='.PDF'], a[href*='download']"):
        downloads.append({
            "text": a.get_text(strip=True),
            "url": urljoin(BASE_URL, a["href"])
        })
    if downloads:
        data["downloads"] = downloads

    return data


def scrape_books(scraper):
    """
    Books sehifelerini scrape et.
    URL: /books  (pagination: ?page=0, ?page=1, ...)
    """
    print("\n" + "=" * 60)
    print("  BOOKS - Scraping books...")
    print("=" * 60)

    existing_titles = get_existing_titles("cfc_books")
    page = 0
    total_new = 0
    consecutive_empty = 0

    while True:
        if page == 0:
            url = f"{BASE_URL}/books"
        else:
            url = f"{BASE_URL}/books?page={page}"

        print(f"\n[Page {page + 1}] {url}")
        soup = scraper.fetch(url)
        if not soup:
            break

        items = _extract_book_items(soup)

        if not items:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                print("  No more book pages.")
                break
            page += 1
            continue

        consecutive_empty = 0
        page_new = 0

        for item in items:
            if item.get("title") in existing_titles:
                continue

            if item.get("image_url"):
                item["image_url"] = _upload_cfc_image_to_cloudinary(
                    item.get("image_url"),
                    item.get("title", "book"),
                    "book",
                )

            if insert_book(item):
                page_new += 1
                total_new += 1
                print(f"    + {item.get('title', 'N/A')} | {item.get('author', 'N/A')}")

        print(f"  Page {page + 1}: {page_new} new books added.")
        page += 1

    print(f"\n  Total new books: {total_new}")
    return total_new


def _extract_book_items(soup):
    """Books list sehifesinden kitab elementlerini cixar."""
    items = []
    root = _content_root(soup)

    # Correct CFC structure: a.card.book > strong.title
    cards = root.select("a.card.book")
    for card_link in cards:
        href = urljoin(BASE_URL, card_link.get("href", ""))
        title_el = card_link.select_one("strong.title")
        title = _clean_text(title_el.get_text(" ", strip=True) if title_el else "")
        if not title:
            continue

        author = ""
        em_el = card_link.select_one("em")
        if em_el:
            author = _clean_text(re.sub(r"^By:\s*", "", em_el.get_text(" ", strip=True), flags=re.IGNORECASE))

        pub_date = ""
        row_text = card_link.get_text(separator="\n")
        m = re.search(r"Published:\s*(.+?)(?:\n|$)", row_text)
        if m:
            pub_date = _clean_text(m.group(1))
        else:
            time_el = card_link.select_one("time")
            if time_el:
                pub_date = _clean_text(time_el.get_text(" ", strip=True))

        img = card_link.select_one("img")
        image_url = urljoin(BASE_URL, img["src"]) if img and img.get("src") else ""

        items.append({
            "title": title,
            "author": author,
            "published_date": pub_date,
            "url": href,
            "image_url": image_url,
        })

    return items


def scrape_resources(scraper):
    """
    Resources sehifelerini scrape et (list + detail).
    URL: /resources  (pagination: ?page=0, ?page=1, ...)
    """
    print("\n" + "=" * 60)
    print("  RESOURCES - Scraping resources...")
    print("=" * 60)

    existing_urls = get_existing_urls("cfc_resources")
    page = 0
    total_new = 0
    consecutive_empty = 0

    while True:
        if page == 0:
            url = f"{BASE_URL}/resources"
        else:
            url = f"{BASE_URL}/resources?page={page}"

        print(f"\n[Page {page + 1}] {url}")
        soup = scraper.fetch(url)
        if not soup:
            break

        items = _extract_resource_list_items(soup)

        if not items:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                print("  No more resource pages.")
                break
            page += 1
            continue

        consecutive_empty = 0
        page_new = 0

        for item in items:
            detail_url = item.get("url", "")
            if detail_url in existing_urls:
                continue

            if detail_url and detail_url.startswith("http"):
                detail_data = _scrape_resource_detail(scraper, detail_url)
                if detail_data:
                    item.update(detail_data)

            if insert_resource(item):
                page_new += 1
                total_new += 1
                print(f"    + {item.get('title', 'N/A')}")

        print(f"  Page {page + 1}: {page_new} new resources added.")
        page += 1

    print(f"\n  Total new resources: {total_new}")
    return total_new


def _extract_resource_list_items(soup):
    """Resources list sehifesinden resurs elementlerini cixar."""
    items = []
    root = _content_root(soup)

    # Correct CFC structure: a.list-item.resource > span.title
    list_items = root.select("a.list-item.resource")
    for res_link in list_items:
        href = urljoin(BASE_URL, res_link.get("href", ""))
        title_el = res_link.select_one("span.title")
        title = _clean_text(title_el.get_text(" ", strip=True) if title_el else "")
        if not title:
            continue

        updated = ""
        time_el = res_link.select_one("time")
        if time_el:
            updated = _clean_text(time_el.get_text(" ", strip=True))
        else:
            date_el = res_link.select_one("span.meta.date, span.date")
            if date_el:
                updated = _clean_text(date_el.get_text(" ", strip=True))
                updated = re.sub(r"^Updated:\s*", "", updated, flags=re.IGNORECASE)

        summary_el = res_link.select_one("span.copy")
        summary = _clean_text(summary_el.get_text(" ", strip=True))[:500] if summary_el else ""

        items.append({
            "title": title,
            "url": href,
            "updated_date": updated,
            "summary": summary,
        })

    if items:
        return items

    # Fallback: yalnız real resource detail linkləri (/resources/<slug>)
    seen = set()
    for a in root.find_all("a", href=True):
        href = urljoin(BASE_URL, a.get("href", ""))
        if "/resources/" not in href:
            continue
        if href.endswith("/resources") or "?page=" in href:
            continue
        title = _clean_text(a.get_text(" ", strip=True))
        if len(title) < 3:
            continue
        if href in seen:
            continue
        seen.add(href)
        items.append({
            "title": title,
            "url": href,
            "updated_date": "",
            "summary": "",
        })

    return items


def _scrape_resource_detail(scraper, url):
    """Resource detail sehifesinden body ve linkleri al."""
    soup = scraper.fetch(url)
    if not soup:
        return {}

    data = {}

    # Posted date
    posted_el = soup.select_one(".field--name-field-posted-on, .field--name-created, .submitted")
    if posted_el:
        data["posted_on"] = posted_el.get_text(strip=True)
        data["posted_on"] = re.sub(r"^Posted On:\s*", "", data["posted_on"], flags=re.IGNORECASE)
    else:
        text = soup.get_text()
        m = re.search(r"Posted On:\s*(.+?)(?:\n|$)", text)
        if m:
            data["posted_on"] = m.group(1).strip()

    # Body
    body = soup.select_one(".field--name-body, article .body, .node__content .field--type-text-with-summary")
    if body:
        data["body"] = body.get_text(strip=True)[:5000]

    # Resource links
    links_section = soup.select_one(".field--name-field-resource-links, .field--name-field-links")
    if links_section:
        resource_links = []
        for a in links_section.find_all("a", href=True):
            resource_links.append({
                "text": a.get_text(strip=True),
                "url": a["href"],
            })
        if resource_links:
            data["resource_links"] = resource_links
    else:
        if body:
            ext_links = []
            for a in body.find_all("a", href=True):
                href = a["href"]
                if href.startswith("http") and "cfcorigami.com" not in href:
                    ext_links.append({
                        "text": a.get_text(strip=True),
                        "url": href,
                    })
            if ext_links:
                data["resource_links"] = ext_links

    return data


def scrape_calls(scraper):
    """
    Calls for Diagrams sehifelerini scrape et.
    URL: /call-diagrams  (pagination: ?page=0, ?page=1, ...)
    """
    print("\n" + "=" * 60)
    print("  CALLS FOR DIAGRAMS - Scraping calls...")
    print("=" * 60)

    existing_titles = get_existing_titles("cfc_calls")
    page = 0
    total_new = 0
    consecutive_empty = 0

    while True:
        if page == 0:
            url = f"{BASE_URL}/call-diagrams"
        else:
            url = f"{BASE_URL}/call-diagrams?page={page}"

        print(f"\n[Page {page + 1}] {url}")
        soup = scraper.fetch(url)
        if not soup:
            break

        items = _extract_call_items(soup)

        if not items:
            consecutive_empty += 1
            if consecutive_empty >= 2:
                print("  No more call pages.")
                break
            page += 1
            continue

        consecutive_empty = 0
        page_new = 0

        for item in items:
            if item.get("title") in existing_titles:
                continue

            if insert_call(item):
                page_new += 1
                total_new += 1
                print(f"    + {item.get('title', 'N/A')} | Deadline: {item.get('submission_deadline', 'N/A')}")

        print(f"  Page {page + 1}: {page_new} new calls added.")
        page += 1

    print(f"\n  Total new calls: {total_new}")
    return total_new


def _extract_call_items(soup):
    """Calls for Diagrams list sehifesinden call elementlerini cixar."""
    items = []
    root = _content_root(soup)

    # CFC-də call list adları əsasən h3 > a[href*='/node/'] içindədir.
    heading_links = root.select("h3 a[href*='/node/']")
    for a in heading_links:
        title = _clean_text(a.get_text(" ", strip=True))
        if not title:
            continue

        href = urljoin(BASE_URL, a.get("href", ""))
        container = a.find_parent(["article", "div", "li", "section"]) or root
        container_text = container.get_text("\n", strip=True)

        posted = ""
        posted_m = re.search(r"Posted on\s*(.+?)(?:\n|$)", container_text, re.IGNORECASE)
        if posted_m:
            posted = _clean_text(posted_m.group(1))

        deadline = ""
        deadline_m = re.search(r"Submission Deadline\s*(.+?)(?:\n|$)", container_text, re.IGNORECASE)
        if deadline_m:
            deadline = _clean_text(deadline_m.group(1))

        items.append({
            "title": title,
            "url": href,
            "posted_on": posted,
            "submission_deadline": deadline,
            "summary": _clean_text(container_text)[:300],
        })

    if items:
        return items

    # Fallback: old generic row parsing
    rows = root.select(".views-row, .view-content .node, article, .node--type-call")
    for row in rows:
        title_el = row.select_one("h2, h3, h4, .views-field-title, .field--name-title")
        title = _clean_text(title_el.get_text(" ", strip=True) if title_el else "")
        if not title:
            continue

        link_el = row.select_one("a[href]")
        href = urljoin(BASE_URL, link_el["href"]) if link_el and link_el.get("href") else ""

        row_text = row.get_text("\n", strip=True)
        posted = ""
        m = re.search(r"Posted on:?\s*(.+?)(?:\n|$)", row_text, re.IGNORECASE)
        if m:
            posted = _clean_text(m.group(1))

        deadline = ""
        m = re.search(r"Submission Deadline:?\s*(.+?)(?:\n|$)", row_text, re.IGNORECASE)
        if m:
            deadline = _clean_text(m.group(1))

        items.append({
            "title": title,
            "url": href,
            "posted_on": posted,
            "submission_deadline": deadline,
            "summary": _clean_text(row_text)[:300],
        })

    return items


# ===================== HELPERS =====================

def _content_root(soup: BeautifulSoup):
    """Main content blokunu qaytar, sidebar/Additional Links səsini azalt."""
    return (
        soup.select_one("main")
        or soup.select_one("#main-content")
        or soup.select_one(".region-content")
        or soup.select_one(".layout-content")
        or soup
    )


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())

def _slug_to_title(url):
    """URL slug-dan oxunabilen bashliq yarat."""
    path = urlparse(url).path
    slug = path.rstrip("/").split("/")[-1]
    return slug.replace("-", " ").replace("_", " ").title()


def _safe_slug(text, max_len=70):
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", (text or "").strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    if not slug:
        slug = "item"
    return slug[:max_len]


def _is_probably_bad_image_url(url):
    u = (url or "").lower()
    blocked = (
        "banner", "logo", "icon", "sprite", "placeholder", "spacer",
        "tracking", "pixel", "ads", "advert", "nomodelimage"
    )
    return any(token in u for token in blocked)


def _upload_cfc_image_to_cloudinary(image_url, title, item_type):
    """Upload image under origami/cfc_models/{model|book} and return Cloudinary URL."""
    if not image_url or _is_probably_bad_image_url(image_url):
        return image_url

    subfolder = "book" if item_type == "book" else "model"
    folder = f"{CFC_CLOUDINARY_ROOT}/{subfolder}"
    public_id = f"cfc_{subfolder}_{_safe_slug(title)}_{int(time.time())}_{random.randint(1000, 9999)}"

    try:
        result = cloudinary.uploader.upload(
            image_url,
            folder=folder,
            public_id=public_id,
            overwrite=False,
            resource_type="image",
        )
        secure_url = result.get("secure_url")
        if secure_url:
            return secure_url
    except Exception as e:
        print(f"    [CLOUD_ERR] {str(e)[:100]}")

    return image_url


def get_table_counts():
    """Cedvellerdeki sayi qaytar."""
    conn = get_connection()
    cur = conn.cursor()
    counts = {}
    for tbl in ["cfc_diagrams", "cfc_books", "cfc_resources", "cfc_calls"]:
        cur.execute(f"SELECT COUNT(*) FROM {tbl}")
        counts[tbl] = cur.fetchone()[0]
    cur.close()
    conn.close()
    return counts


def print_summary():
    """Netice xulasesi cixar."""
    counts = get_table_counts()
    print("\n" + "=" * 60)
    print("  SCRAPING SUMMARY")
    print("=" * 60)
    print(f"  Diagrams : {counts['cfc_diagrams']}")
    print(f"  Books    : {counts['cfc_books']}")
    print(f"  Resources: {counts['cfc_resources']}")
    print(f"  Calls    : {counts['cfc_calls']}")
    print(f"  Database : PostgreSQL ({POSTGRES_HOST})")
    print("=" * 60)


# ===================== MAIN =====================

def main():
    args = sys.argv[1:]

    scrape_all = not any(a in args for a in ["--diagrams", "--books", "--resources", "--calls"])
    do_diagrams = scrape_all or "--diagrams" in args
    do_books = scrape_all or "--books" in args
    do_resources = scrape_all or "--resources" in args
    do_calls = scrape_all or "--calls" in args

    print("=" * 60)
    print("  CFC ORIGAMI SCRAPER -> PostgreSQL")
    print(f"  Source : {BASE_URL}")
    print(f"  Target : PostgreSQL ({POSTGRES_HOST})")
    cats = []
    if do_diagrams: cats.append("diagrams")
    if do_books: cats.append("books")
    if do_resources: cats.append("resources")
    if do_calls: cats.append("calls")
    print(f"  Categories: {', '.join(cats)}")
    print("=" * 60)

    scraper = Scraper()

    try:
        if do_diagrams:
            scrape_diagrams(scraper)

        if do_books:
            scrape_books(scraper)

        if do_resources:
            scrape_resources(scraper)

        if do_calls:
            scrape_calls(scraper)

    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user.")

    print_summary()


if __name__ == "__main__":
    main()
