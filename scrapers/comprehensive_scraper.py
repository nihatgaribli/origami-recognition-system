import psycopg2
import uuid
import requests
from bs4 import BeautifulSoup
import re
import time
import random
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
import cloudinary
import cloudinary.uploader

# ===================== CLOUDINARY CONFIG =====================
import os
from dotenv import load_dotenv

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME", "djig2omyz"),
    api_key=os.getenv("CLOUDINARY_API_KEY", "521356887126377"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET", "eRuxRbyTuK4osGgHF5Zgoj0WsNI"),
    secure=True
)


POSTGRES_HOST = "db-origami-nur-3364.c.aivencloud.com"
POSTGRES_PORT = 19924
POSTGRES_DB = "defaultdb"
POSTGRES_USER = "avnadmin"
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "your_password_here")

BASE_URL = "https://oriwiki.com"

# Global shared connection
_shared_conn = None


def get_connection():
    """Get or create a shared DB connection (reuses single connection)."""
    global _shared_conn
    try:
        if _shared_conn and not _shared_conn.closed:
            # Test if connection is still alive
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
            sslmode="require"
        )
        return _shared_conn
    except Exception as e:
        print(f"PostgreSQL connection failed: {e}")
        return None


def save_model_comprehensive(model_data, retries=3):
    """
    Save model with all related data to appropriate tables.
    Retry mexanizmi ile connection itkileri idarə edir.
    """
    for attempt in range(1, retries + 1):
        pg_conn = get_connection()
        if not pg_conn:
            if attempt < retries:
                time.sleep(2 * attempt)
                continue
            return False
        
        cursor = pg_conn.cursor()
        try:
            result = _save_model_inner(cursor, pg_conn, model_data)
            return result
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            # Connection lost - reset and retry
            global _shared_conn
            _shared_conn = None
            print(f"    [RETRY {attempt}/{retries}] Connection lost, reconnecting...")
            if attempt < retries:
                time.sleep(2 * attempt)
                continue
            print(f" Error: {str(e)[:100]}")
            return False
        except Exception as e:
            error_msg = str(e)[:100]
            print(f" Error: {error_msg}")
            try:
                pg_conn.rollback()
            except Exception:
                pass
            return False
        finally:
            try:
                cursor.close()
            except Exception:
                pass
    return False


def _save_model_inner(cursor, pg_conn, model_data):
    """Model-i DB-yə yazma məntiqi."""
    try:
    # 1. Insert/Get Creator
        creator_name = model_data.get('creator', 'Unknown')
        creator_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"creator.{creator_name}"))
        
        cursor.execute("""
            INSERT INTO creators (creator_id, name_original, name_normalized)
            VALUES (%s, %s, %s)
            ON CONFLICT (creator_id) DO NOTHING;
        """, (
            creator_uuid,
            creator_name,
            creator_name.lower().replace(" ", "_")
        ))
        
        # 2. Insert Model
        model_uuid = str(uuid.uuid4())
        year_val = None
        if model_data.get('year') and model_data.get('year') != 'Unknown':
            try:
                year_val = int(model_data['year'])
            except:
                pass
        
        difficulty_val = model_data.get('difficulty_avg')
        if difficulty_val and difficulty_val != "Unknown":
            try:
                difficulty_val = int(float(difficulty_val))
                # Clamp to 1-8 range for database constraint
                if difficulty_val < 1:
                    difficulty_val = 1
                elif difficulty_val > 5:
                    difficulty_val = 5
            except:
                difficulty_val = None
        else:
            difficulty_val = None
        
        # Check if already exists
        source_url = model_data.get('source_url')
        cursor.execute("SELECT model_id FROM models WHERE source_url = %s;", (source_url,))
        existing = cursor.fetchone()
        
        if existing:
            # Skip silently - already in database
            return True
        
        cursor.execute("""
            INSERT INTO models (
                model_id, creator_id, model_name_original, model_name_normalized,
                year_created, paper_shape, pieces, uses_cutting, uses_glue,
                difficulty, source_url
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """, (
            model_uuid,
            creator_uuid,
            model_data.get('name', 'Unknown'),
            model_data.get('name', '').lower().replace(" ", "_"),
            year_val,
            model_data.get('paper'),
            model_data.get('pieces'),
            model_data.get('cuts', False),
            model_data.get('glue', False),
            difficulty_val,
            source_url
        ))
        
        # 3. Insert Image if URL exists -> upload to Cloudinary
        image_url = model_data.get('image_url')
        if image_url and 'NoModelImage' not in image_url and 'odbbanner' not in image_url and image_url != "No Image":
            image_uuid = str(uuid.uuid4())
            # Upload to Cloudinary
            cloud_url = None
            try:
                model_slug = re.sub(r'[^a-zA-Z0-9_-]', '_', model_data.get('name', 'unknown'))[:80]
                cloud_public_id = f"{model_slug}_{image_uuid[:8]}"
                result = cloudinary.uploader.upload(
                    image_url,
                    public_id=cloud_public_id,
                    folder="origami/models",
                    overwrite=False,
                    resource_type="image"
                )
                cloud_url = result.get("secure_url", "")
                if cloud_url:
                    print(f"    [CLOUD] {cloud_url}")
            except Exception as ce:
                print(f"    [CLOUD_ERR] {str(ce)[:80]}")
                cloud_url = None
            try:
                cursor.execute("""
                    INSERT INTO images (image_id, model_id, url, cloudinary_url, is_primary)
                    VALUES (%s, %s, %s, %s, %s);
                """, (image_uuid, model_uuid, image_url, cloud_url, True))
            except Exception as ex:
                pass  # Image might already exist, ignore
        
        pg_conn.commit()
        print(f" Saved: {model_data.get('name', 'Unknown')} (creator: {creator_name})")
        return True
        
    except Exception as e:
        pg_conn.rollback()
        raise


def get_session():
    """Get requests session."""
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
    return session


def _img_text(tag, attr):
    try:
        return str(tag.get(attr, '')).strip().lower()
    except Exception:
        return ''


def _is_banner_like_image(img_tag, src):
    """Reject common non-model images such as banners, logos, and placeholders."""
    haystack = " ".join([
        src.lower(),
        _img_text(img_tag, 'alt'),
        _img_text(img_tag, 'title'),
        _img_text(img_tag, 'id'),
        " ".join(img_tag.get('class', [])).lower() if img_tag.get('class') else ''
    ])

    blocked_tokens = (
        'banner', 'odbbanner', 'logo', 'icon', 'sprite', 'button', 'menu', 'nav',
        'header', 'footer', 'ad_', 'ads', 'advert', 'tracking', 'pixel',
        'placeholder', 'spacer', 'nomodelimage'
    )
    return any(token in haystack for token in blocked_tokens)


def _is_too_small(img_tag):
    """Skip tiny decorative images when dimensions are available."""
    try:
        w = int(str(img_tag.get('width', '')).strip()) if img_tag.get('width') else None
        h = int(str(img_tag.get('height', '')).strip()) if img_tag.get('height') else None
        if w is not None and w < 120:
            return True
        if h is not None and h < 120:
            return True
    except Exception:
        pass
    return False


def _pick_best_model_image(soup, model_name):
    """
    Oriwiki structure: model image is in <td colspan="4" class="align-center width-80">
    If that TD contains NoModelImage.jpg, no image is available.
    Otherwise, the first img in that TD is the model image.
    """
    # First, try Oriwiki-specific structure: <td colspan="4" class="...">
    main_td = soup.find('td', attrs={'colspan': '4'})
    if not main_td:
        # Fallback: find td with both classes
        all_tds = soup.find_all('td')
        for td in all_tds:
            cls = str(td.get('class', ''))
            if 'align-center' in cls and 'width-80' in cls:
                main_td = td
                break
    
    if main_td:
        img = main_td.find('img', src=True)
        if img:
            src = str(img.get('src', '')).strip()
            if src and 'nomodelimage' not in src.lower():
                abs_src = urljoin(BASE_URL + '/', src)
                return abs_src
            elif src and 'nomodelimage' in src.lower():
                # Explicitly no image available on this model
                return None
    
    # Fallback: generic search with strong filtering (for edge cases)
    if not model_name:
        model_name = ''
    model_name_l = model_name.strip().lower()

    candidates = []
    for img_tag in soup.find_all('img', src=True):
        src = str(img_tag.get('src', '')).strip()
        if not src or src.startswith('data:'):
            continue
        if _is_banner_like_image(img_tag, src):
            continue
        if _is_too_small(img_tag):
            continue

        score = 0
        alt_l = _img_text(img_tag, 'alt')
        title_l = _img_text(img_tag, 'title')
        src_l = src.lower()

        if model_name_l and model_name_l in alt_l:
            score += 6
        if model_name_l and model_name_l in title_l:
            score += 4
        if '/models/' in src_l or 'showmodel' in src_l:
            score += 3
        if 'origami' in src_l:
            score += 1

        abs_src = urljoin(BASE_URL + '/', src)
        candidates.append((score, abs_src))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_src = candidates[0]
    if best_score < 0:
        return None
    return best_src


def scrape_model_details(model_url, session):
    """Scrape model details from page."""
    try:
        response = session.get(model_url, timeout=10)
        if response.status_code != 200:
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        data = {}
        
        # Name
        heading_span = soup.find('span', class_='Heading')
        data['name'] = heading_span.text.strip() if heading_span else "Unknown"
        
        # Creator
        creator_spans = soup.find_all('span', class_='H1')
        data['creator'] = "Unknown"
        for span in creator_spans:
            text = span.text.strip()
            if text and text != "Unknown":
                data['creator'] = text
                break
        
        # Image: only accept high-confidence model image candidates.
        image_url = _pick_best_model_image(soup, data.get('name', ''))
        data['image_url'] = image_url if image_url else "No Image"
        
        # Metadata from tables
        data['year'] = "Unknown"
        data['difficulty'] = "Unknown"
        data['paper'] = "Unknown"
        data['pieces'] = "Unknown"
        data['cuts'] = "Unknown"
        data['glue'] = "Unknown"
        data['rating_avg'] = "Unknown"
        data['difficulty_avg'] = "Unknown"
        data['source_url'] = model_url
        data['description'] = "Unknown"
        
        tables = soup.find_all('table', class_=lambda x: x and 'border-0' in x and 'border-cell-0' in x)
        
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                if len(cells) >= 2:
                    header = cells[0].text.strip().lower()
                    cell_data = cells[1].text.strip()
                    
                    if 'paper' in header:
                        data['paper'] = cell_data
                    elif 'pieces' in header:
                        data['pieces'] = cell_data
                    elif 'cuts' in header:
                        data['cuts'] = cell_data
                    elif 'glue' in header:
                        data['glue'] = cell_data
                    elif 'difficulty' in header or 'level' in header:
                        data['difficulty'] = cell_data
                    elif 'year' in header or 'date' in header:
                        data['year'] = cell_data
        
        # Regex fallback for rating/difficulty
        page_text = soup.get_text(separator='\n')
        if data['rating_avg'] == "Unknown":
            m = re.search(r"Rating\s*\(Average\)\s*([0-9]+(?:\.[0-9]+)?)", page_text, re.IGNORECASE)
            if m:
                data['rating_avg'] = m.group(1)
        if data['difficulty_avg'] == "Unknown":
            m = re.search(r"Difficulty\s*\(Average\)\s*([0-9]+(?:\.[0-9]+)?)", page_text, re.IGNORECASE)
            if m:
                data['difficulty_avg'] = m.group(1)
        
        # Convert yes/no to bool
        try:
            data['cuts'] = str(data['cuts']).lower() in ("yes", "1", "true")
            data['glue'] = str(data['glue']).lower() in ("yes", "1", "true")
        except:
            data['cuts'] = False
            data['glue'] = False
        
        return data
        
    except Exception as e:
        try:
            print(f"Error scraping {model_url}: {e}")
        except:
            print("Error scraping (encoding issue)")
        return None


def scrape_and_save(letter='A', max_pages=1):
    """Scrape models and save to PostgreSQL with parallel processing."""
    session = get_session()
    saved_count = 0
    
    for page_num in range(1, max_pages + 1):
        url = f"https://oriwiki.com/browseModels.php?PN={page_num}&Letter={letter}"
        print(f"\nFetching: {url}")
        
        try:
            response = session.get(url, timeout=10)
            if response.status_code != 200:
                break
            
            soup = BeautifulSoup(response.text, 'html.parser')
            gallery_cells = soup.find_all('td', class_='gallery-width')
            
            if not gallery_cells:
                break
            
            # Collect all model URLs first
            model_urls = []
            for cell in gallery_cells:
                link_tag = cell.find('a', href=True)
                if not link_tag or 'showModel.php' not in link_tag.get('href', ''):
                    continue
                
                href = link_tag.get('href')
                if href.startswith('/'):
                    model_url = BASE_URL + href
                elif href.startswith('http'):
                    model_url = href
                else:
                    model_url = BASE_URL + '/' + href
                model_urls.append(model_url)
            
            # Scrape all models in parallel (10 threads)
            print(f"  Scraping {len(model_urls)} models in parallel (10 workers)...")
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = {executor.submit(scrape_model_details, url, session): url for url in model_urls}
                
                all_model_data = []
                for future in as_completed(futures):
                    model_data = future.result()
                    if model_data:
                        all_model_data.append(model_data)
            
            # Save sequentially (single shared connection, avoids overloading DB)
            print(f"  Saving {len(all_model_data)} models to database...")
            for data in all_model_data:
                try:
                    result = save_model_comprehensive(data)
                    if result:
                        saved_count += 1
                except Exception as e:
                    print(f"  Save error: {str(e)[:80]}")
        
        except Exception as e:
            print(f"Error on page {page_num}: {str(e)[:100]}")
            break
    
    return saved_count


if __name__ == "__main__":
    import sys
    
    letter = sys.argv[1] if len(sys.argv) > 1 else 'A'
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    print(f"Scraping letter '{letter}', max {max_pages} pages...")
    count = scrape_and_save(letter, max_pages)
    print(f"\nTotal saved: {count} models")
