import sys
sys.path.insert(0, '.')

from comprehensive_scraper import scrape_and_save
from cfc_scraping import main as cfc_main
from orc_scraping import run_scrape as orc_run_scrape

letters = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') + list('123456789')
# Note: '#' removed - in URLs # is treated as a fragment, Letter= becomes empty returning all models
# Numbers (1-9) added: website has ~552 models starting with digits

# Resume point for ORIWiki letter scraping.
# Set to 'A' for full restart, or resume from another letter.
start_letter = 'A'
try:
    start_from = letters.index(start_letter.upper())
except ValueError:
    start_from = 0
    start_letter = 'A'

print(f"Starting full scrape of {len(letters)} letters (A-Z + 1-9)...")
print("This will take approximately 60-120 minutes...\n")
print(f"ORIWiki resume letter: {start_letter}\n")

total_models = 0

for i, letter in enumerate(letters, 1):
    if i <= start_from:
        continue
    
    print(f"\n[{i}/{len(letters)}] Scraping letter '{letter}'...")
    print("-" * 50)
    
    try:
        count = scrape_and_save(letter, 999)
        total_models += count
        print(f"\n Saved {count} models for letter '{letter}'")
    except KeyboardInterrupt:
        print(f"\n Interrupted at letter '{letter}'")
        break
    except Exception as e:
        print(f"\n Error scraping letter '{letter}': {e}")
        continue

print(f"\n" + "="*50)
print(f" Full scrape complete!")
print(f" Total models saved: {total_models}")
print(f"="*50)

# ---- CFC Scraping ----
print(f"\n" + "="*50)
print(f" Starting CFC scraping (diagrams, books, resources, calls)...")
print(f"="*50)

# Clear argv so cfc_main() runs all categories
sys.argv = [sys.argv[0]]
try:
    cfc_main()
except KeyboardInterrupt:
    print("\n[!] CFC scraping interrupted by user.")
except Exception as e:
    print(f"\n[!] CFC scraping error: {e}")

# ---- ORC Scraping ----
print(f"\n" + "="*50)
print(" Starting ORC scraping...")
print(f"="*50)

try:
    orc_run_scrape()
except KeyboardInterrupt:
    print("\n[!] ORC scraping interrupted by user.")
except Exception as e:
    print(f"\n[!] ORC scraping error: {e}")
