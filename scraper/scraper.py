"""
Sri Lanka Property Price Predictor — Web Scraper
Scrapes property listings from lankapropertyweb.com
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import os
import re
import argparse
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

# Base URLs for each property type
LISTING_URLS = {
    "House": "https://www.lankapropertyweb.com/sale/index.php?page={page}&property-type=House",
    "Apartment": "https://www.lankapropertyweb.com/sale/index.php?page={page}&property-type=Apartment",
}

DELAY_BETWEEN_REQUESTS = 1.5  # seconds — be respectful to the server
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_data.csv")


# ============================================================
# STEP 1: Get Property Detail Page URLs from Listing Pages
# ============================================================

def get_property_urls_from_page(page_url):
    """Extract individual property detail URLs from a listing page."""
    try:
        response = requests.get(page_url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        # Find all links to property detail pages
        links = soup.find_all("a", href=re.compile(r"property_details-\d+\.html"))
        
        # Extract unique URLs
        urls = set()
        for link in links:
            href = link.get("href", "")
            if "property_details" in href:
                # Make absolute URL if relative
                if href.startswith("/"):
                    href = "https://www.lankapropertyweb.com" + href
                elif not href.startswith("http"):
                    href = "https://www.lankapropertyweb.com/" + href
                urls.add(href)

        return list(urls)
    except Exception as e:
        print(f"  [ERROR] Failed to fetch listing page: {page_url} — {e}")
        return []


def get_total_pages(property_type):
    """Get the total number of pages for a given property type."""
    url = LISTING_URLS[property_type].format(page=1)
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        # Try to extract total count from page header like "Houses for Sale (3926 properties)"
        header_text = soup.get_text()
        count_match = re.search(r"\((\d[\d,]*)\s*propert", header_text)
        if count_match:
            total_count = int(count_match.group(1).replace(",", ""))
            # Each page has approximately 30 listings
            total_pages = (total_count // 30) + 1
            return total_pages

        # Fallback: try pagination links
        pagination = soup.find("ul", class_="pagination")
        if pagination:
            page_links = pagination.find_all("a", href=re.compile(r"page=\d+"))
            max_page = 1
            for link in page_links:
                href = link.get("href", "")
                match = re.search(r"page=(\d+)", href)
                if match:
                    max_page = max(max_page, int(match.group(1)))
            return max_page

        return 100  # Safe default fallback
    except Exception as e:
        print(f"  [ERROR] Failed to get total pages for {property_type}: {e}")
        return 100


# ============================================================
# STEP 2: Scrape Detail Page for Property Data
# ============================================================

def parse_price(price_text):
    """Convert price text like 'Rs. 72,000,000' to a numeric value."""
    if not price_text:
        return None
    # Remove 'Rs.' or 'Rs' prefix first, then strip commas, spaces, and other chars
    cleaned = price_text.strip()
    cleaned = re.sub(r"^Rs\.?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace(",", "").strip()
    # Extract the numeric part
    match = re.search(r"(\d+(?:\.\d+)?)", cleaned)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def parse_floor_area(area_text):
    """Convert floor area text like '720 sq.ft.' to numeric value in sqft."""
    if not area_text:
        return None
    match = re.search(r"([\d,]+(?:\.\d+)?)\s*(?:sq\.?\s*ft|sqft)", area_text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", ""))
    return None


def parse_land_size(land_text):
    """Convert land size text to numeric value in perches."""
    if not land_text:
        return None
    # Try perches first
    match = re.search(r"([\d,]+(?:\.\d+)?)\s*(?:perch|perches|p)", land_text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", ""))
    # Try acres (1 acre = 160 perches)
    match = re.search(r"([\d,]+(?:\.\d+)?)\s*(?:acre|acres|ac)", land_text, re.IGNORECASE)
    if match:
        return float(match.group(1).replace(",", "")) * 160
    return None


def scrape_property_details(url):
    """Scrape a single property detail page and return a dict of features."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "lxml")

        data = {"url": url}

        # --- Title ---
        h1 = soup.find("h1")
        data["title"] = h1.get_text(strip=True) if h1 else None

        # --- Price ---
        price_el = soup.find("span", class_=re.compile(r"main_price"))
        data["price"] = parse_price(price_el.get_text(strip=True)) if price_el else None

        # --- Location (from breadcrumb) ---
        breadcrumb = soup.find("ol", id="nav_breadcrumb")
        if breadcrumb:
            items = breadcrumb.find_all("li")
            # Typically: Home > Sales > District > City > Type
            # We want the city/area (second-to-last or third-to-last item)
            if len(items) >= 4:
                data["location"] = items[-2].get_text(strip=True)
                data["district"] = items[-3].get_text(strip=True) if len(items) >= 5 else None
            elif len(items) >= 3:
                data["location"] = items[-2].get_text(strip=True)
                data["district"] = None
            else:
                data["location"] = None
                data["district"] = None
        else:
            data["location"] = None
            data["district"] = None

        # --- Overview Section (Property Type, Bedrooms, etc.) ---
        overview = soup.find("div", id="Overview") or soup.find("div", class_="overview")
        if overview:
            items = overview.find_all("div", class_="overview-item")
            for item in items:
                label_el = item.find("div", class_="label")
                value_el = item.find("div", class_="value")
                if label_el and value_el:
                    label = label_el.get_text(strip=True).lower()
                    value = value_el.get_text(strip=True)

                    if "property type" in label:
                        data["property_type"] = value
                    elif "bedroom" in label:
                        try:
                            data["bedrooms"] = int(re.search(r"\d+", value).group())
                        except (AttributeError, ValueError):
                            data["bedrooms"] = None
                    elif "bathroom" in label or "wc" in label:
                        try:
                            data["bathrooms"] = int(re.search(r"\d+", value).group())
                        except (AttributeError, ValueError):
                            data["bathrooms"] = None
                    elif "floor area" in label:
                        data["floor_area_sqft"] = parse_floor_area(value)
                    elif "area of land" in label or "land size" in label or "land area" in label:
                        data["land_size_perches"] = parse_land_size(value)
                    elif "age of building" in label or "building age" in label:
                        data["age_of_building"] = value
                    elif "furnishing" in label:
                        data["furnishing"] = value
                    elif "construction" in label:
                        data["construction_status"] = value
                    elif "price per" in label:
                        pass  # skip derived field

        # Fill in missing keys with None
        for key in ["property_type", "bedrooms", "bathrooms", "floor_area_sqft",
                     "land_size_perches", "age_of_building", "furnishing", "construction_status"]:
            if key not in data:
                data[key] = None

        return data

    except Exception as e:
        print(f"  [ERROR] Failed to scrape {url} — {e}")
        return None


# ============================================================
# STEP 3: Main Scraping Pipeline
# ============================================================

def scrape_all(max_pages_per_type=None, resume=True):
    """
    Main scraping function.
    
    Args:
        max_pages_per_type: Limit pages per property type (None = scrape all)
        resume: If True, skip URLs already scraped
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load existing data to enable resume
    existing_urls = set()
    all_records = []
    if resume and os.path.exists(OUTPUT_FILE):
        existing_df = pd.read_csv(OUTPUT_FILE)
        existing_urls = set(existing_df["url"].tolist())
        all_records = existing_df.to_dict("records")
        print(f"[RESUME] Found {len(existing_urls)} previously scraped properties")

    total_scraped = 0
    total_errors = 0

    for prop_type, base_url in LISTING_URLS.items():
        print(f"\n{'='*60}")
        print(f"[SCRAPING] {prop_type} listings...")
        print(f"{'='*60}")

        # Get total pages
        total_pages = get_total_pages(prop_type)
        if max_pages_per_type:
            total_pages = min(total_pages, max_pages_per_type)
        print(f"[INFO] Total pages to scrape: {total_pages}")

        for page_num in range(1, total_pages + 1):
            page_url = base_url.format(page=page_num)
            print(f"\n--- Page {page_num}/{total_pages} ---")

            # Get property URLs from this listing page
            property_urls = get_property_urls_from_page(page_url)
            print(f"  Found {len(property_urls)} properties on this page")

            for i, prop_url in enumerate(property_urls):
                # Skip if already scraped
                if prop_url in existing_urls:
                    continue

                print(f"  [{i+1}/{len(property_urls)}] Scraping: {prop_url[:80]}...")
                data = scrape_property_details(prop_url)

                if data and data.get("price"):
                    all_records.append(data)
                    existing_urls.add(prop_url)
                    total_scraped += 1
                else:
                    total_errors += 1

                # Be respectful to the server
                time.sleep(DELAY_BETWEEN_REQUESTS)

            # Save after each page (incremental save)
            if all_records:
                df = pd.DataFrame(all_records)
                df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
                print(f"  [SAVED] {len(all_records)} total records to {OUTPUT_FILE}")

            time.sleep(DELAY_BETWEEN_REQUESTS)

    # Final summary
    print(f"\n{'='*60}")
    print(f"[DONE] Scraping Complete!")
    print(f"   Total new records scraped: {total_scraped}")
    print(f"   Total errors/skipped: {total_errors}")
    print(f"   Total records in dataset: {len(all_records)}")
    print(f"   Output file: {OUTPUT_FILE}")
    print(f"{'='*60}")


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape Sri Lankan property listings")
    parser.add_argument(
        "--pages", type=int, default=None,
        help="Max pages to scrape per property type (default: all)"
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh instead of resuming"
    )
    args = parser.parse_args()

    print(f"[START] Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    scrape_all(max_pages_per_type=args.pages, resume=not args.no_resume)
    print(f"[END] Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
