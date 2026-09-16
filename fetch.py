"""
fetch_products.py

Downloads real product data + images from the free FakeStoreAPI
(https://fakestoreapi.com/products) and saves them into ./products/
so they can be used with app.py (the visual search demo).

FakeStoreAPI is a live/public API - it always returns whatever products
currently exist on their server (usually ~20 items across categories like
electronics, jewelery, men's/women's clothing).

Usage
-----
    pip install requests
    python fetch_products.py

This will create/refresh:
    products/<id>_<slugified-title>.jpg
    products/products_meta.json   <- id, title, price, category, original url

Re-running it is safe - it just re-downloads/overwrites the same files.
"""

import os
import re
import json
import sys

import requests

API_URL = "https://fakestoreapi.com/products"
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "products")
META_FILE = os.path.join(OUT_DIR, "products_meta.json")

TIMEOUT = 15


def slugify(text: str, max_len: int = 40) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:max_len] or "product"


def fetch_product_list():
    print(f"Fetching product list from {API_URL} ...")
    resp = requests.get(API_URL, timeout=TIMEOUT)
    resp.raise_for_status()
    products = resp.json()
    print(f"  -> {len(products)} products found")
    return products


def download_image(url: str, dest_path: str) -> bool:
    try:
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(r.content)
        return True
    except Exception as e:
        print(f"  [warn] failed to download {url}: {e}")
        return False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    products = fetch_product_list()
    if not products:
        print("No products returned by the API. Exiting.")
        sys.exit(1)

    meta = []
    ok, failed = 0, 0

    for p in products:
        pid = p.get("id")
        title = p.get("title", f"product_{pid}")
        category = p.get("category", "unknown")
        price = p.get("price")
        image_url = p.get("image")

        if not image_url:
            print(f"  [skip] product {pid} has no image url")
            failed += 1
            continue

        # figure out extension from the URL, default to .jpg
        ext = os.path.splitext(image_url)[1].split("?")[0]
        if ext.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            ext = ".jpg"

        filename = f"{pid}_{slugify(title)}{ext}"
        dest_path = os.path.join(OUT_DIR, filename)

        print(f"Downloading [{pid}] {title[:50]!r} -> {filename}")
        if download_image(image_url, dest_path):
            ok += 1
            meta.append(
                {
                    "id": pid,
                    "title": title,
                    "category": category,
                    "price": price,
                    "source_url": image_url,
                    "file": filename,
                }
            )
        else:
            failed += 1

    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone.")
    print(f"  Downloaded : {ok}")
    print(f"  Failed     : {failed}")
    print(f"  Saved to   : {OUT_DIR}")
    print(f"  Metadata   : {META_FILE}")
    print("\nNow just run: python app.py  (it will auto re-index the new images)")


if __name__ == "__main__":
    main()