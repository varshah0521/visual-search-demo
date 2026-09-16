# Visual Search Demo (single file, no Docker)

Upload or capture a photo -> it's converted to a vector embedding -> nearest
neighbor search finds the closest matching products from `products/`.

Everything lives in **one file**: `app.py`. No Docker, no database, no
frontend build step (the UI is plain HTML/JS embedded in the Python file).

## What's inside the zip

```
visual_search/
├── app.py              <- the whole app (Flask server + HTML/JS UI + models)
├── requirements.txt
├── products/            <- sample "product" images (14 placeholders you can replace)
└── cache/                <- created automatically, stores embedding cache (.npz)
```

## 1. Install (one time)

Requires Python 3.9+.

```bash
cd visual_search
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

If you're on CPU-only and want a smaller/faster torch install, you can instead run:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install flask open_clip_torch pillow numpy
```

## 2. Run

```bash
python app.py
```

Then open **http://localhost:5000**

First run downloads the two model checkpoints (needs internet once):
- ResNet50 (ImageNet) — Image Embedding Model / Vision Encoder — ~100 MB
- CLIP ViT-B/32 (OpenAI) — Multimodal Embedding Model — ~350 MB

Both together are well under the 1 GB budget. After the first run, everything
works fully offline, and product embeddings are cached in `cache/*.npz` so
re-indexing only happens if you add/remove images in `products/`.

## 3. Use it

1. Pick a model from the dropdown:
   - **Multimodal Embedding Model (CLIP)** — usually the better default; understands
     semantic/visual concepts (e.g. "shoe" looks similar to another shoe even
     if colors differ).
   - **Image Embedding Model (Vision Encoder / ResNet50)** — pure visual
     similarity based on low/mid-level image features (shape, texture, color).
2. Upload an image, or on mobile tap the box to open your camera.
3. Click **Search** — the matching products list appears below with similarity scores.

## 4. Use your own product catalog

Just drop your real product photos into `products/` (jpg/png/webp), delete
the old placeholder images if you like, and run the app again — it will
detect the changed file list and rebuild the index automatically.

## How the search works (architecture)

```
Uploaded image ──▶ Vision Encoder / CLIP ──▶ query vector (L2-normalized)
                                                     │
Product images ──▶ (same) embedding model ──▶ product vectors (cached)
                                                     │
                              cosine similarity search (numpy dot product)
                                                     │
                                        Top-K matching products
```

No external vector database is used — with a small/medium catalog, a plain
NumPy matrix multiply against cached embeddings is fast enough and keeps the
whole thing dependency-free. If you later scale to 100k+ products, swap the
`nearest_neighbors()` function for FAISS/Annoy without changing anything else.

## Notes

- The bundled `products/` images are synthetic placeholders (simple drawn
  shapes for shirts/shoes/bags/watches/etc.) just so the demo works out of
  the box. Replace them with real photos for meaningful results.
- Everything (routes, models, HTML/CSS/JS) is intentionally kept in the
  single `app.py` file as requested — easy to read top to bottom, easy to
  copy elsewhere.
