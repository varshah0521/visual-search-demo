# Visual Search Demo

Image-based product search for a WooCommerce store. Upload (or snap) a photo and
the app returns the visually closest products from your catalog, with a
similarity score and a link to each product page.

The whole app — Flask routes, model loading, embedding/index code and the
HTML/CSS/JS UI — lives in a single file, `app.py`. There is no database, no
frontend build step and no Docker.

```
visual-search-demo/
├── app.py            <- Flask server + embedded HTML/JS UI + models + search
├── fetch.py          <- optional helper: download sample products from FakeStoreAPI
├── requirements.txt
├── products/         <- sample product images (used by fetch.py, not by app.py)
└── cache/            <- auto-created, stores embedding indexes (index_<mode>.npz)
```

## How the visual search works

```
Uploaded image ──▶ embedding model ──▶ query vector (L2-normalized)
                                              │
WooCommerce products ──▶ first product image ──▶ same model ──▶ product vectors (cached .npz)
                                              │
                        cosine similarity (NumPy dot product), score >= min_score
                                              │
                                     Top-K matching products
```

1. **Catalog fetch** — `fetch_products()` pages through
   `GET {WC_BASE_URL}/wp-json/wc/v3/products` using your consumer key/secret,
   `WC_PER_PAGE` products per page for up to `WC_MAX_PAGES` pages. Products
   without an image are skipped.
2. **Embedding** — each product's first image is downloaded and encoded into a
   vector by the selected model, then L2-normalized:
   - `multimodal` — OpenAI CLIP ViT-B/32 image tower (512-d). Good default;
     captures semantic similarity ("a shoe looks like another shoe").
   - `siglip` — Google SigLIP ViT-B/16 (768-d pooled features). Trained with a
     sigmoid pairwise loss, usually a sharper image-to-image matcher than CLIP.
     Its similarity scores sit in a different range, so the `min_score`
     strictness slider generally needs lowering for this mode.
   - `dino` — Meta DINOv2 ViT-B/14 (768-d, CLS token). Self-supervised, tuned
     towards pure visual similarity (shape, texture, layout).
   - `dino3` — reserved for DINOv3 (`facebook/dinov3-vitb16-pretrain-lvd1689m`),
     a gated Hugging Face repo, which is why `HF_TOKEN` exists.
3. **Index cache** — vectors are saved to `cache/index_<mode>.npz` alongside the
   product IDs. On the next run the cache is reused as long as the catalog's ID
   list is unchanged; otherwise the index is rebuilt.
4. **Search** — the query vector is multiplied against the cached matrix
   (`embs @ query`), which is exactly cosine similarity for normalized vectors.
   Results are sorted descending and cut off at `min_score` (default `0.72`), so
   a query with no good match returns nothing rather than filler. At most
   `top_k` (default 8, max 20) results are returned.

No vector database is needed at this catalog size; for 100k+ products swap
`nearest_neighbors()` for FAISS/Annoy without touching anything else.

## Setup

Requires Python 3.9+ and internet access on first run (model weights) and
whenever the index is rebuilt (product images).

```bash
git clone https://github.com/varshah0521/visual-search-demo.git
cd visual-search-demo
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

CPU-only machines can install a smaller torch build first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Then configure the environment:

```bash
cp .env.example .env
# edit .env and fill in your WooCommerce credentials
```

## Environment variables

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `WC_BASE_URL` | yes | — | Base URL of the WooCommerce store, e.g. `https://shop.example.com` (no trailing slash needed) |
| `WC_CONSUMER_KEY` | yes | — | WooCommerce REST API consumer key (`ck_...`), read access is enough |
| `WC_CONSUMER_SECRET` | yes | — | WooCommerce REST API consumer secret (`cs_...`) |
| `WC_PER_PAGE` | no | `10` | Products requested per API page |
| `WC_MAX_PAGES` | no | `5` | Maximum pages pulled, so `WC_PER_PAGE * WC_MAX_PAGES` products get indexed |
| `HF_TOKEN` | no | — | Hugging Face access token, passed to `from_pretrained` for the SigLIP/DINO modes; required for gated weights such as DINOv3 |

Create the WooCommerce key under **WooCommerce → Settings → Advanced → REST API**.
Generate an `HF_TOKEN` at https://huggingface.co/settings/tokens.

Never commit real values — keep them in `.env` (git-ignored) and copy
`.env.example` as the template.

## Run

```bash
source venv/bin/activate
python app.py
```

Startup prints the device (`cuda`/`cpu`), pre-builds the index for each mode and
then serves on http://localhost:5000 (bound to `0.0.0.0:5000`).

In the UI: pick a model, choose the number of results and the minimum similarity,
upload or capture an image, and hit **Search**. `POST /search` accepts the same
inputs as a multipart form (`image`, `mode`, `top_k`, `min_score`) and responds
with JSON.

A 401 from the store means the API key is missing, revoked, lacks read access, or
a WAF/CDN is blocking `/wp-json/wc/v3/products`.

## Sample data (`fetch.py`)

`fetch.py` is independent of the WooCommerce flow: it downloads ~20 products and
their images from the free [FakeStoreAPI](https://fakestoreapi.com/products) into
`products/`, plus `products/products_meta.json` (id, title, price, category,
source URL). Re-running it overwrites the same files.

```bash
pip install requests
python fetch.py
```

Useful for a local sample catalog; `app.py` in its current form indexes the live
WooCommerce catalog rather than `products/`.
