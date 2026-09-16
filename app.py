import os
# """
# Visual Search Demo  —  single-file app, no Docker required.

# What it does
# ------------
# 1. Upload an image (or capture a photo from your camera on mobile).
# 2. The image is converted into a vector ("embedding") using one of two
#    swappable models:
#      - "vision"     -> Image Embedding Model (pure Vision Encoder, ResNet50)
#      - "multimodal" -> Multimodal Embedding Model (OpenAI CLIP ViT-B/32)
# 3. Every product image in ./products/ is embedded the same way (cached to
#    disk after the first run) and a nearest-neighbor cosine-similarity
#    search returns the closest matching products.

# Both models together are well under 1 GB of weights:
#   - ResNet50 (ImageNet)      ~ 100 MB
#   - CLIP ViT-B/32 (OpenAI)   ~ 350 MB

# Run it
# -------
#     python3 -m venv venv
#     source venv/bin/activate          (Windows: venv\\Scripts\\activate)
#     pip install -r requirements.txt
#     python app.py
#     open http://localhost:5000

# First run will download the two model checkpoints (needs internet once);
# after that everything works fully offline. Product embeddings are cached
# in ./cache/ so re-indexing only happens when you add/remove product images.

# Replace the images in ./products/ with your own real product photos any
# time — the index will rebuild automatically the next time you search.
# """

# import os
# import io
# import json
# import traceback

# import numpy as np
# from PIL import Image
# from flask import Flask, request, jsonify, send_from_directory, Response

# import torch

# # --------------------------------------------------------------------------
# # Config
# # --------------------------------------------------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PRODUCTS_DIR = os.path.join(BASE_DIR, "products")
# CACHE_DIR = os.path.join(BASE_DIR, "cache")
# os.makedirs(PRODUCTS_DIR, exist_ok=True)
# os.makedirs(CACHE_DIR, exist_ok=True)

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# _models = {}  # lazy-loaded model cache: {"vision": (model, preprocess), "multimodal": (...)}


# # --------------------------------------------------------------------------
# # Models
# # --------------------------------------------------------------------------
# def get_vision_model():
#     """Pure Image Embedding Model (Vision Encoder): ResNet50 pretrained on
#     ImageNet, classification head removed -> 2048-d visual feature vector."""
#     if "vision" not in _models:
#         import torchvision.models as tvm
#         import torchvision.transforms as T

#         weights = tvm.ResNet50_Weights.IMAGENET1K_V2
#         model = tvm.resnet50(weights=weights)
#         model.fc = torch.nn.Identity()
#         model.eval().to(DEVICE)

#         preprocess = T.Compose(
#             [
#                 T.Resize(256),
#                 T.CenterCrop(224),
#                 T.ToTensor(),
#                 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ]
#         )
#         _models["vision"] = (model, preprocess)
#     return _models["vision"]


# def get_multimodal_model():
#     """Multimodal Embedding Model: OpenAI CLIP ViT-B/32 image tower.
#     Same embedding space as text, so this model could also match products
#     by a text query in the future -- here we use its image encoder."""
#     if "multimodal" not in _models:
#         import open_clip

#         model, _, preprocess = open_clip.create_model_and_transforms(
#             "ViT-B-32", pretrained="openai"
#         )
#         model.eval().to(DEVICE)
#         _models["multimodal"] = (model, preprocess)
#     return _models["multimodal"]


# def embed_image(img: Image.Image, mode: str) -> np.ndarray:
#     img = img.convert("RGB")
#     with torch.no_grad():
#         if mode == "multimodal":
#             model, preprocess = get_multimodal_model()
#             tensor = preprocess(img).unsqueeze(0).to(DEVICE)
#             feat = model.encode_image(tensor)
#         else:  # "vision"
#             model, preprocess = get_vision_model()
#             tensor = preprocess(img).unsqueeze(0).to(DEVICE)
#             feat = model(tensor)
#         feat = feat / feat.norm(dim=-1, keepdim=True)
#     return feat.cpu().numpy()[0].astype(np.float32)


# # --------------------------------------------------------------------------
# # Product index (build once, cache to disk, rebuild if product list changes)
# # --------------------------------------------------------------------------
# def _index_path(mode):
#     return os.path.join(CACHE_DIR, f"index_{mode}.npz")


# def _list_product_files():
#     return sorted(f for f in os.listdir(PRODUCTS_DIR) if f.lower().endswith(IMG_EXTS))


# def build_or_load_index(mode: str):
#     files = _list_product_files()
#     path = _index_path(mode)

#     if os.path.exists(path):
#         data = np.load(path, allow_pickle=True)
#         cached_files = list(data["files"])
#         if cached_files == files:
#             return np.array(files), data["embs"]

#     embs = []
#     for f in files:
#         img_path = os.path.join(PRODUCTS_DIR, f)
#         try:
#             img = Image.open(img_path)
#             embs.append(embed_image(img, mode))
#         except Exception as e:
#             print(f"[warn] could not embed {f}: {e}")

#     embs_arr = np.stack(embs).astype(np.float32) if embs else np.zeros((0, 512), dtype=np.float32)
#     np.savez(path, files=np.array(files, dtype=object), embs=embs_arr)
#     return np.array(files), embs_arr


# def nearest_neighbors(query_emb: np.ndarray, files: np.ndarray, embs: np.ndarray, top_k: int):
#     if len(files) == 0:
#         return []
#     sims = embs @ query_emb  # cosine similarity, vectors are L2-normalized
#     top_idx = np.argsort(-sims)[:top_k]
#     return [(str(files[i]), float(sims[i])) for i in top_idx]


# # --------------------------------------------------------------------------
# # Flask app
# # --------------------------------------------------------------------------
# app = Flask(__name__)

# HTML_PAGE = """<!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <meta name="viewport" content="width=device-width, initial-scale=1.0">
# <title>Visual Search</title>
# <style>
#   :root {
#     --navy: #0B1E3D;
#     --orange: #FF6A2B;
#     --ice: #6EC6E8;
#     --bg: #F4F6F9;
#   }
#   * { box-sizing: border-box; }
#   body {
#     margin: 0; font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
#     background: var(--bg); color: var(--navy);
#   }
#   header {
#     background: var(--navy); color: #fff; padding: 22px 24px;
#   }
#   header h1 { margin: 0; font-size: 22px; }
#   header p { margin: 4px 0 0; color: var(--ice); font-size: 13px; }
#   .wrap { max-width: 1000px; margin: 0 auto; padding: 24px; }
#   .panel {
#     background: #fff; border-radius: 14px; padding: 22px;
#     box-shadow: 0 2px 10px rgba(0,0,0,0.06); margin-bottom: 24px;
#   }
#   .row { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-end; }
#   .field { display: flex; flex-direction: column; gap: 6px; }
#   label { font-size: 13px; font-weight: 600; color: var(--navy); }
#   select, input[type="file"], input[type="range"] {
#     padding: 9px 10px; border-radius: 8px; border: 1px solid #d7dce3; font-size: 14px;
#   }
#   .dropzone {
#     border: 2px dashed #c3cbd6; border-radius: 12px; padding: 28px;
#     text-align: center; cursor: pointer; transition: border-color .15s;
#     flex: 1; min-width: 240px;
#   }
#   .dropzone.drag { border-color: var(--orange); background: #fff7f2; }
#   .dropzone img { max-height: 160px; border-radius: 8px; margin-top: 10px; }
#   .btn {
#     background: var(--orange); color: #fff; border: none; padding: 11px 20px;
#     border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 14px;
#   }
#   .btn:disabled { background: #c9b7ac; cursor: default; }
#   .btn.secondary { background: var(--navy); }
#   .results { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px,1fr)); gap: 16px; }
#   .card {
#     background: #fff; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 6px rgba(0,0,0,0.08);
#   }
#   .card img { width: 100%; height: 130px; object-fit: cover; display: block; }
#   .card .meta { padding: 8px 10px; font-size: 12px; }
#   .card .name { font-weight: 600; word-break: break-all; }
#   .card .score { color: var(--orange); font-weight: 700; }
#   .status { font-size: 13px; color: #556; margin-top: 8px; }
#   .hidden { display: none; }
#   .spinner {
#     width: 18px; height: 18px; border: 3px solid #eee; border-top-color: var(--orange);
#     border-radius: 50%; display: inline-block; animation: spin 0.8s linear infinite; vertical-align: middle;
#   }
#   @keyframes spin { to { transform: rotate(360deg); } }
# </style>
# </head>
# <body>
# <header>
#   <h1>Visual Search</h1>
#   <p>Upload a photo &mdash; find visually similar products by nearest-neighbor vector search</p>
# </header>

# <div class="wrap">
#   <div class="panel">
#     <div class="row">
#       <div class="dropzone" id="dropzone">
#         <div id="dzText">Upload image or capture photo<br><small>click / drag &amp; drop</small></div>
#         <img id="preview" class="hidden">
#         <input type="file" id="fileInput" accept="image/*" capture="environment" class="hidden">
#       </div>

#       <div class="field">
#         <label for="modelSelect">Embedding Model</label>
#         <select id="modelSelect">
#           <option value="multimodal">Multimodal Embedding Model (CLIP ViT-B/32)</option>
#           <option value="vision">Image Embedding Model (Vision Encoder, ResNet50)</option>
#         </select>
#       </div>

#       <div class="field">
#         <label for="topk">Results: <span id="topkVal">8</span></label>
#         <input type="range" id="topk" min="1" max="14" value="8">
#       </div>

#       <div class="field">
#         <button class="btn" id="searchBtn" disabled>Search</button>
#       </div>
#     </div>
#     <div class="status" id="status"></div>
#   </div>

#   <div class="panel">
#     <h3 style="margin-top:0;">Matching products list</h3>
#     <div class="results" id="results"></div>
#   </div>
# </div>

# <script>
# const dropzone = document.getElementById('dropzone');
# const dzText = document.getElementById('dzText');
# const preview = document.getElementById('preview');
# const fileInput = document.getElementById('fileInput');
# const searchBtn = document.getElementById('searchBtn');
# const statusEl = document.getElementById('status');
# const resultsEl = document.getElementById('results');
# const topk = document.getElementById('topk');
# const topkVal = document.getElementById('topkVal');
# const modelSelect = document.getElementById('modelSelect');

# let currentFile = null;

# topk.addEventListener('input', () => topkVal.textContent = topk.value);
# dropzone.addEventListener('click', () => fileInput.click());

# ['dragover','dragenter'].forEach(ev => dropzone.addEventListener(ev, e => {
#   e.preventDefault(); dropzone.classList.add('drag');
# }));
# ['dragleave','drop'].forEach(ev => dropzone.addEventListener(ev, e => {
#   e.preventDefault(); dropzone.classList.remove('drag');
# }));
# dropzone.addEventListener('drop', e => {
#   if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
# });
# fileInput.addEventListener('change', e => {
#   if (e.target.files.length) handleFile(e.target.files[0]);
# });

# function handleFile(file) {
#   currentFile = file;
#   const reader = new FileReader();
#   reader.onload = e => {
#     preview.src = e.target.result;
#     preview.classList.remove('hidden');
#     dzText.classList.add('hidden');
#   };
#   reader.readAsDataURL(file);
#   searchBtn.disabled = false;
# }

# searchBtn.addEventListener('click', async () => {
#   if (!currentFile) return;
#   searchBtn.disabled = true;
#   statusEl.innerHTML = '<span class="spinner"></span> Embedding image and searching...';
#   resultsEl.innerHTML = '';

#   const fd = new FormData();
#   fd.append('image', currentFile);
#   fd.append('mode', modelSelect.value);
#   fd.append('top_k', topk.value);

#   try {
#     const res = await fetch('/search', { method: 'POST', body: fd });
#     const data = await res.json();
#     if (data.error) throw new Error(data.error);

#     if (!data.results.length) {
#       statusEl.textContent = 'No products indexed yet. Add images to the products/ folder.';
#     } else {
#       statusEl.textContent = `Found ${data.results.length} matches in ${data.elapsed_ms} ms using ${data.mode_label}.`;
#       resultsEl.innerHTML = data.results.map(r => `
#         <div class="card">
#           <img src="${r.url}" alt="${r.file}">
#           <div class="meta">
#             <div class="name">${r.file}</div>
#             <div class="score">${(r.score*100).toFixed(1)}% match</div>
#           </div>
#         </div>
#       `).join('');
#     }
#   } catch (err) {
#     statusEl.textContent = 'Error: ' + err.message;
#   } finally {
#     searchBtn.disabled = false;
#   }
# });
# </script>
# </body>
# </html>
# """


# @app.route("/")
# def home():
#     return Response(HTML_PAGE, mimetype="text/html")


# @app.route("/products/<path:filename>")
# def product_file(filename):
#     return send_from_directory(PRODUCTS_DIR, filename)


# @app.route("/search", methods=["POST"])
# def do_search():
#     import time

#     mode = request.form.get("mode", "multimodal")
#     if mode not in ("multimodal", "vision"):
#         mode = "multimodal"
#     try:
#         top_k = max(1, min(20, int(request.form.get("top_k", 8))))
#     except ValueError:
#         top_k = 8

#     file = request.files.get("image")
#     if file is None:
#         return jsonify({"error": "No image uploaded."}), 400

#     try:
#         t0 = time.time()
#         img = Image.open(io.BytesIO(file.read()))
#         query_emb = embed_image(img, mode)
#         files, embs = build_or_load_index(mode)
#         results = nearest_neighbors(query_emb, files, embs, top_k)
#         elapsed_ms = int((time.time() - t0) * 1000)

#         mode_label = "CLIP (multimodal)" if mode == "multimodal" else "ResNet50 (vision encoder)"
#         return jsonify(
#             {
#                 "mode_label": mode_label,
#                 "elapsed_ms": elapsed_ms,
#                 "results": [
#                     {"file": f, "score": s, "url": f"/products/{f}"} for f, s in results
#                 ],
#             }
#         )
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     print(f"Device: {DEVICE}")
#     print("Pre-building product indexes (first run downloads model weights)...")
#     for m in ("multimodal", "vision"):
#         try:
#             f, e = build_or_load_index(m)
#             print(f"  [{m}] indexed {len(f)} product images -> {e.shape}")
#         except Exception as ex:
#             print(f"  [{m}] index build failed (will retry on first search): {ex}")

#     print("\nOpen http://localhost:5000 in your browser\n")
#     app.run(host="0.0.0.0", port=5000, debug=False)
# """
# Visual Search Demo  —  single-file app, no Docker required.

# What it does
# ------------
# 1. Upload an image (or capture a photo from your camera on mobile).
# 2. The image is converted into a vector ("embedding") using one of two
#    swappable models:
#      - "vision"     -> Image Embedding Model (pure Vision Encoder, ResNet50)
#      - "multimodal" -> Multimodal Embedding Model (OpenAI CLIP ViT-B/32)
# 3. Every product image in ./products/ is embedded the same way (cached to
#    disk after the first run) and a nearest-neighbor cosine-similarity
#    search returns the closest matching products.

# Both models together are well under 1 GB of weights:
#   - ResNet50 (ImageNet)      ~ 100 MB
#   - CLIP ViT-B/32 (OpenAI)   ~ 350 MB

# Run it
# -------
#     python3 -m venv venv
#     source venv/bin/activate          (Windows: venv\\Scripts\\activate)
#     pip install -r requirements.txt
#     python app.py
#     open http://localhost:5000

# First run will download the two model checkpoints (needs internet once);
# after that everything works fully offline. Product embeddings are cached
# in ./cache/ so re-indexing only happens when you add/remove product images.

# Replace the images in ./products/ with your own real product photos any
# time — the index will rebuild automatically the next time you search.
# """

# import os
# import io
# import json
# import traceback

# import numpy as np
# from PIL import Image
# from flask import Flask, request, jsonify, send_from_directory, Response

# import torch

# # --------------------------------------------------------------------------
# # Config
# # --------------------------------------------------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PRODUCTS_DIR = os.path.join(BASE_DIR, "products")
# CACHE_DIR = os.path.join(BASE_DIR, "cache")
# os.makedirs(PRODUCTS_DIR, exist_ok=True)
# os.makedirs(CACHE_DIR, exist_ok=True)

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# _models = {}  # lazy-loaded model cache: {"vision": (model, preprocess), "multimodal": (...)}


# # --------------------------------------------------------------------------
# # Models
# # --------------------------------------------------------------------------
# def get_vision_model():
#     """Pure Image Embedding Model (Vision Encoder): ResNet50 pretrained on
#     ImageNet, classification head removed -> 2048-d visual feature vector."""
#     if "vision" not in _models:
#         import torchvision.models as tvm
#         import torchvision.transforms as T

#         weights = tvm.ResNet50_Weights.IMAGENET1K_V2
#         model = tvm.resnet50(weights=weights)
#         model.fc = torch.nn.Identity()
#         model.eval().to(DEVICE)

#         preprocess = T.Compose(
#             [
#                 T.Resize(256),
#                 T.CenterCrop(224),
#                 T.ToTensor(),
#                 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ]
#         )
#         _models["vision"] = (model, preprocess)
#     return _models["vision"]


# def get_multimodal_model():
#     """Multimodal Embedding Model: OpenAI CLIP ViT-B/32 image tower.
#     Same embedding space as text, so this model could also match products
#     by a text query in the future -- here we use its image encoder."""
#     if "multimodal" not in _models:
#         import open_clip

#         model, _, preprocess = open_clip.create_model_and_transforms(
#             "ViT-B-32", pretrained="openai"
#         )
#         model.eval().to(DEVICE)
#         _models["multimodal"] = (model, preprocess)
#     return _models["multimodal"]


# def embed_image(img: Image.Image, mode: str) -> np.ndarray:
#     img = img.convert("RGB")
#     with torch.no_grad():
#         if mode == "multimodal":
#             model, preprocess = get_multimodal_model()
#             tensor = preprocess(img).unsqueeze(0).to(DEVICE)
#             feat = model.encode_image(tensor)
#         else:  # "vision"
#             model, preprocess = get_vision_model()
#             tensor = preprocess(img).unsqueeze(0).to(DEVICE)
#             feat = model(tensor)
#         feat = feat / feat.norm(dim=-1, keepdim=True)
#     return feat.cpu().numpy()[0].astype(np.float32)


# # --------------------------------------------------------------------------
# # Product index (build once, cache to disk, rebuild if product list changes)
# # --------------------------------------------------------------------------
# def _index_path(mode):
#     return os.path.join(CACHE_DIR, f"index_{mode}.npz")


# def _list_product_files():
#     return sorted(f for f in os.listdir(PRODUCTS_DIR) if f.lower().endswith(IMG_EXTS))


# def build_or_load_index(mode: str):
#     files = _list_product_files()
#     path = _index_path(mode)

#     if os.path.exists(path):
#         data = np.load(path, allow_pickle=True)
#         cached_files = list(data["files"])
#         if cached_files == files:
#             return np.array(files), data["embs"]

#     embs = []
#     for f in files:
#         img_path = os.path.join(PRODUCTS_DIR, f)
#         try:
#             img = Image.open(img_path)
#             embs.append(embed_image(img, mode))
#         except Exception as e:
#             print(f"[warn] could not embed {f}: {e}")

#     embs_arr = np.stack(embs).astype(np.float32) if embs else np.zeros((0, 512), dtype=np.float32)
#     np.savez(path, files=np.array(files, dtype=object), embs=embs_arr)
#     return np.array(files), embs_arr


# def nearest_neighbors(
#     query_emb: np.ndarray,
#     files: np.ndarray,
#     embs: np.ndarray,
#     top_k: int,
#     min_score: float = 0.0,
# ):
#     """Return up to top_k nearest neighbors, but only those whose cosine
#     similarity is >= min_score. If min_score filters everything out, an
#     empty list is returned (caller shows a "no match" message) instead of
#     forcing back irrelevant items just to fill the list."""
#     if len(files) == 0:
#         return []
#     sims = embs @ query_emb  # cosine similarity, vectors are L2-normalized
#     order = np.argsort(-sims)
#     results = []
#     for i in order:
#         score = float(sims[i])
#         if score < min_score:
#             break  # order is descending, so nothing after this will pass either
#         results.append((str(files[i]), score))
#         if len(results) >= top_k:
#             break
#     return results


# # --------------------------------------------------------------------------
# # Flask app
# # --------------------------------------------------------------------------
# app = Flask(__name__)

# HTML_PAGE = """<!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <meta name="viewport" content="width=device-width, initial-scale=1.0">
# <title>Visual Search</title>
# <style>
#   :root {
#     --navy: #0B1E3D;
#     --orange: #FF6A2B;
#     --ice: #6EC6E8;
#     --bg: #F4F6F9;
#   }
#   * { box-sizing: border-box; }
#   body {
#     margin: 0; font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
#     background: var(--bg); color: var(--navy);
#   }
#   header {
#     background: var(--navy); color: #fff; padding: 22px 24px;
#   }
#   header h1 { margin: 0; font-size: 22px; }
#   header p { margin: 4px 0 0; color: var(--ice); font-size: 13px; }
#   .wrap { max-width: 1000px; margin: 0 auto; padding: 24px; }
#   .panel {
#     background: #fff; border-radius: 14px; padding: 22px;
#     box-shadow: 0 2px 10px rgba(0,0,0,0.06); margin-bottom: 24px;
#   }
#   .row { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-end; }
#   .field { display: flex; flex-direction: column; gap: 6px; }
#   label { font-size: 13px; font-weight: 600; color: var(--navy); }
#   select, input[type="file"], input[type="range"] {
#     padding: 9px 10px; border-radius: 8px; border: 1px solid #d7dce3; font-size: 14px;
#   }
#   .dropzone {
#     border: 2px dashed #c3cbd6; border-radius: 12px; padding: 28px;
#     text-align: center; cursor: pointer; transition: border-color .15s;
#     flex: 1; min-width: 240px;
#   }
#   .dropzone.drag { border-color: var(--orange); background: #fff7f2; }
#   .dropzone img { max-height: 160px; border-radius: 8px; margin-top: 10px; }
#   .btn {
#     background: var(--orange); color: #fff; border: none; padding: 11px 20px;
#     border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 14px;
#   }
#   .btn:disabled { background: #c9b7ac; cursor: default; }
#   .btn.secondary { background: var(--navy); }
#   .results { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px,1fr)); gap: 16px; }
#   .card {
#     background: #fff; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 6px rgba(0,0,0,0.08);
#   }
#   .card img { width: 100%; height: 130px; object-fit: cover; display: block; }
#   .card .meta { padding: 8px 10px; font-size: 12px; }
#   .card .name { font-weight: 600; word-break: break-all; }
#   .card .score { color: var(--orange); font-weight: 700; }
#   .status { font-size: 13px; color: #556; margin-top: 8px; }
#   .no-match {
#     text-align: center; padding: 36px 20px; color: #778; font-size: 14px;
#   }
#   .no-match strong { color: var(--navy); display: block; margin-bottom: 6px; font-size: 16px; }
#   .hidden { display: none; }
#   .spinner {
#     width: 18px; height: 18px; border: 3px solid #eee; border-top-color: var(--orange);
#     border-radius: 50%; display: inline-block; animation: spin 0.8s linear infinite; vertical-align: middle;
#   }
#   @keyframes spin { to { transform: rotate(360deg); } }
# </style>
# </head>
# <body>
# <header>
#   <h1>Visual Search</h1>
#   <p>Upload a photo &mdash; find visually similar products by nearest-neighbor vector search</p>
# </header>

# <div class="wrap">
#   <div class="panel">
#     <div class="row">
#       <div class="dropzone" id="dropzone">
#         <div id="dzText">Upload image or capture photo<br><small>click / drag &amp; drop</small></div>
#         <img id="preview" class="hidden">
#         <input type="file" id="fileInput" accept="image/*" capture="environment" class="hidden">
#       </div>

#       <div class="field">
#         <label for="modelSelect">Embedding Model</label>
#         <select id="modelSelect">
#           <option value="multimodal">Multimodal Embedding Model (CLIP ViT-B/32)</option>
#           <option value="vision">Image Embedding Model (Vision Encoder, ResNet50)</option>
#         </select>
#       </div>

#       <div class="field">
#         <label for="topk">Results: <span id="topkVal">8</span></label>
#         <input type="range" id="topk" min="1" max="14" value="8">
#       </div>

#       <div class="field">
#         <label for="minScore">Match strictness: <span id="minScoreVal">72</span>%</label>
#         <input type="range" id="minScore" min="40" max="95" value="72">
#       </div>

#       <div class="field">
#         <button class="btn" id="searchBtn" disabled>Search</button>
#       </div>
#     </div>
#     <div class="status" id="status"></div>
#   </div>

#   <div class="panel">
#     <h3 style="margin-top:0;">Matching products list</h3>
#     <div class="results" id="results"></div>
#   </div>
# </div>

# <script>
# const dropzone = document.getElementById('dropzone');
# const dzText = document.getElementById('dzText');
# const preview = document.getElementById('preview');
# const fileInput = document.getElementById('fileInput');
# const searchBtn = document.getElementById('searchBtn');
# const statusEl = document.getElementById('status');
# const resultsEl = document.getElementById('results');
# const topk = document.getElementById('topk');
# const topkVal = document.getElementById('topkVal');
# const minScore = document.getElementById('minScore');
# const minScoreVal = document.getElementById('minScoreVal');
# const modelSelect = document.getElementById('modelSelect');

# let currentFile = null;

# topk.addEventListener('input', () => topkVal.textContent = topk.value);
# minScore.addEventListener('input', () => minScoreVal.textContent = minScore.value);
# dropzone.addEventListener('click', () => fileInput.click());

# ['dragover','dragenter'].forEach(ev => dropzone.addEventListener(ev, e => {
#   e.preventDefault(); dropzone.classList.add('drag');
# }));
# ['dragleave','drop'].forEach(ev => dropzone.addEventListener(ev, e => {
#   e.preventDefault(); dropzone.classList.remove('drag');
# }));
# dropzone.addEventListener('drop', e => {
#   if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
# });
# fileInput.addEventListener('change', e => {
#   if (e.target.files.length) handleFile(e.target.files[0]);
# });

# function handleFile(file) {
#   currentFile = file;
#   const reader = new FileReader();
#   reader.onload = e => {
#     preview.src = e.target.result;
#     preview.classList.remove('hidden');
#     dzText.classList.add('hidden');
#   };
#   reader.readAsDataURL(file);
#   searchBtn.disabled = false;
# }

# searchBtn.addEventListener('click', async () => {
#   if (!currentFile) return;
#   searchBtn.disabled = true;
#   statusEl.innerHTML = '<span class="spinner"></span> Embedding image and searching...';
#   resultsEl.innerHTML = '';

#   const fd = new FormData();
#   fd.append('image', currentFile);
#   fd.append('mode', modelSelect.value);
#   fd.append('top_k', topk.value);
#   fd.append('min_score', (parseFloat(minScore.value) / 100).toFixed(2));

#   try {
#     const res = await fetch('/search', { method: 'POST', body: fd });
#     const data = await res.json();
#     if (data.error) throw new Error(data.error);

#     if (!data.results.length) {
#       statusEl.textContent = `Checked all products in ${data.elapsed_ms} ms using ${data.mode_label} — none met the ${minScore.value}% strictness threshold.`;
#       resultsEl.innerHTML = `
#         <div class="no-match">
#           <strong>No match found</strong>
#           Nothing in the product catalog looks similar enough to this image.<br>
#           Try lowering "Match strictness", or add a closer product photo to <code>products/</code>.
#         </div>`;
#     } else {
#       statusEl.textContent = `Found ${data.results.length} matches in ${data.elapsed_ms} ms using ${data.mode_label}.`;
#       resultsEl.innerHTML = data.results.map(r => `
#         <div class="card">
#           <img src="${r.url}" alt="${r.file}">
#           <div class="meta">
#             <div class="name">${r.file}</div>
#             <div class="score">${(r.score*100).toFixed(1)}% match</div>
#           </div>
#         </div>
#       `).join('');
#     }
#   } catch (err) {
#     statusEl.textContent = 'Error: ' + err.message;
#   } finally {
#     searchBtn.disabled = false;
#   }
# });
# </script>
# </body>
# </html>
# """


# @app.route("/")
# def home():
#     return Response(HTML_PAGE, mimetype="text/html")


# @app.route("/products/<path:filename>")
# def product_file(filename):
#     return send_from_directory(PRODUCTS_DIR, filename)


# @app.route("/search", methods=["POST"])
# def do_search():
#     import time

#     mode = request.form.get("mode", "multimodal")
#     if mode not in ("multimodal", "vision"):
#         mode = "multimodal"
#     try:
#         top_k = max(1, min(20, int(request.form.get("top_k", 8))))
#     except ValueError:
#         top_k = 8

#     try:
#         # similarity threshold, 0.0-1.0 (cosine similarity on normalized
#         # CLIP/ResNet embeddings typically lands ~0.5-0.95 for genuine matches)
#         min_score = float(request.form.get("min_score", 0.72))
#     except ValueError:
#         min_score = 0.72
#     min_score = max(0.0, min(0.99, min_score))

#     file = request.files.get("image")
#     if file is None:
#         return jsonify({"error": "No image uploaded."}), 400

#     try:
#         t0 = time.time()
#         img = Image.open(io.BytesIO(file.read()))
#         query_emb = embed_image(img, mode)
#         files, embs = build_or_load_index(mode)
#         results = nearest_neighbors(query_emb, files, embs, top_k, min_score=min_score)
#         elapsed_ms = int((time.time() - t0) * 1000)

#         mode_label = "CLIP (multimodal)" if mode == "multimodal" else "ResNet50 (vision encoder)"
#         return jsonify(
#             {
#                 "mode_label": mode_label,
#                 "elapsed_ms": elapsed_ms,
#                 "min_score": min_score,
#                 "results": [
#                     {"file": f, "score": s, "url": f"/products/{f}"} for f, s in results
#                 ],
#             }
#         )
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500


# if __name__ == "__main__":
#     print(f"Device: {DEVICE}")
#     print("Pre-building product indexes (first run downloads model weights)...")
#     for m in ("multimodal", "vision"):
#         try:
#             f, e = build_or_load_index(m)
#             print(f"  [{m}] indexed {len(f)} product images -> {e.shape}")
#         except Exception as ex:
#             print(f"  [{m}] index build failed (will retry on first search): {ex}")

#     print("\nOpen http://localhost:5000 in your browser\n")
#     app.run(host="0.0.0.0", port=5000, debug=False)



# import os
# import io
# import json
# import traceback
 
# import numpy as np
# from PIL import Image
# from flask import Flask, request, jsonify, send_from_directory, Response
 
# import torch
 
# # --------------------------------------------------------------------------
# # Config
# # --------------------------------------------------------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PRODUCTS_DIR = os.path.join(BASE_DIR, "products")
# CACHE_DIR = os.path.join(BASE_DIR, "cache")
# os.makedirs(PRODUCTS_DIR, exist_ok=True)
# os.makedirs(CACHE_DIR, exist_ok=True)
 
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
 
# MODES = ("multimodal", "vision", "dino")
# MODE_LABELS = {
#     "multimodal": "CLIP (multimodal)",
#     "vision": "ResNet50 (vision encoder)",
#     "dino": "DINOv2 (self-supervised vision encoder)",
# }
# # Embedding dimensionality per mode, used only for the empty-catalog
# # fallback array so it has the right shape before anything is indexed.
# EMBED_DIMS = {"multimodal": 512, "vision": 2048, "dino": 768}
 
# _models = {}  # lazy-loaded model cache: {"vision": (...), "multimodal": (...), "dino": (...)}
 
 
# # --------------------------------------------------------------------------
# # Models
# # --------------------------------------------------------------------------
# def get_vision_model():
#     """Pure Image Embedding Model (Vision Encoder): ResNet50 pretrained on
#     ImageNet, classification head removed -> 2048-d visual feature vector."""
#     if "vision" not in _models:
#         import torchvision.models as tvm
#         import torchvision.transforms as T
 
#         weights = tvm.ResNet50_Weights.IMAGENET1K_V2
#         model = tvm.resnet50(weights=weights)
#         model.fc = torch.nn.Identity()
#         model.eval().to(DEVICE)
 
#         preprocess = T.Compose(
#             [
#                 T.Resize(256),
#                 T.CenterCrop(224),
#                 T.ToTensor(),
#                 T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#             ]
#         )
#         _models["vision"] = (model, preprocess)
#     return _models["vision"]
 
 
# def get_multimodal_model():
#     """Multimodal Embedding Model: OpenAI CLIP ViT-B/32 image tower.
#     Same embedding space as text, so this model could also match products
#     by a text query in the future -- here we use its image encoder."""
#     if "multimodal" not in _models:
#         import open_clip
 
#         model, _, preprocess = open_clip.create_model_and_transforms(
#             "ViT-B-32", pretrained="openai"
#         )
#         model.eval().to(DEVICE)
#         _models["multimodal"] = (model, preprocess)
#     return _models["multimodal"]
 
 
# def get_dino_model():
#     """Self-Supervised Vision Encoder: Meta DINOv2 ViT-B/14, trained without
#     any labels or text captions -> 768-d visual feature vector (CLS token).
#     Tends to be the strongest option for pure visual-similarity search
#     (as opposed to text-to-image search, where CLIP/SigLIP-style models
#     are the better fit)."""
#     if "dino" not in _models:
#         from transformers import AutoImageProcessor, AutoModel
 
#         processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
#         model = AutoModel.from_pretrained("facebook/dinov2-base")
#         model.eval().to(DEVICE)
#         _models["dino"] = (model, processor)
#     return _models["dino"]
 
 
# def embed_image(img: Image.Image, mode: str) -> np.ndarray:
#     img = img.convert("RGB")
#     with torch.no_grad():
#         if mode == "multimodal":
#             model, preprocess = get_multimodal_model()
#             tensor = preprocess(img).unsqueeze(0).to(DEVICE)
#             feat = model.encode_image(tensor)
#         elif mode == "dino":
#             model, processor = get_dino_model()
#             inputs = processor(images=img, return_tensors="pt")
#             inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
#             outputs = model(**inputs)
#             feat = outputs.last_hidden_state[:, 0, :]  # CLS token
#         else:  # "vision"
#             model, preprocess = get_vision_model()
#             tensor = preprocess(img).unsqueeze(0).to(DEVICE)
#             feat = model(tensor)
#         feat = feat / feat.norm(dim=-1, keepdim=True)
#     return feat.cpu().numpy()[0].astype(np.float32)
 
 
# # --------------------------------------------------------------------------
# # Product index (build once, cache to disk, rebuild if product list changes)
# # --------------------------------------------------------------------------
# def _index_path(mode):
#     return os.path.join(CACHE_DIR, f"index_{mode}.npz")
 
 
# def _list_product_files():
#     return sorted(f for f in os.listdir(PRODUCTS_DIR) if f.lower().endswith(IMG_EXTS))
 
 
# def build_or_load_index(mode: str):
#     files = _list_product_files()
#     path = _index_path(mode)
 
#     if os.path.exists(path):
#         data = np.load(path, allow_pickle=True)
#         cached_files = list(data["files"])
#         if cached_files == files:
#             return np.array(files), data["embs"]
 
#     embs = []
#     for f in files:
#         img_path = os.path.join(PRODUCTS_DIR, f)
#         try:
#             img = Image.open(img_path)
#             embs.append(embed_image(img, mode))
#         except Exception as e:
#             print(f"[warn] could not embed {f}: {e}")
 
#     dim = EMBED_DIMS.get(mode, 512)
#     embs_arr = np.stack(embs).astype(np.float32) if embs else np.zeros((0, dim), dtype=np.float32)
#     np.savez(path, files=np.array(files, dtype=object), embs=embs_arr)
#     return np.array(files), embs_arr
 
 
# def nearest_neighbors(
#     query_emb: np.ndarray,
#     files: np.ndarray,
#     embs: np.ndarray,
#     top_k: int,
#     min_score: float = 0.0,
# ):
#     """Return up to top_k nearest neighbors, but only those whose cosine
#     similarity is >= min_score. If min_score filters everything out, an
#     empty list is returned (caller shows a "no match" message) instead of
#     forcing back irrelevant items just to fill the list."""
#     if len(files) == 0:
#         return []
#     sims = embs @ query_emb  # cosine similarity, vectors are L2-normalized
#     order = np.argsort(-sims)
#     results = []
#     for i in order:
#         score = float(sims[i])
#         if score < min_score:
#             break  # order is descending, so nothing after this will pass either
#         results.append((str(files[i]), score))
#         if len(results) >= top_k:
#             break
#     return results
 
 
# # --------------------------------------------------------------------------
# # Flask app
# # --------------------------------------------------------------------------
# app = Flask(__name__)
 
# HTML_PAGE = """<!DOCTYPE html>
# <html lang="en">
# <head>
# <meta charset="UTF-8">
# <meta name="viewport" content="width=device-width, initial-scale=1.0">
# <title>Visual Search</title>
# <style>
#   :root {
#     --navy: #0B1E3D;
#     --orange: #FF6A2B;
#     --ice: #6EC6E8;
#     --bg: #F4F6F9;
#   }
#   * { box-sizing: border-box; }
#   body {
#     margin: 0; font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
#     background: var(--bg); color: var(--navy);
#   }
#   header {
#     background: var(--navy); color: #fff; padding: 22px 24px;
#   }
#   header h1 { margin: 0; font-size: 22px; }
#   header p { margin: 4px 0 0; color: var(--ice); font-size: 13px; }
#   .wrap { max-width: 1000px; margin: 0 auto; padding: 24px; }
#   .panel {
#     background: #fff; border-radius: 14px; padding: 22px;
#     box-shadow: 0 2px 10px rgba(0,0,0,0.06); margin-bottom: 24px;
#   }
#   .row { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-end; }
#   .field { display: flex; flex-direction: column; gap: 6px; }
#   label { font-size: 13px; font-weight: 600; color: var(--navy); }
#   select, input[type="file"], input[type="range"] {
#     padding: 9px 10px; border-radius: 8px; border: 1px solid #d7dce3; font-size: 14px;
#   }
#   .dropzone {
#     border: 2px dashed #c3cbd6; border-radius: 12px; padding: 28px;
#     text-align: center; cursor: pointer; transition: border-color .15s;
#     flex: 1; min-width: 240px;
#   }
#   .dropzone.drag { border-color: var(--orange); background: #fff7f2; }
#   .dropzone img { max-height: 160px; border-radius: 8px; margin-top: 10px; }
#   .btn {
#     background: var(--orange); color: #fff; border: none; padding: 11px 20px;
#     border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 14px;
#   }
#   .btn:disabled { background: #c9b7ac; cursor: default; }
#   .btn.secondary { background: var(--navy); }
#   .results { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px,1fr)); gap: 16px; }
#   .card {
#     background: #fff; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 6px rgba(0,0,0,0.08);
#   }
#   .card img { width: 100%; height: 130px; object-fit: cover; display: block; }
#   .card .meta { padding: 8px 10px; font-size: 12px; }
#   .card .name { font-weight: 600; word-break: break-all; }
#   .card .score { color: var(--orange); font-weight: 700; }
#   .status { font-size: 13px; color: #556; margin-top: 8px; }
#   .no-match {
#     text-align: center; padding: 36px 20px; color: #778; font-size: 14px;
#   }
#   .no-match strong { color: var(--navy); display: block; margin-bottom: 6px; font-size: 16px; }
#   .hidden { display: none; }
#   .spinner {
#     width: 18px; height: 18px; border: 3px solid #eee; border-top-color: var(--orange);
#     border-radius: 50%; display: inline-block; animation: spin 0.8s linear infinite; vertical-align: middle;
#   }
#   @keyframes spin { to { transform: rotate(360deg); } }
# </style>
# </head>
# <body>
# <header>
#   <h1>Visual Search</h1>
#   <p>Upload a photo &mdash; find visually similar products by nearest-neighbor vector search</p>
# </header>
 
# <div class="wrap">
#   <div class="panel">
#     <div class="row">
#       <div class="dropzone" id="dropzone">
#         <div id="dzText">Upload image or capture photo<br><small>click / drag &amp; drop</small></div>
#         <img id="preview" class="hidden">
#         <input type="file" id="fileInput" accept="image/*" capture="environment" class="hidden">
#       </div>
 
#       <div class="field">
#         <label for="modelSelect">Embedding Model</label>
#         <select id="modelSelect">
#           <option value="multimodal">Multimodal Embedding Model (CLIP ViT-B/32)</option>
#           <option value="vision">Image Embedding Model (Vision Encoder, ResNet50)</option>
#           <option value="dino">Self-Supervised Vision Encoder (DINOv2 ViT-B/14)</option>
#         </select>
#       </div>
 
#       <div class="field">
#         <label for="topk">Results: <span id="topkVal">8</span></label>
#         <input type="range" id="topk" min="1" max="14" value="8">
#       </div>
 
#       <div class="field">
#         <label for="minScore">Match strictness: <span id="minScoreVal">72</span>%</label>
#         <input type="range" id="minScore" min="40" max="95" value="72">
#       </div>
 
#       <div class="field">
#         <button class="btn" id="searchBtn" disabled>Search</button>
#       </div>
#     </div>
#     <div class="status" id="status"></div>
#   </div>
 
#   <div class="panel">
#     <h3 style="margin-top:0;">Matching products list</h3>
#     <div class="results" id="results"></div>
#   </div>
# </div>
 
# <script>
# const dropzone = document.getElementById('dropzone');
# const dzText = document.getElementById('dzText');
# const preview = document.getElementById('preview');
# const fileInput = document.getElementById('fileInput');
# const searchBtn = document.getElementById('searchBtn');
# const statusEl = document.getElementById('status');
# const resultsEl = document.getElementById('results');
# const topk = document.getElementById('topk');
# const topkVal = document.getElementById('topkVal');
# const minScore = document.getElementById('minScore');
# const minScoreVal = document.getElementById('minScoreVal');
# const modelSelect = document.getElementById('modelSelect');
 
# let currentFile = null;
 
# topk.addEventListener('input', () => topkVal.textContent = topk.value);
# minScore.addEventListener('input', () => minScoreVal.textContent = minScore.value);
# dropzone.addEventListener('click', () => fileInput.click());
 
# ['dragover','dragenter'].forEach(ev => dropzone.addEventListener(ev, e => {
#   e.preventDefault(); dropzone.classList.add('drag');
# }));
# ['dragleave','drop'].forEach(ev => dropzone.addEventListener(ev, e => {
#   e.preventDefault(); dropzone.classList.remove('drag');
# }));
# dropzone.addEventListener('drop', e => {
#   if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
# });
# fileInput.addEventListener('change', e => {
#   if (e.target.files.length) handleFile(e.target.files[0]);
# });
 
# function handleFile(file) {
#   currentFile = file;
#   const reader = new FileReader();
#   reader.onload = e => {
#     preview.src = e.target.result;
#     preview.classList.remove('hidden');
#     dzText.classList.add('hidden');
#   };
#   reader.readAsDataURL(file);
#   searchBtn.disabled = false;
# }
 
# searchBtn.addEventListener('click', async () => {
#   if (!currentFile) return;
#   searchBtn.disabled = true;
#   statusEl.innerHTML = '<span class="spinner"></span> Embedding image and searching...';
#   resultsEl.innerHTML = '';
 
#   const fd = new FormData();
#   fd.append('image', currentFile);
#   fd.append('mode', modelSelect.value);
#   fd.append('top_k', topk.value);
#   fd.append('min_score', (parseFloat(minScore.value) / 100).toFixed(2));
 
#   try {
#     const res = await fetch('/search', { method: 'POST', body: fd });
#     const data = await res.json();
#     if (data.error) throw new Error(data.error);
 
#     if (!data.results.length) {
#       statusEl.textContent = `Checked all products in ${data.elapsed_ms} ms using ${data.mode_label} — none met the ${minScore.value}% strictness threshold.`;
#       resultsEl.innerHTML = `
#         <div class="no-match">
#           <strong>No match found</strong>
#           Nothing in the product catalog looks similar enough to this image.<br>
#           Try lowering "Match strictness", or add a closer product photo to <code>products/</code>.
#         </div>`;
#     } else {
#       statusEl.textContent = `Found ${data.results.length} matches in ${data.elapsed_ms} ms using ${data.mode_label}.`;
#       resultsEl.innerHTML = data.results.map(r => `
#         <div class="card">
#           <img src="${r.url}" alt="${r.file}">
#           <div class="meta">
#             <div class="name">${r.file}</div>
#             <div class="score">${(r.score*100).toFixed(1)}% match</div>
#           </div>
#         </div>
#       `).join('');
#     }
#   } catch (err) {
#     statusEl.textContent = 'Error: ' + err.message;
#   } finally {
#     searchBtn.disabled = false;
#   }
# });
# </script>
# </body>
# </html>
# """
 
 
# @app.route("/")
# def home():
#     return Response(HTML_PAGE, mimetype="text/html")
 
 
# @app.route("/products/<path:filename>")
# def product_file(filename):
#     return send_from_directory(PRODUCTS_DIR, filename)
 
 
# @app.route("/search", methods=["POST"])
# def do_search():
#     import time
 
#     mode = request.form.get("mode", "multimodal")
#     if mode not in MODES:
#         mode = "multimodal"
#     try:
#         top_k = max(1, min(20, int(request.form.get("top_k", 8))))
#     except ValueError:
#         top_k = 8
 
#     try:
#         # similarity threshold, 0.0-1.0 (cosine similarity on normalized
#         # embeddings typically lands ~0.5-0.95 for genuine matches)
#         min_score = float(request.form.get("min_score", 0.72))
#     except ValueError:
#         min_score = 0.72
#     min_score = max(0.0, min(0.99, min_score))
 
#     file = request.files.get("image")
#     if file is None:
#         return jsonify({"error": "No image uploaded."}), 400
 
#     try:
#         t0 = time.time()
#         img = Image.open(io.BytesIO(file.read()))
#         query_emb = embed_image(img, mode)
#         files, embs = build_or_load_index(mode)
#         results = nearest_neighbors(query_emb, files, embs, top_k, min_score=min_score)
#         elapsed_ms = int((time.time() - t0) * 1000)
 
#         return jsonify(
#             {
#                 "mode_label": MODE_LABELS[mode],
#                 "elapsed_ms": elapsed_ms,
#                 "min_score": min_score,
#                 "results": [
#                     {"file": f, "score": s, "url": f"/products/{f}"} for f, s in results
#                 ],
#             }
#         )
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"error": str(e)}), 500
 
 
# if __name__ == "__main__":
#     print(f"Device: {DEVICE}")
#     print("Pre-building product indexes (first run downloads model weights)...")
#     for m in MODES:
#         try:
#             f, e = build_or_load_index(m)
#             print(f"  [{m}] indexed {len(f)} product images -> {e.shape}")
#         except Exception as ex:
#             print(f"  [{m}] index build failed (will retry on first search): {ex}")
 
#     print("\nOpen http://localhost:5000 in your browser\n")
#     app.run(host="0.0.0.0", port=5000, debug=False)


"""
Visual Search demo (WooCommerce-backed)
=========================================
1. Upload an image (or capture a photo from your camera on mobile).
2. The image is converted into a vector ("embedding") using one of two
   swappable models:
     - "multimodal" -> Multimodal Embedding Model (OpenAI CLIP ViT-B/32)
     - "dino"       -> Self-Supervised Vision Encoder (Meta DINOv2 ViT-B/14)
3. The product catalog is pulled live from your WooCommerce store's REST
   API (paginated, WC_PER_PAGE products per page, up to WC_MAX_PAGES pages)
   instead of a local folder. Each product's first image is downloaded and
   embedded the same way, cached to disk, and a nearest-neighbor cosine-
   similarity search returns the closest matching products.

Both models together are comfortably under 1 GB of weights:
  - CLIP ViT-B/32 (OpenAI)     ~ 350 MB
  - DINOv2 ViT-B/14 (Meta)     ~ 330 MB

WooCommerce REST API
---------------------
Credentials are read from environment variables -- never hardcode a live
consumer key/secret into this file, since it's easy to accidentally commit
or share a script like this.

    export WC_BASE_URL="https://wpemc.egrovetech.com"
    export WC_CONSUMER_KEY="ck_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    export WC_CONSUMER_SECRET="cs_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

Optional:
    export WC_PER_PAGE=10      # products per API page (WooCommerce default/your limit)
    export WC_MAX_PAGES=5      # how many pages to pull -> WC_PER_PAGE * WC_MAX_PAGES products indexed

If you get a 401 from the API: check WordPress -> WooCommerce -> Settings
-> Advanced -> REST API that the key still exists and has at least Read
access, and that nothing (Wordfence, a WAF, a CDN) is blocking external
requests to /wp-json/wc/v3/products.

Run it
-------
    python3 -m venv venv
    source venv/bin/activate          (Windows: venv\\Scripts\\activate)
    pip install -r requirements.txt   (needs: flask, torch, torchvision, open_clip_torch,
                                        transformers, pillow, numpy, requests)
    export WC_CONSUMER_KEY="..."
    export WC_CONSUMER_SECRET="..."
    python app.py
    open http://localhost:5000

First run will download the two model checkpoints (needs internet once);
model weights then work fully offline, but the product catalog and images
are always fetched live from your store, so that part still needs internet
every time the index is rebuilt.
"""

import os
import io
import traceback

import numpy as np
import requests
from PIL import Image
from flask import Flask, request, jsonify, Response

import torch

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
 
MODES = ("multimodal", "dino", "dino3")
MODE_LABELS = {
    "multimodal": "CLIP (multimodal)",
    "dino": "DINOv2 (self-supervised vision encoder)",
    "dino3": "DINOv3 (self-supervised vision encoder)",
}
# Embedding dimensionality per mode, used only for the empty-catalog
# fallback array so it has the right shape before anything is indexed.
EMBED_DIMS = {"multimodal": 512, "dino": 768, "dino3": 768}
 
DINOV3_MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
_models = {}  # lazy-loaded model cache: {"multimodal": (...), "dino": (...)}

# WooCommerce REST API
WC_BASE_URL = os.environ.get("WC_BASE_URL", "https://wpemc.egrovetech.com").rstrip("/")
WC_CONSUMER_KEY = os.environ.get("WC_CONSUMER_KEY", "ck_e097ddcc64c4670b62532b2fa3addb1d46d7d136")
WC_CONSUMER_SECRET = os.environ.get("WC_CONSUMER_SECRET", "cs_6689631b47559b3140f83bef64dfd41560f16216")
WC_PER_PAGE = int(os.environ.get("WC_PER_PAGE", "10"))
WC_MAX_PAGES = int(os.environ.get("WC_MAX_PAGES", "5"))
WC_TIMEOUT = 15  # seconds, for both the products list call and each image download

HF_TOKEN = os.environ.get("HF_TOKEN")
# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def get_multimodal_model():
    """Multimodal Embedding Model: OpenAI CLIP ViT-B/32 image tower.
    Same embedding space as text, so this model could also match products
    by a text query in the future -- here we use its image encoder."""
    if "multimodal" not in _models:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model.eval().to(DEVICE)
        _models["multimodal"] = (model, preprocess)
    return _models["multimodal"]


def get_dino_model():
    """Self-Supervised Vision Encoder: Meta DINOv2 ViT-B/14, trained without
    any labels or text captions -> 768-d visual feature vector (CLS token).
    Tends to be the strongest option for pure visual-similarity search
    (as opposed to text-to-image search, where CLIP-style models fit better)."""
    if "dino" not in _models:
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained("facebook/dinov2-base")
        model.eval().to(DEVICE)
        _models["dino"] = (model, processor)
    return _models["dino"]


def embed_image(img: Image.Image, mode: str) -> np.ndarray:
    img = img.convert("RGB")
    with torch.no_grad():
        if mode == "dino":
            model, processor = get_dino_model()
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)
            feat = outputs.last_hidden_state[:, 0, :]  # CLS token
        else:  # "multimodal"
            model, preprocess = get_multimodal_model()
            tensor = preprocess(img).unsqueeze(0).to(DEVICE)
            feat = model.encode_image(tensor)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy()[0].astype(np.float32)


# --------------------------------------------------------------------------
# WooCommerce product catalog
# --------------------------------------------------------------------------
def fetch_products():
    """Pull products from the WooCommerce REST API, paginating with
    WC_PER_PAGE per page until a page comes back empty or WC_MAX_PAGES
    is hit. Returns the raw list of WooCommerce product dicts."""
    if not WC_CONSUMER_KEY or not WC_CONSUMER_SECRET:
        raise RuntimeError(
            "WC_CONSUMER_KEY / WC_CONSUMER_SECRET are not set. "
            "Export them before running (see the module docstring)."
        )

    products = []
    for page in range(1, WC_MAX_PAGES + 1):
        resp = requests.get(
            f"{WC_BASE_URL}/wp-json/wc/v3/products",
            params={
                "consumer_key": WC_CONSUMER_KEY,
                "consumer_secret": WC_CONSUMER_SECRET,
                "per_page": WC_PER_PAGE,
                "page": page,
            },
            timeout=WC_TIMEOUT,
        )
        if resp.status_code == 401:
            raise RuntimeError(
                "WooCommerce API returned 401 Unauthorized -- check the key's "
                "permissions in WooCommerce > Settings > Advanced > REST API, "
                "and that nothing is blocking external requests to your store."
            )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        products.extend(batch)
    return products


def _product_records(products):
    """Reduce raw WooCommerce product dicts down to the fields the UI and
    embedding step actually need, skipping any product with no image."""
    records = []
    for p in products:
        images = p.get("images") or []
        if not images or not images[0].get("src"):
            continue
        records.append(
            {
                "id": str(p.get("id")),
                "name": p.get("name") or f"Product {p.get('id')}",
                "price": p.get("price") or "",
                "permalink": p.get("permalink") or "",
                "image_url": images[0]["src"],
            }
        )
    return records


def _download_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=WC_TIMEOUT)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content))


# --------------------------------------------------------------------------
# Embedding index (build once per mode, cache to disk, rebuild if the
# product catalog changes)
# --------------------------------------------------------------------------
def _index_path(mode):
    return os.path.join(CACHE_DIR, f"index_{mode}.npz")


def build_or_load_index(mode: str):
    records = _product_records(fetch_products())
    ids = [r["id"] for r in records]
    path = _index_path(mode)

    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        if list(data["ids"]) == ids:
            return records, data["embs"]

    embs = []
    kept_records = []
    for r in records:
        try:
            img = _download_image(r["image_url"])
            embs.append(embed_image(img, mode))
            kept_records.append(r)
        except Exception as e:
            print(f"[warn] could not embed product {r['id']} ({r['name']}): {e}")

    dim = EMBED_DIMS.get(mode, 512)
    embs_arr = np.stack(embs).astype(np.float32) if embs else np.zeros((0, dim), dtype=np.float32)
    np.savez(path, ids=np.array([r["id"] for r in kept_records], dtype=object), embs=embs_arr)
    return kept_records, embs_arr


def nearest_neighbors(
    query_emb: np.ndarray,
    records: list,
    embs: np.ndarray,
    top_k: int,
    min_score: float = 0.0,
):
    """Return up to top_k nearest neighbors, but only those whose cosine
    similarity is >= min_score. If min_score filters everything out, an
    empty list is returned (caller shows a "no match" message) instead of
    forcing back irrelevant items just to fill the list."""
    if len(records) == 0:
        return []
    sims = embs @ query_emb  # cosine similarity, vectors are L2-normalized
    order = np.argsort(-sims)
    results = []
    for i in order:
        score = float(sims[i])
        if score < min_score:
            break  # order is descending, so nothing after this will pass either
        results.append((records[i], score))
        if len(results) >= top_k:
            break
    return results


# --------------------------------------------------------------------------
# Flask app
# --------------------------------------------------------------------------
app = Flask(__name__)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Visual Search</title>
<style>
  :root {
    --navy: #0B1E3D;
    --orange: #FF6A2B;
    --ice: #6EC6E8;
    --bg: #F4F6F9;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, "Segoe UI", Roboto, Arial, sans-serif;
    background: var(--bg); color: var(--navy);
  }
  header {
    background: var(--navy); color: #fff; padding: 22px 24px;
  }
  header h1 { margin: 0; font-size: 22px; }
  header p { margin: 4px 0 0; color: var(--ice); font-size: 13px; }
  .wrap { max-width: 1000px; margin: 0 auto; padding: 24px; }
  .panel {
    background: #fff; border-radius: 14px; padding: 22px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.06); margin-bottom: 24px;
  }
  .row { display: flex; gap: 16px; flex-wrap: wrap; align-items: flex-end; }
  .field { display: flex; flex-direction: column; gap: 6px; }
  label { font-size: 13px; font-weight: 600; color: var(--navy); }
  select, input[type="file"], input[type="range"] {
    padding: 9px 10px; border-radius: 8px; border: 1px solid #d7dce3; font-size: 14px;
  }
  .dropzone {
    border: 2px dashed #c3cbd6; border-radius: 12px; padding: 28px;
    text-align: center; cursor: pointer; transition: border-color .15s;
    flex: 1; min-width: 240px;
  }
  .dropzone.drag { border-color: var(--orange); background: #fff7f2; }
  .dropzone img { max-height: 160px; border-radius: 8px; margin-top: 10px; }
  .btn {
    background: var(--orange); color: #fff; border: none; padding: 11px 20px;
    border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 14px;
  }
  .btn:disabled { background: #c9b7ac; cursor: default; }
  .btn.secondary { background: var(--navy); }
  .results { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px,1fr)); gap: 16px; }
  .card {
    background: #fff; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 6px rgba(0,0,0,0.08);
    display: block; text-decoration: none; color: inherit;
  }
  .card img { width: 100%; height: 130px; object-fit: cover; display: block; background: #eef1f5; }
  .card .meta { padding: 8px 10px; font-size: 12px; }
  .card .name { font-weight: 600; word-break: break-word; }
  .card .score { color: var(--orange); font-weight: 700; }
  .status { font-size: 13px; color: #556; margin-top: 8px; }
  .no-match {
    text-align: center; padding: 36px 20px; color: #778; font-size: 14px;
  }
  .no-match strong { color: var(--navy); display: block; margin-bottom: 6px; font-size: 16px; }
  .hidden { display: none; }
  .spinner {
    width: 18px; height: 18px; border: 3px solid #eee; border-top-color: var(--orange);
    border-radius: 50%; display: inline-block; animation: spin 0.8s linear infinite; vertical-align: middle;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<header>
  <h1>Visual Search</h1>
  <p>Upload a photo &mdash; find visually similar products from your store by nearest-neighbor vector search</p>
</header>

<div class="wrap">
  <div class="panel">
    <div class="row">
      <div class="dropzone" id="dropzone">
        <div id="dzText">Upload image or capture photo<br><small>click / drag &amp; drop</small></div>
        <img id="preview" class="hidden">
        <input type="file" id="fileInput" accept="image/*" capture="environment" class="hidden">
      </div>

      <div class="field">
        <label for="modelSelect">Embedding Model</label>
        <select id="modelSelect">
          <option value="multimodal">Multimodal Embedding Model (CLIP ViT-B/32)</option>
          <option value="dino">Self-Supervised Vision Encoder (DINOv2 ViT-B/14)</option>
        </select>
      </div>

      <div class="field">
        <label for="topk">Results: <span id="topkVal">8</span></label>
        <input type="range" id="topk" min="1" max="14" value="8">
      </div>

      <div class="field">
        <label for="minScore">Match strictness: <span id="minScoreVal">72</span>%</label>
        <input type="range" id="minScore" min="10" max="95" value="72">
      </div>

      <div class="field">
        <button class="btn" id="searchBtn" disabled>Search</button>
      </div>
    </div>
    <div class="status" id="status"></div>
  </div>
q                             
  <div class="panel">
    <h3 style="margin-top:0;">Matching products</h3>
    <div class="results" id="results"></div>
  </div>
</div>

<script>
const dropzone = document.getElementById('dropzone');
const dzText = document.getElementById('dzText');
const preview = document.getElementById('preview');
const fileInput = document.getElementById('fileInput');
const searchBtn = document.getElementById('searchBtn');
const statusEl = document.getElementById('status');
const resultsEl = document.getElementById('results');
const topk = document.getElementById('topk');
const topkVal = document.getElementById('topkVal');
const minScore = document.getElementById('minScore');
const minScoreVal = document.getElementById('minScoreVal');
const modelSelect = document.getElementById('modelSelect');

let currentFile = null;

topk.addEventListener('input', () => topkVal.textContent = topk.value);
minScore.addEventListener('input', () => minScoreVal.textContent = minScore.value);
dropzone.addEventListener('click', () => fileInput.click());

['dragover','dragenter'].forEach(ev => dropzone.addEventListener(ev, e => {
  e.preventDefault(); dropzone.classList.add('drag');
}));
['dragleave','drop'].forEach(ev => dropzone.addEventListener(ev, e => {
  e.preventDefault(); dropzone.classList.remove('drag');
}));
dropzone.addEventListener('drop', e => {
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', e => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

function handleFile(file) {
  currentFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    preview.src = e.target.result;
    preview.classList.remove('hidden');
    dzText.classList.add('hidden');
  };
  reader.readAsDataURL(file);
  searchBtn.disabled = false;
}

searchBtn.addEventListener('click', async () => {
  if (!currentFile) return;
  searchBtn.disabled = true;
  statusEl.innerHTML = '<span class="spinner"></span> Embedding image and searching your catalog...';
  resultsEl.innerHTML = '';

  const fd = new FormData();
  fd.append('image', currentFile);
  fd.append('mode', modelSelect.value);
  fd.append('top_k', topk.value);
  fd.append('min_score', (parseFloat(minScore.value) / 100).toFixed(2));

  try {
    const res = await fetch('/search', { method: 'POST', body: fd });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    if (!data.results.length) {
      statusEl.textContent = `Checked all products in ${data.elapsed_ms} ms using ${data.mode_label} — none met the ${minScore.value}% strictness threshold.`;
      resultsEl.innerHTML = `
        <div class="no-match">
          <strong>No match found</strong>
          Nothing in your store's catalog looks similar enough to this image.<br>
          Try lowering "Match strictness".
        </div>`;
    } else {
      statusEl.textContent = `Found ${data.results.length} matches in ${data.elapsed_ms} ms using ${data.mode_label}.`;
      resultsEl.innerHTML = data.results.map(r => `
        <a class="card" href="${r.permalink}" target="_blank" rel="noopener">
          <img src="${r.url}" alt="${r.name}">
          <div class="meta">
            <div class="name">${r.name}</div>
            <div class="score">${(r.score*100).toFixed(1)}% match${r.price ? ' &middot; $' + r.price : ''}</div>
          </div>
        </a>
      `).join('');
    }
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
  } finally {
    searchBtn.disabled = false;
  }
});
</script>
</body>
</html>
"""


@app.route("/")
def home():
    return Response(HTML_PAGE, mimetype="text/html")


@app.route("/search", methods=["POST"])
def do_search():
    import time

    mode = request.form.get("mode", "multimodal")
    if mode not in MODES:
        mode = "multimodal"
    try:
        top_k = max(1, min(20, int(request.form.get("top_k", 8))))
    except ValueError:
        top_k = 8

    try:
        # similarity threshold, 0.0-1.0 (cosine similarity on normalized
        # embeddings typically lands ~0.5-0.95 for genuine matches)
        min_score = float(request.form.get("min_score", 0.72))
    except ValueError:
        min_score = 0.72
    min_score = max(0.0, min(0.99, min_score))

    file = request.files.get("image")
    if file is None:
        return jsonify({"error": "No image uploaded."}), 400

    try:
        t0 = time.time()
        img = Image.open(io.BytesIO(file.read()))
        query_emb = embed_image(img, mode)
        records, embs = build_or_load_index(mode)
        results = nearest_neighbors(query_emb, records, embs, top_k, min_score=min_score)
        elapsed_ms = int((time.time() - t0) * 1000)

        return jsonify(
            {
                "mode_label": MODE_LABELS[mode],
                "elapsed_ms": elapsed_ms,
                "min_score": min_score,
                "results": [
                    {
                        "name": r["name"],
                        "price": r["price"],
                        "permalink": r["permalink"],
                        "url": r["image_url"],
                        "score": s,
                    }
                    for r, s in results
                ],
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    if not WC_CONSUMER_KEY or not WC_CONSUMER_SECRET:
        print(
            "[warn] WC_CONSUMER_KEY / WC_CONSUMER_SECRET not set -- searches will fail "
            "until you export them (see the module docstring)."
        )
    print("Pre-building product indexes (first run downloads model weights + product images)...")
    for m in MODES:
        try:
            recs, e = build_or_load_index(m)
            print(f"  [{m}] indexed {len(recs)} product images -> {e.shape}")
        except Exception as ex:
            print(f"  [{m}] index build failed (will retry on first search): {ex}")

    print("\nOpen http://localhost:5000 in your browser\n")
    app.run(host="0.0.0.0", port=5000, debug=False)