from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pdfplumber
from pathlib import Path
import shutil
import pytesseract
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import uuid

app = FastAPI()

UPLOAD_DIR = Path("backend/data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Load model once on startup
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create FAISS index for 384-dim vectors (MiniLM)
dimension = 384
index = faiss.IndexFlatL2(dimension)

# Store metadata parallel to FAISS vectors
metadata_store = []

def get_theme_label_from_centroid(embeddings, texts):
    centroid = np.mean(embeddings, axis=0)
    sims = cosine_similarity([centroid], embeddings)[0]
    best_idx = np.argmax(sims)
    return texts[best_idx]

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in ["pdf", "jpg", "jpeg", "png"]:
        raise HTTPException(status_code=400, detail="Unsupported file format.")

    file_location = UPLOAD_DIR / file.filename
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return JSONResponse({"filename": file.filename, "message": "Uploaded successfully."})

@app.post("/extract_embed")
async def extract_and_embed(file: UploadFile = File(...)):
    # Save file temporarily
    temp_path = UPLOAD_DIR / file.filename
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    paragraphs = []

    # PDF text extraction
    if temp_path.suffix.lower() == ".pdf":
        with pdfplumber.open(temp_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text:
                    paras = [p.strip() for p in text.split('\n\n') if p.strip()]
                    for i, para in enumerate(paras, start=1):
                        paragraphs.append({
                            "text": para,
                            "page": page_num,
                            "paragraph_id": i
                        })

    # Image OCR extraction
    elif temp_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        # Open image with PIL
        image = Image.open(temp_path)
        text = pytesseract.image_to_string(image)
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text detected in image.")
        # Split text by paragraphs (double newline)
        paras = [p.strip() for p in text.split('\n\n') if p.strip()]
        for i, para in enumerate(paras, start=1):
            paragraphs.append({
                "text": para,
                "page": 1,  # Images don't have pages, default to 1
                "paragraph_id": i
            })

    else:
        raise HTTPException(status_code=400, detail="Unsupported file format for extraction.")

    # Generate embeddings for all paragraphs
    texts = [p["text"] for p in paragraphs]
    embeddings = model.encode(texts)

    # Add embeddings to FAISS index and metadata_store
    for i, embedding in enumerate(embeddings):
        index.add(np.array([embedding]))
        metadata_store.append({
            "file": file.filename,
            "page": paragraphs[i]["page"],
            "paragraph_id": paragraphs[i]["paragraph_id"],
            "text": paragraphs[i]["text"]
        })

    return {
        "message": "Text extracted and embedded successfully",
        "paragraphs": len(paragraphs)
    }

@app.get("/search")
async def search(query: str, top_k: int = 5):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), top_k)

    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        meta = metadata_store[idx]
        results.append({
            "score": float(dist),
            "text": meta["text"],
            "file": meta["file"],
            "page": meta["page"],
            "paragraph_id": meta["paragraph_id"]
        })
    return {"query": query, "results": results}



@app.get("/search_clustered")
async def search_clustered(query: str, top_k: int = 10, num_clusters: int = 3):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), top_k)

    valid_indices = [idx for idx in I[0] if idx != -1]
    if not valid_indices:
        return {"query": query, "themes": []}

    matched_embeddings = []
    matched_metadata = []

    for idx in valid_indices:
        matched_embeddings.append(model.encode([metadata_store[idx]["text"]])[0])
        matched_metadata.append(metadata_store[idx])

    kmeans = KMeans(n_clusters=min(num_clusters, len(matched_embeddings)), random_state=42)
    labels = kmeans.fit_predict(matched_embeddings)

    clustered_output = {}

    # Group embeddings and texts by cluster for labeling
    cluster_to_embeddings = {}
    cluster_to_texts = {}

    for label, meta, emb in zip(labels, matched_metadata, matched_embeddings):
        cluster_to_embeddings.setdefault(label, []).append(emb)
        cluster_to_texts.setdefault(label, []).append(meta["text"])

    for label in cluster_to_embeddings:
        theme_label = get_theme_label_from_centroid(
            np.array(cluster_to_embeddings[label]), 
            cluster_to_texts[label]
        )
        key = f"Theme {label + 1}: {theme_label[:100]}..."  # Short label preview

        if key not in clustered_output:
            clustered_output[key] = {
                "label": key,
                "results": []
            }

    # Append results under the labeled cluster
    for label, meta in zip(labels, matched_metadata):
        theme_key = f"Theme {label + 1}: {get_theme_label_from_centroid(np.array(cluster_to_embeddings[label]), cluster_to_texts[label])[:100]}..."
        clustered_output[theme_key]["results"].append({
            "text": meta["text"],
            "file": meta["file"],
            "page": meta["page"],
            "paragraph_id": meta["paragraph_id"]
        })

    themes = list(clustered_output.values())

    return {
        "query": query,
        "themes": themes
    }