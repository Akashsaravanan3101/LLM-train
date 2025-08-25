import os, glob, json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

KB_DIR = os.path.join(os.path.dirname(__file__), "..", "knowledge_base")
KB_DIR = os.path.abspath(KB_DIR)
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
OUT_DIR = os.path.abspath(OUT_DIR)

os.makedirs(OUT_DIR, exist_ok=True)

def read_text(path):
    try:
        with open(path, "rb") as f:
            return f.read().decode(errors="ignore")
    except Exception as e:
        print(f"Skipping {path}: {e}")
        return ""

def chunk_text(text, size=1200, overlap=100):
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks

def main():
    paths = []
    for ext in ("*.txt","*.md","*.html","*.htm","*.pdf"):  # we will treat pdf as text; real use: extract properly
        paths.extend(glob.glob(os.path.join(KB_DIR, "**", ext), recursive=True))
    if not paths:
        print("No KB files found, add documents to knowledge_base/")
        return

    model = SentenceTransformer("all-MiniLM-L6-v2")
    docs = []
    for p in paths:
        text = read_text(p)
        if not text.strip():
            continue
        for idx, c in enumerate(chunk_text(text)):
            docs.append({"id": f"kb://{os.path.basename(p)}#{idx}", "text": c})

    if not docs:
        print("No text chunks created. Check your knowledge_base files.")
        return

    emb = model.encode([d["text"] for d in docs], convert_to_numpy=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    faiss.write_index(index, os.path.join(OUT_DIR, "kb.index"))
    with open(os.path.join(OUT_DIR, "kb_docs.json"), "w") as f:
        json.dump(docs, f)
    print(f"Wrote index and docs to {OUT_DIR}")

if __name__ == "__main__":
    main()
