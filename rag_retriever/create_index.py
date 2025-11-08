"""
Create FAISS index for SHL product catalog
"""

import os
import pickle
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -----------------------------
# File paths
# -----------------------------
CATALOG_PATH = "data/raw/shl_catalog.csv"
INDEX_DIR = "data/indexes"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.pkl")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.split())


def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    # Load catalog
    print("📂 Loading SHL catalog...")
    df = pd.read_csv(CATALOG_PATH)
    print(f"✅ Loaded {len(df)} records")

    # Prepare text corpus
    docs = []
    for _, row in df.iterrows():
        doc = f"{row.get('Assessment Name', '')}. {row.get('Category', '')}. {row.get('Description', '')}"
        docs.append(clean_text(doc))

    # Embedding model
    print("⚙️ Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("🔢 Generating embeddings...")
    embeddings = model.encode(docs, show_progress_bar=True, convert_to_numpy=True)

    # Build FAISS index
    print("💾 Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))

    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(df.to_dict(orient="records"), f)

    print(f"✅ FAISS index saved to: {INDEX_PATH}")
    print(f"✅ Metadata saved to: {METADATA_PATH}")
    print("🎉 Index creation complete!")


if __name__ == "__main__":
    main()
