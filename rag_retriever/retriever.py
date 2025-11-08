# rag_retriever/retriever.py
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

INDEX_PATH = "data/indexes/faiss_index.bin"
METADATA_PATH = "data/indexes/metadata.pkl"

class Retriever:
    def __init__(self):
        print("🔍 Loading FAISS index and metadata...")
        self.index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            self.products = pickle.load(f)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Retriever ready!")

    def retrieve(self, query: str, top_k: int = 10):
        q_emb = self.model.encode([query])
        D, I = self.index.search(np.array(q_emb, dtype=np.float32), top_k)
        results = []
        for idx in I[0]:
            prod = self.products[idx]
            results.append({"product": prod})
        return results
