"""
Generate SHL vijay_kumar.csv and compute Mean Recall@10
Author: Vijay Kumar (AI Intern Assessment)
"""

import pandas as pd
import csv
import numpy as np
import os
from typing import List, Tuple


# ------------------------------
# 1️⃣  Mock retriever placeholder
# ------------------------------
# Replace this with your actual Retriever class (from retriever.retriever import Retriever)
# Ensure retriever.retrieve(query, top_k) returns a list of dicts with 'product' key containing 'url'

from rag_retriever.retriever import Retriever
class MockRetriever:
    """Temporary retriever for testing"""
    def __init__(self):
        # Load the real FAISS-based retriever
        self.retriever = Retriever()

    def retrieve(self, query: str, top_k: int = 10):
            # For now, return dummy URLs if you’re testing without FAISS
            return self.retriever.retrieve(query, top_k)
            #return [
               # {"product": {"url": f"https://www.shl.com/test/1-query-name{i}-query-{query.replace(' ', '-')}"}}
               # for i in range(1, top_k + 1)
           #python main.py
    #def retrieve(self, query: str, top_k: int = 10):
        # Dummy URLs for testing – replace with real retriever results
        #return [{"product": {"url": f"https://www.shl.com/test/1-query-name{i}-{query.replace(' ', '-')}"}} for i in
                #range(1, top_k + 1)]


# Initialize retriever
retriever = MockRetriever()

# ------------------------------
# 2️⃣  Config & helper functions
# ------------------------------
EXCEL_PATH = "Gen_AI Dataset.xlsx"  # Input Excel file
OUTPUT_CSV = "vijay_kumar.csv"  # SHL submission format file
TOP_K = 10  # Number of recommendations


def parse_relevant_urls(cell_value: str) -> List[str]:
    """Split comma/semicolon-separated URLs into list"""
    if pd.isna(cell_value) or not str(cell_value).strip():
        return []
    return [url.strip() for url in str(cell_value).replace(";", ",").split(",") if url.strip()]


def mean_recall_at_k(preds: List[List[str]], trues: List[List[str]], k: int = 10) -> float:
    """Compute Mean Recall@K for labeled data"""
    recalls = []
    for gt, pr in zip(trues, preds):
        if not gt:
            continue
        hits = sum(1 for url in gt if url in pr[:k])
        recalls.append(hits / len(gt))
    return round(float(np.mean(recalls)), 3) if recalls else 0.0


# ------------------------------
# 3️⃣  Load dataset
# ------------------------------
df = pd.read_excel(r"C:\Gen_AI Dataset.xlsx")
df.columns = [c.strip() for c in df.columns]  # Clean column names

# Expect columns: Query, Relevant_Assessment_URLs, Type
if "Query" not in df.columns:
    raise ValueError("Excel must have a 'Query' column")

if "Relevant_Assessment_URLs" not in df.columns:
    df["Relevant_Assessment_URLs"] = None
if "Type" not in df.columns:
    # Automatically mark as train/test based on presence of labels
    df["Type"] = df["Relevant_Assessment_URLs"].apply(lambda x: "train" if pd.notna(x) and str(x).strip() else "test")

# ------------------------------
# 4️⃣  Generate predictions
# ------------------------------
predictions_for_eval = []
ground_truths = []

rows_for_submission = []

for _, row in df.iterrows():
    query = str(row["Query"]).strip()
    true_urls = parse_relevant_urls(row["Relevant_Assessment_URLs"])
    preds = retriever.retrieve(query, top_k=TOP_K)
    pred_urls = [p["product"]["url"] for p in preds]

    # store for eval
    predictions_for_eval.append(pred_urls)
    ground_truths.append(true_urls)

    # Add to submission rows
    for url in pred_urls:
        rows_for_submission.append([query, url])

# ------------------------------
# 5️⃣  Compute Mean Recall@10
# ------------------------------
labeled_mask = df["Type"].eq("train")
recall_score = mean_recall_at_k(
    [p for p, m in zip(predictions_for_eval, labeled_mask) if m],
    [g for g, m in zip(ground_truths, labeled_mask) if m],
    k=TOP_K
)

print(f"\n✅ Mean Recall@{TOP_K}: {recall_score}")
print(f"Total labeled queries: {labeled_mask.sum()}")
print(f"Total test queries: {(~labeled_mask).sum()}")

# ------------------------------
# 6️⃣  Save vijay_kumar.csv (Appendix 3 format)
# ------------------------------
os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Query", "Assessment_url"])
    writer.writerows(rows_for_submission)

print(f"\n📁 Submission file generated successfully: {OUTPUT_CSV}")
print("   Format: Each query repeated for its top-10 Assessment URLs (Appendix 3).")
