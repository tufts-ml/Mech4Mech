from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import numpy as np


"""
Purpose: Generates the embeddings for the training data for the feedback mechanism and saves them with the corresponding labels. 
Saves the embeddings and labels in (data) -> (feedback_mechanism) -> (trainind_data.npz)
"""

repo_root = Path(__file__).resolve().parents[2]
data_dir = repo_root / "data" / "feedback_mechanism"
csv_files = sorted(data_dir.glob("*.csv"))
text_col = "Text"
student_col = "Student"
hf_access_token = "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"
all_texts = []
all_labels = []

for f in csv_files:
    df = pd.read_csv(f)
    texts = df[text_col].dropna().astype(str).tolist()
    label_cols = [c for c in df.columns if c != text_col and c != student_col]
    labels = df[label_cols].astype(int).values.tolist()
    all_texts.extend(texts)
    all_labels.extend(labels)

model = SentenceTransformer("google/embeddinggemma-300m", token=hf_access_token)
embeddings = model.encode(all_texts, normalize_embeddings=False)
emb_128 = embeddings[:, :128]
norms = np.linalg.norm(emb_128, axis=1, keepdims=True)
emb_128 = emb_128 / norms

path = data_dir/"training_data.npz"

np.savez_compressed(
    path,
    embeddings=emb_128,
    labels=all_labels,
    texts=np.array(all_texts, dtype=object),     # keep exact strings
    label_names=np.array(["No evidence", "Target phenomena", "Set-up conditions", "Entities", "Activities", "Properties", "Organization", "Chaining"], dtype=object),
    model_name=np.array("google/embeddinggemma-300m", dtype=str),
    normalized=np.array(True),
    emb_dim=np.array(emb_128.shape[1]),
    )