from sentence_transformers import SentenceTransformer
from pathlib import Path
import pandas as pd
import numpy as np


"""
Purpose: Generates the embeddings for the new user data. 
Saves the embeddings in-> (data/new_user) -> (user_data.npz)
"""

text_col="Text"
student_col="Student"
output_name="user_data.npz"
model_name="google/embeddinggemma-300m"

repo_root = Path(__file__).resolve().parents[2]
data_dir = repo_root / "data" / "new_user"
csv_files = sorted(data_dir.glob("*.csv"))


dfs = []
for f in csv_files:
    df = pd.read_csv(f)

    if text_col not in df.columns:
        raise ValueError(f"{text_col!r} not found in {f}")
    if student_col not in df.columns:
        raise ValueError(f"{student_col!r} not found in {f}")

    # Keep only rows with both student and text present
    df = df[[student_col, text_col]].copy()
    df = df.dropna(subset=[student_col, text_col])

    # Normalize to strings
    df[student_col] = df[student_col].astype(str).str.strip()
    df[text_col] = df[text_col].astype(str)

    dfs.append(df)

all_df = pd.concat(dfs, ignore_index=True)

if len(all_df) == 0:
    raise ValueError("No valid rows found after dropping missing student/text values.")

    # ---------------------------------------------------
    # 2. Determine unique students = entities (J)
    # ---------------------------------------------------
    # Sorted for reproducible ordering
students = sorted(all_df[student_col].unique().tolist())
J = len(students)

student_to_idx = {student: j for j, student in enumerate(students)}

print(f"Found {J} unique students:")
print(students)

    # ---------------------------------------------------
    # 3. Define timesteps = each row is one timestep
    # ---------------------------------------------------
T = len(all_df)
utter_texts = all_df[text_col].tolist()
utter_speakers = all_df[student_col].tolist()

    # ---------------------------------------------------
    # 4. Load model
    # ---------------------------------------------------
model = SentenceTransformer(model_name)

    # ---------------------------------------------------
    # 5. Embed all utterances + silence
    # ---------------------------------------------------
silence_text = "silence"

# Embed all utterances in one batch
utter_embeddings = model.encode(
    utter_texts,
    normalize_embeddings=False,
    convert_to_numpy=True,
)

    # Embed silence once
silence_embedding = model.encode(
    [silence_text],
    normalize_embeddings=False,
    convert_to_numpy=True,
)[0]

# Keep first 128 dims
utter_embeddings = utter_embeddings[:, :128]
silence_embedding = silence_embedding[:128]

    # L2 normalize
utter_norms = np.linalg.norm(utter_embeddings, axis=1, keepdims=True)
utter_norms = np.clip(utter_norms, 1e-12, None)
utter_embeddings = utter_embeddings / utter_norms

silence_norm = np.linalg.norm(silence_embedding)
silence_norm = max(silence_norm, 1e-12)
silence_embedding = silence_embedding / silence_norm

# ---------------------------------------------------
# 6. Build TxJx128 tensor
# ---------------------------------------------------
D = 128
X = np.tile(silence_embedding[None, None, :], (T, J, 1))

speaker_indices = np.zeros(T, dtype=np.int64)

for t, speaker in enumerate(utter_speakers):
    j = student_to_idx[speaker]
    speaker_indices[t] = j
    X[t, j, :] = utter_embeddings[t]

# ---------------------------------------------------
# 7. Save dataset
# ---------------------------------------------------
output_path = data_dir / output_name

np.savez_compressed(
    output_path,
    observations=X,  # shape (T, J, 128)
    texts=np.array(utter_texts, dtype=object),          # length T
    speakers=np.array(utter_speakers, dtype=object),    # length T
    speaker_indices=speaker_indices,                    # length T
    student_names=np.array(students, dtype=object),     # length J
    silence_embedding=silence_embedding,                # shape (128,)
    model_name=np.array(model_name, dtype=str),
    normalized=np.array(True),
    emb_dim=np.array(D),
    T=np.array(T),
    J=np.array(J),
)

print(f"Saved dataset to: {output_path}")
print(f"observations shape: {X.shape}")

 


