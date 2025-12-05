from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

#GENERATES THE TRAINING AND TEST DATA FOR UNSUPERVISED INFERENCE. 

def get_training_data(
    max_J: int = 4,
):
    """
    Load training CSVs and produce tensors of shape (T, max_J, D) per file. Returns all data X of shape (N, T, max_J, D).

    For each file:
      - Let J_file = # unique students in that file.
      - Real students occupy indices [0, J_file-1].
      - Remaining [J_file, max_J-1] slots are filled with the 'silence' embedding.
      - Labels are one-hot over C classes.
        * Speaking students get their one-hot label from the CSV.
        * Silent/padded students get one-hot 'No evidence'.

    Returns
    -------
    all_X : list of np.ndarray
        One entry per training CSV, each of shape (T, max_J, D).
    all_Y : list of np.ndarray
        One entry per training CSV, each of shape (T, max_J, C).
    all_students : list of list
        One entry per training CSV, each a list of length max_J with student
        identifiers for real students and placeholder names for padded slots.
    example_end_times: list 
        Segment end times (length T), starts with a -1 in the first index. 
    """

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "unsupervised_inference" / "training"
    csv_files = sorted(data_dir.glob("*.csv"))

    text_col = "Text"
    student_col = "Student"
    hf_access_token = "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"

    model = SentenceTransformer("google/embeddinggemma-300m", token=hf_access_token)

    # --- Precompute normalized "silence" embedding ---
    silence_emb = model.encode(["silence"], normalize_embeddings=False)
    silence_emb_128 = silence_emb[:, :128]
    silence_emb_128 /= np.linalg.norm(silence_emb_128, axis=1, keepdims=True)  # (1, D)

    all_X = []   # list of arrays, each (T, max_J, D)
    all_Y = []   # list of arrays, each (T, max_J, C)
    all_students = []
    example_end_times = [-1]
    no_evidence_label: str = "No evidence"
   
    for f in csv_files:
        df = pd.read_csv(f)

        texts = df[text_col].fillna("").astype(str).tolist()
        speakers = df[student_col].astype(str).tolist()

        # Label columns and 'No evidence' index
        label_cols = [c for c in df.columns if c not in (text_col, student_col)]
        assert no_evidence_label in label_cols, \
            f"'{no_evidence_label}' must be one of {label_cols}"
        no_evidence_idx = label_cols.index(no_evidence_label)

        # Labels per utterance: (T, C), assumed one-hot
        labels_per_utterance = df[label_cols].astype(int).values
        T, C = labels_per_utterance.shape

        # Sanity check one-hot
        row_sums = labels_per_utterance.sum(axis=1)
        if not np.all(row_sums == 1):
            bad_rows = np.where(row_sums != 1)[0]
            raise ValueError(
                f"Non one-hot label rows in {f.name} at indices {bad_rows}. "
                f"Each row over {label_cols} must have exactly one '1'."
            )

        # Students in this file
        students = sorted(df[student_col].astype(str).unique())
        J_file = len(students)
        if J_file > max_J:
            raise ValueError(
                f"File {f.name} has {J_file} students, which exceeds max_J={max_J}."
            )

        student_to_idx = {s: j for j, s in enumerate(students)}

        # Utterance embeddings: (T, D), normalized
        embeddings = model.encode(texts, normalize_embeddings=False)
        emb_128 = embeddings[:, :128]
        emb_128 /= np.linalg.norm(emb_128, axis=1, keepdims=True)

        # --- Initialize X: (T, max_J, D) with silence everywhere ---
        X = np.tile(silence_emb_128, (T, max_J, 1))

        # --- Initialize Y: (T, max_J, C) with one-hot 'No evidence' everywhere ---
        Y = np.zeros((T, max_J, C), dtype=int)
        Y[:, :, no_evidence_idx] = 1

        # --- Fill in actual students in [0, J_file-1] ---
        for t, speaker in enumerate(speakers):
            j = student_to_idx[speaker]  # 0 <= j < J_file <= max_J
            X[t, j, :] = emb_128[t]
            Y[t, j, :] = labels_per_utterance[t, :]

        # Pad student list to length max_J for bookkeeping
        padded_students = list(students)
        while len(padded_students) < max_J:
            padded_students.append(f"PAD_STUDENT_{len(padded_students)}")

        all_X.append(X)
        all_Y.append(Y)
        all_students.append(padded_students)
        if len(example_end_times) == 1: 
            example_end_times.append(T)
        else: 
            example_end_times.append(T + example_end_times[-1])

    return all_X, all_Y, all_students, example_end_times


def get_test_data(
    max_J: int = 4,
):
    """
    Load test CSVs and produce padded tensors of shape (T, max_J, D).

    For each file:
      - Actual students (J_file <= max_J) get their utterance embeddings.
      - Remaining (max_J - J_file) student slots are filled with the
        'silence' embedding.
      - Labels are one-hot over C classes; silent/padded students get
        one-hot 'No evidence'.

    Returns
    -------
    all_X : list of np.ndarray
        One entry per test CSV, each of shape (T, max_J, D).
    all_Y : list of np.ndarray
        One entry per test CSV, each of shape (T, max_J, C).
    all_students : list of list
        One entry per test CSV, each a list of length max_J with student
        identifiers for real students and placeholder names for padded slots.
    example_end_times : list 
        The index of the ending of each segment. Should be length T. 
    """

    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / "data" / "unsupervised_inference" / "test"
    csv_files = sorted(data_dir.glob("*.csv"))

    text_col = "Text"
    student_col = "Student"
    hf_access_token = "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"

    model = SentenceTransformer("google/embeddinggemma-300m", token=hf_access_token)

    # --- Precompute normalized "silence" embedding ---
    silence_emb = model.encode(["silence"], normalize_embeddings=False)
    silence_emb_128 = silence_emb[:, :128]
    silence_emb_128 /= np.linalg.norm(silence_emb_128, axis=1, keepdims=True)  # (1, D)

    all_X = []   # (T, max_J, D)
    all_Y = []   # (T, max_J, C)
    all_students = []  # list of length max_J per file
    no_evidence_label: str = "No evidence"
    example_end_times = [-1]

    for f in csv_files:
        df = pd.read_csv(f)

        texts = df[text_col].fillna("").astype(str).tolist()
        speakers = df[student_col].astype(str).tolist()

        # Label columns and 'No evidence' index
        label_cols = [c for c in df.columns if c not in (text_col, student_col)]
        assert no_evidence_label in label_cols, \
            f"'{no_evidence_label}' must be one of {label_cols}"
        no_evidence_idx = label_cols.index(no_evidence_label)

        # Labels per utterance: (T, C), assumed one-hot
        labels_per_utterance = df[label_cols].astype(int).values
        T, C = labels_per_utterance.shape

        # Optional: sanity check one-hot
        row_sums = labels_per_utterance.sum(axis=1)
        if not np.all(row_sums == 1):
            bad_rows = np.where(row_sums != 1)[0]
            raise ValueError(
                f"Non one-hot label rows in {f.name} at indices {bad_rows}. "
                f"Each row over {label_cols} must have exactly one '1'."
            )

        # Students actually present in this test file
        students = sorted(df[student_col].astype(str).unique())
        J_file = len(students)
        if J_file > max_J:
            raise ValueError(
                f"File {f.name} has {J_file} students, which exceeds max_J={max_J}."
            )

        student_to_idx = {s: j for j, s in enumerate(students)}

        # Embeddings for utterances: (T, D), normalized
        embeddings = model.encode(texts, normalize_embeddings=False)
        emb_128 = embeddings[:, :128]
        emb_128 /= np.linalg.norm(emb_128, axis=1, keepdims=True)

        # --- Initialize X: (T, max_J, D) with silence everywhere ---
        X = np.tile(silence_emb_128, (T, max_J, 1))

        # --- Initialize Y: (T, max_J, C) with one-hot 'No evidence' everywhere ---
        Y = np.zeros((T, max_J, C), dtype=int)
        Y[:, :, no_evidence_idx] = 1

        # --- Fill in actual students in the first J_file slots ---
        for t, speaker in enumerate(speakers):
            j = student_to_idx[speaker]  # 0 <= j < J_file <= max_J
            X[t, j, :] = emb_128[t]
            Y[t, j, :] = labels_per_utterance[t, :]

        # Pad student list to length max_J (for bookkeeping)
        padded_students = list(students)
        while len(padded_students) < max_J:
            padded_students.append(f"PAD_STUDENT_{len(padded_students)}")

        all_X.append(X)
        all_Y.append(Y)
        all_students.append(padded_students)
        if len(example_end_times) == 1: 
            example_end_times.append(T)
        else: 
            example_end_times.append(T + example_end_times[-1])

    return all_X, all_Y, all_students, example_end_times

