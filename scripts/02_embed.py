import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


INPUT_FILE = "data/arxiv_subset.parquet"
OUTPUT_DIR = "embeddings"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "embeddings.npy")
MODEL_NAME = "allenai/specter2_base"


def main():
    print(f"Loading dataset from {INPUT_FILE}...")
    df = pd.read_parquet(INPUT_FILE)

    print("Preparing texts for encoding...")
    texts = (df["title"] + " [SEP] " + df["abstract"]).tolist()

    print(f"Loading model '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print("\nExecution Summary:")
    print(f"Total processed texts: {len(texts)}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Norm of the first embedding: {np.linalg.norm(embeddings[0]):.4f}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(OUTPUT_FILE, embeddings)
    print(f"\nEmbeddings successfully saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()