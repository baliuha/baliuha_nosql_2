import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


load_dotenv()

INPUT_PARQUET = "data/arxiv_subset.parquet"
INPUT_EMBEDDINGS = "embeddings/embeddings.npy"
INDEX_NAME = "arxiv-papers"
VECTOR_DIM = 768
BATCH_SIZE = 200   # Pinecone recommends batches up to 200 vectors


def main():
    # init the Pinecone client
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index: '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=VECTOR_DIM,
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"  # standard free tier region
            )
        )
        # wait till index is created
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(1)
    else:
        print(f"Index '{INDEX_NAME}' already exists. Connecting...")

    # сonnect to the index
    index = pc.Index(INDEX_NAME)

    print("Loading data and embeddings into memory...")
    df = pd.read_parquet(INPUT_PARQUET)
    embeddings = np.load(INPUT_EMBEDDINGS)

    total_rows = len(df)
    print(f"Total records to process: {total_rows}")

    for i in tqdm(range(0, total_rows, BATCH_SIZE), desc="Upload to Pinecone"):
        batch_df = df.iloc[i: i + BATCH_SIZE]
        batch_emb = embeddings[i: i + BATCH_SIZE]

        vectors_to_upsert = []
        for j in range(len(batch_df)):
            row = batch_df.iloc[j]
            global_idx = i + j

            abstract_text = str(row.get("abstract", ""))[:500]
            authors_text = str(row.get("authors", ""))[:200]

            vector_id = f"paper_{global_idx}"
            metadata = {
                "arxiv_id": str(row.get("arxiv_id", "")),
                "title": str(row.get("title", "")),
                "abstract": abstract_text,
                "authors": authors_text,
                "year": int(row.get("year", 0)),
                "category": str(row.get("category", ""))
            }

            # NumPy arrays must be converted to Python lists for Pinecone
            vectors_to_upsert.append({
                "id": vector_id,
                "values": batch_emb[j].tolist(),
                "metadata": metadata
            })
        # insert or update the batch of vectors into Pinecone 
        index.upsert(vectors=vectors_to_upsert)

    stats = index.describe_index_stats()
    total_vectors = stats.total_vector_count
    print("Upload Completed!")
    print(f"Total vectors currently stored in '{INDEX_NAME}': {total_vectors}")


if __name__ == "__main__":
    main()
