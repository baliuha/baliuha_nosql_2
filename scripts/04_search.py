import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer


load_dotenv()

INDEX_NAME = "arxiv-papers"
MODEL_NAME = "allenai/specter2_base"
TOP_K = 5
LOCAL_EMBEDDINGS_FILE = "embeddings/embeddings.npy"


def search_pinecone(query: str, title: str, model, index, df, filters=None) -> list:
    """Performs Pinecone search with optional filters and prints results"""

    print(f"\n{'='*80}\n{title}\n{'='*80}")
    query_emb = model.encode(query, normalize_embeddings=True).tolist()

    results = index.query(
        vector=query_emb,
        top_k=TOP_K,
        include_metadata=True,
        filter=filters
    )

    for i, match in enumerate(results['matches'], 1):
        meta = match['metadata']

        row_idx = int(match['id'].split('_')[1])
        abstract_text = str(df.iloc[row_idx]['abstract'])[:200] + "..."

        print(f"{i}. [Score: {match['score']:.4f}] {meta.get('title')}")
        print(f"ID: {match['id']} | Category: {meta.get('category')} | Year: {meta.get('year')}")
        print(f"Abstract: {abstract_text}\n")

    return query_emb  # embedding for local comparison later


def main():
    # init Pinecone, Model, and Dataset
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index(INDEX_NAME)
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

    # filtered search examples
    q_filter = "reinforcement learning"
    search_pinecone(q_filter, "FILTER A: RL (>= 2020, cs.LG)",
                    model, index, df,
                    filters={"year": {"$gte": 2020}, "category": {"$eq": "cs.LG"}})
    search_pinecone(q_filter, "FILTER B: RL (<= 2015, ANY CAT)",
                    model, index, df,
                    filters={"year": {"$lte": 2015}})

    # pure semantic search without filters
    q_pure = "teaching machines to recognize objects in pictures"
    q_emb = search_pinecone(q_pure, "PURE SEMANTIC SEARCH RESULTS",
                            model, index, df)
    q_vec = np.array(q_emb)
    local_emb = np.load(LOCAL_EMBEDDINGS_FILE)

    # calculate metrics
    dot_scores = np.dot(local_emb, q_vec)
    cosine_scores = dot_scores / (np.linalg.norm(local_emb, axis=1) * np.linalg.norm(q_vec))
    l2_distances = np.linalg.norm(local_emb - q_vec, axis=1)

    # get indices of Top K results
    top_dot = np.argsort(dot_scores)[-TOP_K:][::-1]
    top_cos = np.argsort(cosine_scores)[-TOP_K:][::-1]
    top_l2 = np.argsort(l2_distances)[:TOP_K]

    print(f"\n{'='*80}\nLOCAL METRICS COMPARISON\n{'='*80}")
    print(f"Query: '{q_pure}'\n")
    print(f"{'Rank':<5} | {'Dot Product ID':<20} | {'Cosine ID':<20} | {'L2 Distance ID':<20}")
    print("-" * 80)
    for i in range(TOP_K):
        print(f"{i+1:<5} | paper_{top_dot[i]:<6} ({dot_scores[top_dot[i]]:.3f}) | "
              f"paper_{top_cos[i]:<6} ({cosine_scores[top_cos[i]]:.3f}) | "
              f"paper_{top_l2[i]:<6} ({l2_distances[top_l2[i]]:.3f})")


if __name__ == "__main__":
    main()
