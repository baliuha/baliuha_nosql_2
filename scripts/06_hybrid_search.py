import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

load_dotenv()


def print_top5(method_name: str, results: list, df: pd.DataFrame):
    print(f"\n{'='*30}{method_name} Top 5{'='*30}")
    for i, (paper_id, score) in enumerate(results[:5], 1):
        idx = int(paper_id.split('_')[1])
        row = df.iloc[idx]
        print(f"{i}. [Score: {score:.4f}] {row['title']}")
        print(f" Abstract: {str(row['abstract'])[:150]}...\n")


def main():
    index = Pinecone(api_key=os.environ["PINECONE_API_KEY"]).Index("arxiv-papers")
    model = SentenceTransformer("allenai/specter2_base", device="cuda")
    df = pd.read_parquet("data/arxiv_subset.parquet").reset_index(drop=True)

    # BM25 Index
    print("Building BM25 index...")
    corpus = (df["title"].fillna("") + " " + df["abstract"].fillna("")).str.lower().str.split()
    bm25 = BM25Okapi(corpus.tolist())

    queries = [
        "BERT fine-tuning",
        "Yann LeCun convolutional networks",
        "making computers understand human emotions from text"
    ]

    for q in queries:
        print(f"\n{'='*70}\nQUERY: '{q}'\n{'='*70}")

        # BM25 Search
        q_tokens = q.lower().split()
        bm25_scores = bm25.get_scores(q_tokens)
        top_bm25_idx = np.argsort(bm25_scores)[-10:][::-1] # Get top 10
        # matches with score > 0
        bm25_res = [(f"paper_{i}", bm25_scores[i]) for i in top_bm25_idx if bm25_scores[i] > 0]

        # Vector Search (Pinecone)
        q_emb = model.encode(q, normalize_embeddings=True).tolist()
        vec_data = index.query(vector=q_emb, top_k=10)['matches']
        vec_res = [(match['id'], match['score']) for match in vec_data]

        # Hybrid Search (RRF Formula)
        rrf_scores = {}
        # score = 1 / (60 + rank)
        for rank, (pid, _) in enumerate(bm25_res, 1):
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (60 + rank))
        for rank, (pid, _) in enumerate(vec_res, 1):
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (1.0 / (60 + rank))

        # sort hybrid results
        hybrid_res = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        print_top5("BM25", bm25_res, df)
        print_top5("VECTOR", vec_res, df)
        print_top5("HYBRID (RRF)", hybrid_res, df)


if __name__ == "__main__":
    main()
