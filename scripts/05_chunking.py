import os
import re
import time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer


load_dotenv()

MODEL_NAME = "allenai/specter2_base"
VECTOR_DIM = 768
INDEX_FIXED = "arxiv-chunks-fixed"
INDEX_SEMANTIC = "arxiv-chunks-semantic"
BATCH_SIZE = 100


def setup_pinecone_index(pc: Pinecone, index_name: str):
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if index_name not in existing_indexes:
        print(f"Creating index '{index_name}'...")
        pc.create_index(
            name=index_name,
            dimension=VECTOR_DIM,
            metric="dotproduct",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
    else:
        print(f"Index '{index_name}' already exists.")
    return pc.Index(index_name)


def fixed_size_chunking(text: str, chunk_size=50, overlap=10) -> list:
    words = text.split()
    chunks = []
    # step by (chunk_size - overlap) to ensure the sliding window
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i: i + chunk_size])
        chunks.append(chunk)
        # stop if we've reached the end of the text
        if i + chunk_size >= len(words):
            break
    return chunks


def semantic_chunking(text: str, max_words=50) -> list:
    # split by punctuation followed by a space
    sentences = re.split(r'(?<=[.?!])\s+', text.strip())
    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        if not sentence:
            continue
        sentence_word_count = len(sentence.split())

        if current_word_count + sentence_word_count > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_word_count = sentence_word_count
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_word_count

    # the last remaining chunk
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks


def process_and_upload_chunks(df: pd.DataFrame, index, model, strategy_func, chunk_prefix: str):
    vectors_to_upsert = []

    print(f"\nProcessing chunks using {strategy_func.__name__}...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row.get('abstract', ''))
        chunks = strategy_func(text)

        for i, chunk_text in enumerate(chunks):
            # encode chunk (L2 normalized for dotproduct)
            embedding = model.encode(chunk_text, normalize_embeddings=True).tolist()

            metadata = {
                "arxiv_id": str(row.get("arxiv_id", "")),
                "title": str(row.get("title", ""))[:200],  # truncate
                "chunk_text": chunk_text[:2000],  # Pinecone metadata size limit
                "chunk_idx": int(i),
                "year": int(row.get("year", 0)),
                "category": str(row.get("category", ""))
            }

            vectors_to_upsert.append({
                "id": f"{chunk_prefix}_{idx}_chunk_{i}",
                "values": embedding,
                "metadata": metadata
            })

            # upload in batches
            if len(vectors_to_upsert) >= BATCH_SIZE:
                index.upsert(vectors=vectors_to_upsert)
                vectors_to_upsert = []

    # upload remaining vectors
    if vectors_to_upsert:
        index.upsert(vectors=vectors_to_upsert)


def perform_search(query: str, index, model, title: str):
    print(f"\n{'='*80}\n{title}\n{'='*80}")
    print(f"Query: '{query}'\n")

    query_emb = model.encode(query, normalize_embeddings=True).tolist()
    results = index.query(vector=query_emb, top_k=5, include_metadata=True)

    for i, match in enumerate(results['matches'], 1):
        meta = match['metadata']
        chunk_preview = meta.get('chunk_text', '')[:150].replace('\n', ' ') + "..."
        print(f"{i}. [Score: {match['score']:.4f}] {meta.get('title')}")
        print(f" [Chunk #{meta.get('chunk_idx')}] {chunk_preview}\n")


def main():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    model = SentenceTransformer(MODEL_NAME, device="cuda")
    df = pd.read_parquet("data/arxiv_subset.parquet")

    print("Selecting top 30 articles with the longest abstracts...")
    df['abstract_len'] = df['abstract'].fillna("").apply(lambda x: len(x.split()))
    top_30_df = df.nlargest(30, 'abstract_len').reset_index(drop=True)

    index_fixed = setup_pinecone_index(pc, INDEX_FIXED)
    index_semantic = setup_pinecone_index(pc, INDEX_SEMANTIC)

    process_and_upload_chunks(top_30_df, index_fixed, model, fixed_size_chunking, "fixed")
    process_and_upload_chunks(top_30_df, index_semantic, model, semantic_chunking, "semantic")

    print("\nUpload Completed!")

    test_query = "novel techniques in artificial neural networks and deep learning"
    perform_search(test_query, index_fixed, model, "SEARCH IN FIXED-SIZE CHUNKS")
    perform_search(test_query, index_semantic, model, "SEARCH IN SEMANTIC CHUNKS")


if __name__ == "__main__":
    main()
