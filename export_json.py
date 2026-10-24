import pickle
import json
import numpy as np

with open("vector_store.pkl", "rb") as f:
    data = pickle.load(f)

chunks     = data["chunks"]
embeddings = data["embeddings"]

vector_store_json = []
for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
    vector_store_json.append({
        "chunk_id":  i + 1,
        "texte":     chunk.strip(),
        "embedding": emb.tolist()
    })

with open("vector_store.json", "w", encoding="utf-8") as f:
    json.dump(vector_store_json, f, ensure_ascii=False, indent=2)

print(f"Exporte {len(vector_store_json)} chunks -> vector_store.json")
