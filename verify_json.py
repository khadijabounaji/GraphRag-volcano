import os, json

path = "vector_store.json"
size = os.path.getsize(path)
print(f"vector_store.json : {size:,} octets ({size/1024/1024:.1f} MB)")

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"{len(data)} chunks dans le JSON")
print()
print("=== Exemple chunk 1 (200 premiers caracteres) ===")
print(data[0]["texte"][:200])
print()
emb = data[0]["embedding"]
print(f"Embedding: [{emb[0]:.6f}, {emb[1]:.6f}, ...] ({len(emb)} dims)")
