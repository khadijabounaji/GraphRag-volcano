import pickle

with open("vector_store.pkl", "rb") as f:
    data = pickle.load(f)

chunks = data["chunks"]
print(f"{len(chunks)} chunks trouves")

with open("chunks_export.txt", "w", encoding="utf-8") as f:
    for i, chunk in enumerate(chunks):
        f.write(f"=== CHUNK {i+1} ===\n")
        f.write(chunk.strip())
        f.write("\n\n")

print("Exporte -> chunks_export.txt")
