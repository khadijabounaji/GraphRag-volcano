"""
step2_embedding_retrieval.py - Etape 2 : Embedding + Retrieval (6 methodes)
=============================================================================
Chunking  : M4 Sliding Window (window=300, step=150)
Embedding : all-mpnet-base-v2 (768 dims, normalise L2)
Metriques : Precision@5, Recall@5, F1, MRR, nDCG@5
Ground truth : TF-IDF percentile 90% (neutre, sans cross-encoder bloquant)
=============================================================================
Usage      : python step2_embedding_retrieval.py
Dependances: pip install faiss-cpu sentence-transformers rank-bm25 scikit-learn numpy
"""

import os, re, csv, sys, time, warnings, unicodedata
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
TEXT_PATH       = r"C:\Users\hp\Downloads\THese_nettoye100.txt"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
TOP_K           = 5
CHUNK_WINDOW    = 300
CHUNK_STEP      = 150

TEST_QUERIES = [
    "Age des syenites du Piton des Neiges",
    "Formation des calderas a La Reunion",
    "Methodes geochronologiques U-Pb sur zircons",
    "Rift zone N030 et sill zones",
    "Breche des Orangers cirque Mafate",
    "Temperature des sources thermales Cilaos",
    "Complexe hypovolcanique gabbroique",
    "Modele hydrothermal de Benard 2020",
]

# ---------------------------------------------------------------------------
# 1. CHARGEMENT TEXTE
# ---------------------------------------------------------------------------
def load_text_file(path):
    if not os.path.exists(path):
        print(f"[ERREUR] Fichier introuvable : {path}"); sys.exit(1)
    raw = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            with open(path, "r", encoding=enc) as f:
                raw = f.read()
            print(f"  Fichier lu en {enc} ({len(raw):,} caracteres bruts)")
            break
        except (UnicodeDecodeError, LookupError):
            continue
    if raw is None:
        print("[ERREUR] Encodage inconnu."); sys.exit(1)
    text = unicodedata.normalize("NFKC", raw)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x80-\xFF\u0100-\uFFFF]", " ", text)
    text = text.replace("\t", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

# ---------------------------------------------------------------------------
# 2. CHUNKING M4
# ---------------------------------------------------------------------------
def chunk_sliding_window(text, window=CHUNK_WINDOW, step=CHUNK_STEP):
    words = text.split()
    return [" ".join(words[i:i+window])
            for i in range(0, len(words), step)
            if len(" ".join(words[i:i+window]).strip()) > 30]

# ---------------------------------------------------------------------------
# 3. EMBEDDING
# ---------------------------------------------------------------------------
def compute_embeddings(chunks, model):
    print(f"  Calcul des embeddings ({len(chunks)} chunks)...")
    emb = model.encode(chunks, show_progress_bar=True, batch_size=64,
                        normalize_embeddings=True)
    return np.array(emb, dtype=np.float32)

# ---------------------------------------------------------------------------
# 4. SIX METHODES DE RETRIEVAL
# ---------------------------------------------------------------------------
def retrieve_cosine(query, chunks, emb, model, k=TOP_K):
    q = model.encode([query], normalize_embeddings=True)
    s = cosine_similarity(q, emb)[0]
    t = np.argsort(s)[::-1][:k]
    return [(int(i), float(s[i]), chunks[i]) for i in t]

def build_faiss_index(emb):
    import faiss
    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb.astype(np.float32))
    return idx

def retrieve_faiss(query, chunks, faiss_idx, emb, model, k=TOP_K):
    q = model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = faiss_idx.search(q, k)
    return [(int(i), float(s), chunks[i])
            for i, s in zip(indices[0], scores[0]) if i >= 0]

def build_bm25_index(chunks):
    from rank_bm25 import BM25Okapi
    return BM25Okapi([c.lower().split() for c in chunks])

def retrieve_bm25(query, chunks, bm25, k=TOP_K):
    s = bm25.get_scores(query.lower().split())
    t = np.argsort(s)[::-1][:k]
    return [(int(i), float(s[i]), chunks[i]) for i in t]

def retrieve_dpr(query, chunks, emb, model, k=TOP_K):
    q = model.encode([query], normalize_embeddings=True)
    s = np.dot(emb, q.T).flatten()
    t = np.argsort(s)[::-1][:k]
    return [(int(i), float(s[i]), chunks[i]) for i in t]

def retrieve_hybrid(query, chunks, emb, model, bm25, k=TOP_K, alpha=0.5):
    q   = model.encode([query], normalize_embeddings=True)
    vs  = cosine_similarity(q, emb)[0]
    bs  = bm25.get_scores(query.lower().split())
    def mm(a): return (a-a.min())/(a.max()-a.min()+1e-10)
    c   = alpha*mm(vs) + (1-alpha)*mm(bs)
    t   = np.argsort(c)[::-1][:k]
    return [(int(i), float(c[i]), chunks[i]) for i in t]

def retrieve_reranking(query, chunks, emb, model, tv, tm, k=TOP_K, cands=20):
    q    = model.encode([query], normalize_embeddings=True)
    s    = cosine_similarity(q, emb)[0]
    top  = np.argsort(s)[::-1][:cands]
    qt   = tv.transform([query.lower()])
    rs   = cosine_similarity(qt, tm[top])[0]
    rnk  = sorted(zip(top, rs), key=lambda x: x[1], reverse=True)
    return [(int(i), float(sc), chunks[i]) for i, sc in rnk[:k]]

# ---------------------------------------------------------------------------
# 5. GROUND TRUTH TF-IDF PERCENTILE 90%
# ---------------------------------------------------------------------------
def build_ground_truth(queries, chunks, tv, tm, pct=90):
    gt = {}
    for qi, q in enumerate(queries):
        qv   = tv.transform([q.lower()])
        sims = cosine_similarity(qv, tm)[0]
        thr  = np.percentile(sims, pct)
        rel  = set(np.where(sims >= thr)[0])
        if not rel: rel = {int(np.argmax(sims))}
        mn, mx = sims.min(), sims.max()
        graded = {ci: float((sims[ci]-mn)/(mx-mn+1e-10)) for ci in range(len(chunks))}
        gt[qi] = {"relevant_set": rel, "graded_scores": graded}
    return gt

# ---------------------------------------------------------------------------
# 6. METRIQUES : P, R, F1, MRR, nDCG
# ---------------------------------------------------------------------------
def prf1(retrieved, relevant):
    r, rel = set(retrieved), relevant
    tp = len(r & rel)
    p  = tp/len(r)   if r   else 0.0
    rc = tp/len(rel) if rel  else 0.0
    f1 = 2*p*rc/(p+rc) if (p+rc)>0 else 0.0
    return round(p,4), round(rc,4), round(f1,4)

def mrr(retrieved, relevant):
    for rank, idx in enumerate(retrieved, 1):
        if idx in relevant: return round(1.0/rank, 4)
    return 0.0

def ndcg(retrieved, graded, k=TOP_K):
    dcg  = sum(max(0.0, graded.get(i,0.0))/np.log2(r+2)
               for r, i in enumerate(retrieved[:k]))
    idcg = sum(max(0.0, v)/np.log2(r+2)
               for r, v in enumerate(sorted(graded.values(), reverse=True)[:k]))
    return round(dcg/idcg, 4) if idcg>0 else 0.0

def all_metrics(retrieved, gt_entry):
    rel = gt_entry["relevant_set"]
    gr  = gt_entry["graded_scores"]
    p, r, f1 = prf1(retrieved, rel)
    return {"precision": p, "recall": r, "f1": f1,
            "mrr": mrr(retrieved, rel), "ndcg": ndcg(retrieved, gr)}

# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------
def main():
    print("="*82)
    print("  STEP 2 : EMBEDDING + RETRIEVAL (6 METHODES)")
    print(f"  Chunking  : M4 Sliding Window (window={CHUNK_WINDOW}, step={CHUNK_STEP})")
    print(f"  Embedding : {EMBEDDING_MODEL}")
    print(f"  Metriques : Precision@5  Recall@5  F1  MRR  nDCG@5")
    print("="*82)

    if os.path.exists("vector_store.pkl"):
        os.remove("vector_store.pkl")
        print("  Ancien vector_store.pkl supprime.\n")

    print(f"Chargement : {TEXT_PATH}")
    text = load_text_file(TEXT_PATH)
    print(f"  Texte pret : {len(text):,} chars | {len(text.split()):,} mots\n")

    print("Chunking M4 - Sliding Window...")
    chunks = chunk_sliding_window(text)
    print(f"  {len(chunks)} chunks generes.\n")

    print("[1/4] Chargement du modele d'embedding...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Modele charge : {EMBEDDING_MODEL}\n")

    print("[2/4] Calcul des embeddings...")
    emb = compute_embeddings(chunks, model)
    print(f"  Shape : {emb.shape}\n")

    print("[3/4] Sauvegarde VectorStore...")
    from vector_store import VectorStore
    VectorStore(chunks=chunks, embeddings=emb).save("vector_store.pkl")
    print()

    print("[4/4] Construction des index...")
    faiss_idx = build_faiss_index(emb)
    bm25_idx  = build_bm25_index(chunks)
    tv        = TfidfVectorizer(max_features=8000, sublinear_tf=True)
    tm        = tv.fit_transform([c.lower() for c in chunks])
    print("  FAISS, BM25, TF-IDF construits.\n")

    print("Construction du ground truth (TF-IDF percentile 90%)...")
    gt = build_ground_truth(TEST_QUERIES, chunks, tv, tm)
    print(f"  Ground truth pret pour {len(TEST_QUERIES)} queries.\n")

    methods = {
        "M1 - Cosine":    lambda q: retrieve_cosine(q, chunks, emb, model),
        "M2 - FAISS":     lambda q: retrieve_faiss(q, chunks, faiss_idx, emb, model),
        "M3 - BM25":      lambda q: retrieve_bm25(q, chunks, bm25_idx),
        "M4 - DPR":       lambda q: retrieve_dpr(q, chunks, emb, model),
        "M5 - Hybride":   lambda q: retrieve_hybrid(q, chunks, emb, model, bm25_idx),
        "M6 - Reranking": lambda q: retrieve_reranking(q, chunks, emb, model, tv, tm),
    }

    all_results = []
    agg = {n: {"precision":[],"recall":[],"f1":[],"mrr":[],"ndcg":[],"time_ms":[]}
           for n in methods}

    print("="*82)
    print("  RESULTATS PAR QUERY")
    print("="*82)

    for qi, query in enumerate(TEST_QUERIES):
        print(f"\n--- Q{qi+1}/{len(TEST_QUERIES)}: \"{query}\"")
        print("-"*75)
        for mn, mf in methods.items():
            t0  = time.perf_counter()
            res = mf(query)
            ms  = round((time.perf_counter()-t0)*1000, 2)
            idx = [r[0] for r in res]
            m   = all_metrics(idx, gt[qi])
            top = (res[0][2][:55].encode("ascii", errors="replace").decode("ascii")
                   if res else "N/A")
            print(f"  {mn:18s} | P={m['precision']:.3f} R={m['recall']:.3f} "
                  f"F1={m['f1']:.3f} MRR={m['mrr']:.3f} nDCG={m['ndcg']:.3f} "
                  f"| {ms:7.1f}ms | \"{top}...\"")
            all_results.append({"Query":f"Q{qi+1}","Methode":mn,
                "Precision":m["precision"],"Recall":m["recall"],
                "F1-score":m["f1"],"MRR":m["mrr"],"nDCG@5":m["ndcg"],
                "Temps(ms)":ms,"Top chunk":top})
            for k in ("precision","recall","f1","mrr","ndcg"):
                agg[mn][k].append(m[k])
            agg[mn]["time_ms"].append(ms)

    print("\n"+"="*82)
    print(f"  TABLEAU COMPARATIF GLOBAL — {len(TEST_QUERIES)} QUERIES")
    print("="*82)

    summary = []
    for name, a in agg.items():
        ap=round(float(np.mean(a["precision"])),4)
        ar=round(float(np.mean(a["recall"])),4)
        af=round(float(np.mean(a["f1"])),4)
        am=round(float(np.mean(a["mrr"])),4)
        an=round(float(np.mean(a["ndcg"])),4)
        at=round(float(np.mean(a["time_ms"])),2)
        gs=round(0.30*an+0.25*am+0.25*af+0.10*ar+0.10*max(0,1-at/500),4)
        summary.append({"Methode":name,"P@5":ap,"R@5":ar,"F1":af,
                        "MRR":am,"nDCG@5":an,"Latence(ms)":at,"Score Global":gs})

    summary.sort(key=lambda x: x["Score Global"], reverse=True)
    keys   = list(summary[0].keys())
    widths = {k: max(len(k), max(len(str(r[k])) for r in summary)) for k in keys}
    print("  ".join(k.ljust(widths[k]) for k in keys))
    print("-"*82)
    for row in summary:
        print("  ".join(str(row[k]).ljust(widths[k]) for k in keys))

    best = summary[0]
    print(f"\n  MEILLEURE METHODE : {best['Methode']}")
    print(f"  Score Global={best['Score Global']:.4f} | nDCG@5={best['nDCG@5']:.4f} | "
          f"MRR={best['MRR']:.4f} | F1={best['F1']:.4f}")

    with open("step2_embedding_results.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(all_results[0].keys()))
        w.writeheader(); w.writerows(all_results)
    with open("step2_embedding_global.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=keys)
        w.writeheader(); w.writerows(summary)

    print("\n  Resultats detailles -> step2_embedding_results.csv")
    print("  Resultats globaux   -> step2_embedding_global.csv")
    print("  Pipeline termine avec succes.")
    print("="*82)

if __name__ == "__main__":
    main()