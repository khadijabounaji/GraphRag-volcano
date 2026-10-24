"""
step1_chunking.py - Etape 1 : Chunking NLP (7 methodes) pour RAG academique
=============================================================================
Source  : fichier texte propre (.txt)
Methodes: M1 Fixed-Size | M2 Sentence | M3 Paragraph | M4 Sliding Window
          M5 Semantic   | M6 Recursive | M7 Token-Based
Metriques: Nb chunks, Longueur moy, Ecart-type, Intra-sim, Inter-sim,
           Recall@5, Precision@5, F1-score, Score Global
=============================================================================
Usage      : python step1_chunking.py
Dependances: pip install scikit-learn numpy
"""

import os
import re
import csv
import sys
import warnings
import unicodedata

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

warnings.filterwarnings("ignore")
np.random.seed(42)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TEXT_PATH = r"C:\Users\hp\Downloads\THese_nettoye100.txt"

TEST_QUERIES = [
    "methodologie de recherche qualitative",
    "resultats experimentaux et analyse statistique",
    "analyse des donnees et interpretation",
    "conclusion perspectives et travaux futurs",
    "revue de litterature et etat de l art",
    "modele theorique et cadre conceptuel",
    "validation et evaluation du systeme",
]

TOP_K = 5

PARAMS = {
    "M1": {"size": 600,  "overlap": 80},
    "M4": {"window": 300, "step": 150},
    "M5": {"threshold": 0.30},
    "M6": {"max_size": 700},
    "M7": {"max_tokens": 200},
}

# ---------------------------------------------------------------------------
# 1. CHARGEMENT DU TEXTE
# ---------------------------------------------------------------------------

def load_text_file(path: str) -> str:
    if not os.path.exists(path):
        print(f"[ERREUR] Fichier introuvable : {path}")
        sys.exit(1)

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
        print("[ERREUR] Encodage inconnu.")
        sys.exit(1)

    text = unicodedata.normalize("NFKC", raw)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x80-\xFF\u0100-\uFFFF]", " ", text)
    text = text.replace("\t", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    print(f"  Texte nettoye : {len(text):,} chars | {len(text.split()):,} mots\n")
    return text


# ---------------------------------------------------------------------------
# 2. SEPT METHODES DE CHUNKING
# ---------------------------------------------------------------------------

def method1_fixed_size(text, size=PARAMS["M1"]["size"],
                        overlap=PARAMS["M1"]["overlap"]):
    chunks, start = [], 0
    while start < len(text):
        end   = min(start + size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 30:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def method2_sentence_based(text, min_len=40):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) >= min_len]


def method3_paragraph_based(text, min_len=80):
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if len(p.strip()) >= min_len]


def method4_sliding_window(text, window=PARAMS["M4"]["window"],
                            step=PARAMS["M4"]["step"]):
    words  = text.split()
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i: i + window])
        if len(chunk.strip()) > 30:
            chunks.append(chunk)
    return chunks


def method5_semantic(text, threshold=PARAMS["M5"]["threshold"]):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text)
                 if len(s.strip()) >= 40]
    if len(sentences) < 2:
        return sentences

    tfidf = TfidfVectorizer(max_features=5000, sublinear_tf=True).fit_transform(sentences)
    chunks, current_group = [], [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(tfidf[i - 1], tfidf[i])[0][0]
        if sim >= threshold:
            current_group.append(sentences[i])
        else:
            merged = " ".join(current_group)
            if len(merged) > 30:
                chunks.append(merged)
            current_group = [sentences[i]]

    if current_group:
        merged = " ".join(current_group)
        if len(merged) > 30:
            chunks.append(merged)
    return chunks


def method6_recursive_split(text, max_size=PARAMS["M6"]["max_size"],
                              separators=None):
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    def _split(txt, seps):
        if not seps or len(txt) <= max_size:
            return [txt] if txt.strip() else []
        sep    = seps[0]
        parts  = txt.split(sep)
        result, current = [], ""
        for part in parts:
            candidate = (current + sep + part) if current else part
            if len(candidate) <= max_size:
                current = candidate
            else:
                if current.strip():
                    result.extend(_split(current, seps[1:]))
                current = part
        if current.strip():
            result.extend(_split(current, seps[1:]))
        return result

    return [c.strip() for c in _split(text, separators) if len(c.strip()) > 30]


def method7_token_based(text, max_tokens=PARAMS["M7"]["max_tokens"]):
    words  = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens):
        chunk = " ".join(words[i: i + max_tokens])
        if len(chunk.strip()) > 30:
            chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# 3. METRIQUES
# ---------------------------------------------------------------------------

def intra_similarity(chunks):
    """Coherence interne : compare les deux moities de chaque chunk."""
    if len(chunks) < 2:
        return 1.0
    sims = []
    for chunk in chunks:
        words = chunk.split()
        half  = max(len(words) // 2, 1)
        a, b  = " ".join(words[:half]), " ".join(words[half:])
        if a and b:
            try:
                m = TfidfVectorizer(min_df=1).fit_transform([a, b])
                sims.append(cosine_similarity(m[0], m[1])[0][0])
            except Exception:
                continue
    return float(np.mean(sims)) if sims else 0.5


def inter_similarity(chunks):
    """Redondance entre chunks (plus bas = mieux)."""
    if len(chunks) < 2:
        return 1.0
    sample = chunks if len(chunks) <= 300 else list(
        np.random.choice(chunks, 300, replace=False))
    vect   = TfidfVectorizer(max_features=5000, min_df=1)
    matrix = vect.fit_transform(sample)
    sim_m  = cosine_similarity(matrix)
    np.fill_diagonal(sim_m, 0)
    upper  = sim_m[np.triu_indices_from(sim_m, k=1)]
    return float(np.mean(upper))


def compute_retrieval_metrics(chunks, queries, k=TOP_K):
    """Recall@k, Precision@k, F1 via TF-IDF proxy."""
    if len(chunks) < 2:
        return {"recall_at_k": 0.0, "precision_at_k": 0.0, "f1_score": 0.0}

    all_docs    = chunks + queries
    tfidf       = TfidfVectorizer(max_features=8000, sublinear_tf=True).fit_transform(all_docs)
    chunk_vecs  = tfidf[:len(chunks)]
    query_vecs  = tfidf[len(chunks):]

    recalls, precisions = [], []
    for q_vec in query_vecs:
        sims      = cosine_similarity(q_vec, chunk_vecs)[0]
        top_k_idx = set(np.argsort(sims)[::-1][:k])
        threshold = np.percentile(sims, 90)
        relevant  = set(np.where(sims >= threshold)[0])
        if not relevant:
            relevant = {int(np.argmax(sims))}
        tp = len(relevant & top_k_idx)
        recalls.append(tp / len(relevant))
        precisions.append(tp / len(top_k_idx) if top_k_idx else 0.0)

    avg_r = float(np.mean(recalls))
    avg_p = float(np.mean(precisions))
    f1    = (2 * avg_p * avg_r / (avg_p + avg_r)) if (avg_p + avg_r) > 0 else 0.0
    return {"recall_at_k": avg_r, "precision_at_k": avg_p, "f1_score": f1}


def evaluate_method(name, chunks, queries):
    print(f"  -> {len(chunks):>4} chunks | intra...", end=" ", flush=True)
    lengths = [len(c) for c in chunks]
    avg_len = float(np.mean(lengths)) if lengths else 0.0
    std_len = float(np.std(lengths))  if lengths else 0.0
    intra   = intra_similarity(chunks)
    print("inter...", end=" ", flush=True)
    inter   = inter_similarity(chunks)
    print("retrieval...", end=" ", flush=True)
    ret     = compute_retrieval_metrics(chunks, queries)
    print("OK")

    global_score = (
        0.25 * ret["f1_score"]
        + 0.20 * ret["recall_at_k"]
        + 0.15 * ret["precision_at_k"]
        + 0.20 * intra
        + 0.15 * (1.0 - inter)
        + 0.05 * min(1.0, avg_len / 800)
    )
    return {
        "Methode":              name,
        "Nb chunks":            len(chunks),
        "Longueur moy (chars)": round(avg_len, 1),
        "Ecart-type":           round(std_len, 1),
        "Intra-similarity":     round(intra, 4),
        "Inter-similarity":     round(inter, 4),
        "Recall@5":             round(ret["recall_at_k"], 4),
        "Precision@5":          round(ret["precision_at_k"], 4),
        "F1-score":             round(ret["f1_score"], 4),
        "Score Global":         round(global_score, 4),
    }


# ---------------------------------------------------------------------------
# 4. MAIN
# ---------------------------------------------------------------------------

def main():
    print("=" * 77)
    print("  STEP 1 - CHUNKING NLP (7 METHODES) | RAG Academique")
    print("=" * 77)

    print(f"\nChargement : {TEXT_PATH}")
    text = load_text_file(TEXT_PATH)

    print("Generation des chunks...")
    methods = {
        "M1 - Fixed-Size":      method1_fixed_size(text),
        "M2 - Sentence-Based":  method2_sentence_based(text),
        "M3 - Paragraph-Based": method3_paragraph_based(text),
        "M4 - Sliding Window":  method4_sliding_window(text),
        "M5 - Semantic":        method5_semantic(text),
        "M6 - Recursive Split": method6_recursive_split(text),
        "M7 - Token-Based":     method7_token_based(text),
    }

    print("\nEvaluation des methodes :")
    results = []
    for name, chunks in methods.items():
        print(f"\n[{name}]")
        metrics = evaluate_method(name, chunks, TEST_QUERIES)
        results.append(metrics)

    results.sort(key=lambda x: x["Score Global"], reverse=True)

    print("\n" + "=" * 77)
    print("  TABLEAU COMPARATIF")
    print("=" * 77)
    keys   = list(results[0].keys())
    widths = {k: max(len(k), max(len(str(r[k])) for r in results)) for k in keys}
    header = "  ".join(k.ljust(widths[k]) for k in keys)
    print(header)
    print("-" * len(header))
    for row in results:
        print("  ".join(str(row[k]).ljust(widths[k]) for k in keys))

    best = results[0]
    print(f"\n  MEILLEURE METHODE : {best['Methode']}")
    print(f"  Score Global : {best['Score Global']:.4f} | "
          f"F1 : {best['F1-score']:.4f} | "
          f"Recall@5 : {best['Recall@5']:.4f}")

    csv_path = "step1_chunking_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Resultats exportes -> {csv_path}")
    print("=" * 77)


if __name__ == "__main__":
    main()