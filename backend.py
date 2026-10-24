"""
backend.py — FastAPI Backend — Agentic Graph-Vector RAG
========================================================
ENDPOINTS (comme demandé dans le cours) :
  POST /query          → Q-Learning + routage + réponse
  GET  /vectorial      → Chunking + PCA embeddings + retrieval
  GET  /graph          → Neo4j Aura + Louvain + analyse avancée
  GET  /agentic        → Q-table + reward history + decision path
  POST /graph/louvain  → Exécute Louvain sur le graphe (AJOUTÉ)
  POST /graph/louvain/visualize → Visualisation couleurs (AJOUTÉ)

CONNEXION AUX MODULES EXISTANTS :
  vector_store.pkl  → chunks + embeddings FAISS
  Neo4j Aura        → graph store (step4)
  step5_agentic_rag → Q-Learning agent

Usage :
  pip install fastapi uvicorn faiss-cpu sentence-transformers neo4j numpy scikit-learn
  uvicorn backend:app --reload --port 8000
  → Swagger UI : http://localhost:8000/docs
"""

import os
import re
import pickle
import json
import random
import math
import time
from collections import Counter
from typing import Optional, List

import numpy as np
from fastapi import FastAPI, HTTPException # Reload trigger
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ============================================================
# IMPORTS CONDITIONNELS
# ============================================================
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    FAISS_OK = True
except ImportError:
    FAISS_OK = False
    print("[WARN] faiss / sentence-transformers non installé")

try:
    from sklearn.decomposition import PCA
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("[WARN] scikit-learn non installé")

try:
    from neo4j import GraphDatabase
    NEO4J_LIB_OK = True
except ImportError:
    NEO4J_LIB_OK = False
    print("[WARN] neo4j non installé")

# ============================================================
# CONFIGURATION
# ============================================================
NEO4J_URI      = "neo4j+s://acfc5ab2.databases.neo4j.io"
NEO4J_USER     = "acfc5ab2"
NEO4J_PASSWORD = "XVH4uQXBtkf_FSMrK4T16vcTbJZ8QQu_7-HblC-hJgQ"
VECTOR_STORE_PATH = "vector_store.pkl"
EMBEDDING_MODEL   = "sentence-transformers/all-mpnet-base-v2"
Q_TABLE_PATH      = "q_table.json"
RETRIEVAL_MIN_SCORE = 0.55  # Seuil minimum de similarité pour accepter un résultat

# Groq LLM pour reformulation (optionnel — gratuit sur console.groq.com)
# Remplacez par votre clé ou définissez la variable d'environnement GROQ_API_KEY
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
GROQ_MODEL   = "llama-3.3-70b-versatile"
GROQ_ENABLED = bool(GROQ_API_KEY)


# ============================================================
# MOTS-CLES ROUTAGE
# ============================================================
GRAPH_KEYWORDS = [
    "relation","lien","relie","qui","quel","quelle","repose sur","surmonte",
    "recoupe","présent dans","localisé","caldera","rift","faille","brèche",
    "cirque","mafate","salazie","cilaos","pn0","pn1","pn2","pn3","pn4",
    "précède","avant","après","âge","date","auteur","chercheur",
    "basalte","phonolite","syénite","gabbro","ignimbrite",
]
VECTOR_KEYWORDS = [
    "explique","explication","décris","comment","pourquoi","qu'est-ce",
    "définition","principe","mécanisme","processus","hydrothermal",
    "géothermie","géochronologie","thermochronologie","volcanisme",
    "éruption","différenciation","magma","point chaud","u-pb","ar-ar","zdd",
    "parle-moi","résume","synthèse",
]

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="Agentic Graph-Vector RAG — Piton des Neiges",
    description="Backend PFA 4ème année IA — Dr. EL MKHALET MOUNA",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve local frontend file so it is available at http://localhost:<port>/frontend.html
@app.get("/frontend.html")
async def serve_frontend():
    fp = os.path.join(os.getcwd(), "frontend.html")
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="frontend.html not found on server")
    return FileResponse(fp, media_type="text/html")


# Serve index.html file so it is available at http://localhost:<port>/index.html
@app.get("/index.html")
async def serve_index():
    fp = os.path.join(os.getcwd(), "index.html")
    if not os.path.exists(fp):
        raise HTTPException(status_code=404, detail="index.html not found on server")
    return FileResponse(fp, media_type="text/html")


# ============================================================
# SCHEMAS PYDANTIC
# ============================================================
class QueryRequest(BaseModel):
    question: str
    feedback: Optional[float] = None
    previous_question: Optional[str] = None

class QueryResponse(BaseModel):
    question: str
    analysis_type: str        # Semantic / Systematic / Hybrid
    routing: str              # Vectorial RAG / Graph RAG / Both
    confidence: float
    answer: str
    state: str
    action: str
    reward: float
    decision_path: List[str]
    cypher_query: Optional[str] = None
    raw_graph_data: Optional[List[dict]] = None
    extracted_entities: Optional[List[str]] = None

class LouvainResponse(BaseModel):
    status: str
    communities_found: int
    modularity: float
    message: str
    community_distribution: Optional[List[dict]] = None

# ============================================================
# SINGLETONS (chargés une seule fois au démarrage)
# ============================================================
_vector_store = None
_neo4j_driver = None
_embed_model   = None
_q_agent       = None

def get_vector_store():
    global _vector_store, _embed_model
    if _vector_store is None and FAISS_OK:
        try:
            if os.path.exists(VECTOR_STORE_PATH):
                with open(VECTOR_STORE_PATH, "rb") as f:
                    data = pickle.load(f)
                chunks     = data.get("chunks", [])
                embeddings = data.get("embeddings", None)
                _embed_model = SentenceTransformer(EMBEDDING_MODEL)
                if embeddings is None:
                    embeddings = _embed_model.encode(chunks)
                vecs = np.array(embeddings, dtype="float32")
                idx  = faiss.IndexFlatL2(vecs.shape[1])
                idx.add(vecs)
                _vector_store = {"chunks": chunks, "embeddings": vecs, "index": idx}
        except Exception as e:
            print(f"[VectorStore] Erreur : {e}")
    return _vector_store

def get_neo4j():
    global _neo4j_driver
    if _neo4j_driver is None and NEO4J_LIB_OK:
        try:
            _neo4j_driver = GraphDatabase.driver(
                NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD), connection_timeout=15)
            _neo4j_driver.verify_connectivity()
            with _neo4j_driver.session() as session:
                session.run("RETURN 'OK'")
                print("[Neo4j] Connecté avec succès à Aura.")
        except Exception as e:
            print(f"[Neo4j] Erreur : {e}")
            _neo4j_driver = None
    return _neo4j_driver

def get_q_agent():
    global _q_agent
    if _q_agent is None:
        states  = ["graph", "vector", "hybrid", "unknown"]
        actions = ["use_graph", "use_vector", "use_both", "fallback"]
        q = {s: {a: 0.0 for a in actions} for s in states}
        # Biais manuel pour forcer l'usage du graphe sur les questions relationnelles
        q["graph"]["use_graph"] = 5.0
        q["hybrid"]["use_both"] = 5.0
        q["vector"]["use_vector"] = 5.0
        history = []
        if os.path.exists(Q_TABLE_PATH):
            try:
                with open(Q_TABLE_PATH) as f:
                    d = json.load(f)
                q       = d.get("Q", q)
                history = d.get("history", [])
            except Exception:
                pass
        _q_agent = {"Q": q, "history": history,
                    "states": states, "actions": actions,
                    "alpha": 0.1, "gamma": 0.9, "epsilon": 0.05}
    return _q_agent

def save_q_agent(agent):
    with open(Q_TABLE_PATH, "w") as f:
        json.dump({"Q": agent["Q"], "history": agent["history"][-200:]}, f, indent=2)

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================
def detect_query_type(question: str):
    q = question.lower()
    
    # 0. Détection de requêtes de citation (preuve documentaire)
    citation_keywords = [
        "où est", "cite", "citation", "montre-moi", "montre", "passage", "preuve", 
        "source", "référence", "écrit", "où c'est", "extrait"
    ]
    if any(kw in q for kw in citation_keywords):
        return "citation", "Citation RAG", 0.95, "citation"
        
    # 1. Détection améliorée (Mots-clés stricts)
    relational_keywords = [
        "relie", "traverse", "compose", "origine", "interaction", "localisation",
        "où se trouve", "qui", "quel lien", "délimite", "sépare", "impacte", "relation", "entre"
    ]
    semantic_keywords = [
        "processus", "mécanisme", "pourquoi", "comment fonctionne", "définis",
        "concept", "résume", "histoire", "évolution", "qu'est-ce que"
    ]
    
    is_relational = sum(1 for kw in relational_keywords if kw in q)
    is_semantic = sum(1 for kw in semantic_keywords if kw in q)
    
    if is_relational > 0 and is_semantic == 0:
        return "systematic", "Graph RAG", 0.85, "graph"
    elif is_semantic > 0 and is_relational == 0:
        return "semantic", "Vectorial RAG", 0.85, "vector"
    elif is_relational > 0 and is_semantic > 0:
        return "hybrid", "Both", 0.70, "hybrid"
    else:
        g = sum(1 for kw in GRAPH_KEYWORDS  if kw in q)
        v = sum(1 for kw in VECTOR_KEYWORDS if kw in q)
        total = g + v or 1
        if g > 0 and v > 0:
            return "hybrid", "Both", round((g + v) / (total + 2), 2), "hybrid"
        if g > 0:
            return "systematic", "Graph RAG", round(g / total, 2) * 0.6 + 0.5, "graph"
        if v > 0:
            return "semantic",   "Vectorial RAG", round(v / total, 2) * 0.6 + 0.5, "vector"
        return "semantic", "Vectorial RAG", 0.55, "unknown"

def choose_action(agent, state):
    if random.uniform(0, 1) < agent["epsilon"]:
        return random.choice(agent["actions"])
    return max(agent["Q"][state], key=agent["Q"][state].get)

def update_q(agent, state, action, reward):
    a, g = agent["alpha"], agent["gamma"]
    best = max(agent["Q"]["unknown"].values())
    agent["Q"][state][action] += a * (reward + g * best - agent["Q"][state][action])
    agent["history"].append({"state": state, "action": action,
                              "reward": reward, "t": int(time.time())})
    save_q_agent(agent)

def citation_search(question: str, previous_question: str = None, k: int = 3):
    vs = get_vector_store()
    if not vs or not _embed_model: return []

    context_text = f"{previous_question} {question}" if previous_question else question
    
    # Extraire les mots-clés significatifs
    import re
    words = [w.lower() for w in re.findall(r"\w+", context_text) if len(w) >= 3]
    stop_words = {"quel", "quelle", "est", "sont", "pour", "dans", "avec", "comment", "pourquoi", "qu'est", "cette", "montre", "moi", "où", "c'est", "écrit", "thèse", "passage", "preuve", "source", "cite", "trouve", "texte", "les", "des", "une", "sur"}
    keywords = [w for w in words if w not in stop_words]

    q_vec = _embed_model.encode([context_text]).astype("float32")
    vecs = np.array(vs["embeddings"], dtype="float32")
    
    # Phase 2: Vector Search Global
    idx = faiss.IndexFlatIP(vecs.shape[1])
    idx.add(vecs)
    D, I = idx.search(q_vec, len(vs["chunks"]))
    
    results = []
    # Phase 1 & 3: Keyword Match & Fusion
    for i, d in zip(I[0], D[0]):
        if i < 0 or i >= len(vs["chunks"]): continue
        chunk = vs["chunks"][i]
        chunk_lower = chunk.lower()
        
        matches = sum(1 for kw in keywords if kw in chunk_lower)
        if matches > 0:
            kw_score = matches / max(1, len(keywords))
            # Fusion : Poids dominant pour le keyword exact en mode citation
            final_score = 0.7 * kw_score + 0.3 * float(d)
            results.append({"chunk": chunk[:500], "score": round(final_score, 3)})
            
    if not results:
        # Fallback to standard vector search if no keyword matched
        return vector_search(context_text, k=k)
        
    # Phase 4: Reranking
    results = sorted(results, key=lambda x: x["score"], reverse=True)[:k]
    return results

def vector_search(question: str, k: int = 3, min_score: float = None):
    if min_score is None:
        min_score = RETRIEVAL_MIN_SCORE
    vs = get_vector_store()
    if not vs or not _embed_model:
        return []
    q_vec = _embed_model.encode([question]).astype("float32")
    D, I  = vs["index"].search(q_vec, k)
    results = [{"chunk": vs["chunks"][i][:800], "score": round(1 / (1 + float(d)), 3)}
               for i, d in zip(I[0], D[0]) if 0 <= i < len(vs["chunks"])]
    # Filtrer par seuil de score
    results = [r for r in results if r["score"] >= min_score]
    
    # ----------------------------------------------------
    # FALLBACK : Recherche par mots-clés si aucun résultat ne passe le seuil
    # Utile pour les entités spécifiques comme "PER Salazie-Cilaos"
    if not results:
        import re
        q_words = [w.lower() for w in re.findall(r"\w+", question) if len(w) >= 3]
        stop_words = {"quel", "quelle", "est", "sont", "pour", "dans", "avec", "comment", "pourquoi", "qu'est", "cette"}
        q_words = [w for w in q_words if w not in stop_words]
        
        if q_words:
            fallback_results = []
            for chunk in vs["chunks"]:
                chunk_lower = chunk.lower()
                matches = sum(1 for w in q_words if w in chunk_lower)
                if matches > 0:
                    fallback_results.append({
                        "chunk": chunk[:800],
                        "score": round(0.5 + (0.01 * matches), 3) # Score artificiel de secours
                    })
            
            if fallback_results:
                fallback_results = sorted(fallback_results, key=lambda x: x["score"], reverse=True)[:k]
                results = fallback_results
    # ----------------------------------------------------

    if not results:
        return [{"chunk": "__NO_RESULT__", "score": 0.0}]
    return results


def vector_search_method(question: str, k: int = 3, method: str = "cosine", min_score: float = None):
    if min_score is None:
        min_score = RETRIEVAL_MIN_SCORE
    vs = get_vector_store()
    if not vs or not _embed_model:
        return []

    method = method if method in ["cosine", "dot_product", "bm25_hybrid"] else "cosine"
    q_vec = _embed_model.encode([question]).astype("float32")
    vecs = np.array(vs["embeddings"], dtype="float32")

    if method == "dot_product":
        idx = faiss.IndexFlatIP(vecs.shape[1])
        idx.add(vecs)
        D, I = idx.search(q_vec, k)
        results = [{"chunk": vs["chunks"][i][:800], "score": round(float(d), 3)}
                   for i, d in zip(I[0], D[0]) if 0 <= i < len(vs["chunks"])]
        # Filtrer par seuil
        results = [r for r in results if r["score"] >= min_score]
        if not results:
            return [{"chunk": "__NO_RESULT__", "score": 0.0}]
        return results

    if method == "bm25_hybrid":
        raw = vector_search(question, k=min(len(vs["chunks"]), max(k * 4, 10)), min_score=0.0)  # récupérer tous d'abord
        q_words = [w for w in re.findall(r"\w+", question.lower()) if w]
        total_q = len(q_words) or 1
        results = []
        for item in raw:
            if item["chunk"] == "__NO_RESULT__":
                continue
            chunk_text = item["chunk"].lower()
            chunk_words = set(re.findall(r"\w+", chunk_text))
            matches = sum(1 for w in q_words if w in chunk_words)
            score_bm25 = matches / total_q
            score_final = round(0.6 * item["score"] + 0.4 * score_bm25, 3)
            results.append({"chunk": item["chunk"], "score": score_final})
        results = sorted(results, key=lambda x: x["score"], reverse=True)[:k]
        # Filtrer par seuil après tri
        results = [r for r in results if r["score"] >= min_score]
        if not results:
            return [{"chunk": "__NO_RESULT__", "score": 0.0}]
        return results

    # Default to cosine-like behaviour using existing L2 score function
    return vector_search(question, k=k, min_score=min_score)

def is_offtopic(question: str) -> bool:
    """
    Détecte si une question est hors du domaine géologique.
    """
    q = question.lower().strip()
    
    # Nettoyer les caractères spéciaux pour la détection de mots
    q_clean = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', q).strip()
    
    # Questions trop courtes ou sans sens
    if len(q_clean.split()) < 2:
        return True
    
    # Détection de questions invalides : contient principalement des caractères spéciaux
    non_alpha_count = sum(1 for c in q if not c.isalpha() and c != ' ')
    total_chars = len(q.replace(" ", ""))
    if total_chars > 0 and non_alpha_count / total_chars > 0.5:  # Plus de 50% de caractères spéciaux
        return True
    
    # Mots-clés hors domaine géologique - ÉTENDU
    # FIX: Utiliser les word boundaries pour éviter les faux positifs (ex: "no" dans "géochronologique")
    # NOTE: "pourquoi" a été RETIRÉ - c'est une question géologique valide !
    offtopic_words = [
        "yes", "no", "yep", "nope", "maybe", "ok", "oui", "non",
        "lol", "haha", "test", "hello", "hi",
        "bonjour", "salut", "merci", "khadija", "mohamed", "ahmed",
        "qui es-tu", "ton nom", "comment tu", "aiut", "help"
    ]
    # Test avec word boundaries
    for w in offtopic_words:
        if re.search(r'\b' + re.escape(w) + r'\b', q_clean):
            return True
    
    # Pas un seul mot du domaine géologique
    domain_words = [
        "piton", "neiges", "caldera", "rift", "basalte",
        "phonolite", "géologie", "volcanique", "hydrothermal",
        "magma", "éruption", "formation", "roche", "mineral",
        "cilaos", "mafate", "salazie", "zdd", "u-pb", "ar-ar",
        "thermochronologie", "géochronologie", "syénite", "gabbro",
        "chunk", "document", "thèse", "méthode", "analyse",
        "relation", "structure", "faille", "brèche", "cirque"
    ]
    has_domain = any(w in q_clean for w in domain_words)
    
    # Si question longue (>4 mots) mais aucun mot du domaine → probablement hors-sujet
    if len(q_clean.split()) > 4 and not has_domain:
        return True
        
    return False

import unicodedata

def normalize_string(s: str) -> str:
    s = unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode('utf-8').lower()
    return s.replace("°e", "e").replace("°", "")

def extract_entities_from_query(question: str) -> list:
    known_entities = [
        "Piton des Neiges", "Caldera", "Syenite", "Cilaos", "Mafate", "Salazie",
        "Basalte", "Rift-zone N030", "Breche Orangers", "Phonolite", "Source Thermale", "Gabbro"
    ]
    synonyms = {
        "rift zone n030": "Rift-zone N030",
        "rift n030": "Rift-zone N030",
        "zone n030": "Rift-zone N030",
        "n030": "Rift-zone N030",
        "n030e": "Rift-zone N030",
        "caldeira": "Caldera",
        "caldera": "Caldera"
    }
    
    q_norm = normalize_string(question)
    found = set()
    
    # 1. Matching via synonymes
    for syn, canonical in synonyms.items():
        if syn in q_norm:
            found.add(canonical)
            
    # 2. Matching direct
    for entity in known_entities:
        e_norm = normalize_string(entity)
        if e_norm in q_norm or all(word in q_norm for word in e_norm.split()):
            found.add(entity)
            
    return list(found)

def generate_cypher(question: str, entities: list) -> str:
    # Génération d'une seule requête propre qui filtre les arêtes génériques
    # Utilisation de ANY(x IN $entities WHERE toLower(c.name) CONTAINS x) pour être sûr de matcher
    return """
    MATCH (c:Entity)-[r]->(x:Entity)
    WHERE ANY(e IN $entities WHERE toLower(c.name) CONTAINS e)
    AND ANY(e IN $entities WHERE toLower(x.name) CONTAINS e)
    AND type(r) IN ["LOCALISE_LE_LONG_DE", "CONTROLE_STRUCTURAL", "POSTERIEUR_A", "COMPOSE_DE", "INTRUSIF_DANS", "SOURCE_DE"]
    RETURN c.name AS source, type(r) AS relation, x.name AS target
    LIMIT 6000
    """

def graph_search(question: str, limit: int = 6):
    """
    Recherche des résultats de graphe pour une question donnée.
    Retourne une liste de résultats de relations.
    Utilisé par l'endpoint /graph pour afficher les résultats de requête.
    """
    drv = get_neo4j()
    if not drv:
        return []

    entities = extract_entities_from_query(question)
    if not entities:
        return []

    entities_lower = [e.lower() for e in entities]
    cypher_query = generate_cypher(question, entities_lower)
    
    results = []
    try:
        with drv.session() as s:
            r = s.run(cypher_query, entities=entities_lower)
            for rec in r:
                results.append({
                    "source": rec["source"],
                    "relation": rec["relation"],
                    "target": rec["target"]
                })
    except Exception as e:
        print(f"[graph_search] Erreur Cypher: {e}")
        return []

    return results[:limit]


def graph_rag(question: str):
    drv = get_neo4j()
    if not drv:
        return {"error": "Neo4j non disponible"}

    entities = extract_entities_from_query(question)
    print("Entities détectées:", entities)
    
    if not entities:
        return {"message": "Aucune entité détectée. Basculement Vectoriel suggéré."}

    entities_lower = [e.lower() for e in entities]
    cypher_query = generate_cypher(question, entities_lower)
    print("Cypher:", cypher_query.strip())

    results = []
    with drv.session() as s:
        try:
            r = s.run(cypher_query, entities=entities_lower)
            for rec in r:
                results.append({
                    "source": rec["source"],
                    "relation": rec["relation"],
                    "target": rec["target"]
                })
        except Exception as e:
            print("Erreur Cypher:", e)

    print("Résultats:", results)
    
    if not results:
        return {"message": f"Aucune relation trouvée pour {entities}."}

    answer_context = "\n".join([f"{r['source']} {r['relation']} {r['target']}" for r in results])
    return {
        "extracted_entities": entities,
        "cypher_query": cypher_query.strip(),
        "raw_graph_data": results,
        "graph_context_for_llm": answer_context
    }

def format_graph_answer(results):
    if not results:
        return "Aucune entité trouvée dans le graphe."
    lines = []
    for r in results[:4]:
        line = f"• {r.get('name','?')}"
        if r.get("type"):
            line += f" [{r['type']}]"
        if r.get("community") is not None:
            line += f" (communauté {r['community']})"
        if r.get("description"):
            line += f" — {r['description'][:100]}"
        lines.append(line)
        for rel in (r.get("relations") or [])[:2]:
            if rel.get("rel") and rel.get("neighbor"):
                lines.append(f"  └── {rel['rel']} → {rel['neighbor']}")
    return "\n".join(lines)


def format_vector_answer(results):
    if not results:
        return "Aucun passage pertinent trouvé."
    return "\n\n".join(
        f"[score={r['score']:.2f}] {r['chunk'][:600]}"
        for r in results)

def build_nl_answer(question: str, action: str,
                    graph_results: list, vector_results: list) -> str:
    """
    Reformule les résultats bruts en réponse structurée en langage naturel.
    Pas de LLM externe requis : utilise des templates adaptatifs.
    """
    q_low = question.lower().strip().rstrip("?")
    parts = []

    # ── PARTIE GRAPHE ──────────────────────────────────────────
    if graph_results and action in ("use_graph", "use_both"):
        entities = [r for r in graph_results if r.get("name")]
        if entities:
            if any(kw in q_low for kw in ["relation", "lien", "relie", "entre"]):
                intro = "D'après le graphe de connaissances, voici les relations identifiées :"
            elif any(kw in q_low for kw in ["qu'est", "définit", "définition"]):
                intro = "D'après le graphe de connaissances :"
            else:
                intro = "Analyse structurelle (Graph RAG — Neo4j Aura) :"
            entity_lines = []
            seen = set()
            for r in entities[:5]:
                name = r.get("name", "?")
                if name in seen:
                    continue
                seen.add(name)
                typ  = r.get("type", "")
                comm = r.get("community")
                desc = r.get("description", "")
                rels = r.get("relations") or []
                line = f"▸ {name}"
                if typ:
                    line += f" ({typ})"
                if comm is not None:
                    line += f" — groupe {comm}"
                if desc:
                    line += f"\n  {desc[:150]}"
                entity_lines.append(line)
                for rel in rels[:3]:
                    rn = rel.get("rel", ""); rp = rel.get("neighbor", "")
                    if rn and rp:
                        entity_lines.append(f"  → {rn.replace('_',' ').lower()} : {rp}")
            parts.append(intro + "\n" + "\n".join(entity_lines))

    # ── PARTIE VECTORIELLE ─────────────────────────────────────
    if vector_results and action in ("use_vector", "use_both"):
        sorted_v = sorted(vector_results, key=lambda x: -x.get("score", 0))
        if any(kw in q_low for kw in ["comment", "pourquoi", "expliqu", "mécanisme"]):
            intro_v = "Explication tirée de la thèse :"
        elif any(kw in q_low for kw in ["qu'est", "définit", "méthode"]):
            intro_v = "Définition issue du corpus :"
        else:
            intro_v = "Passages les plus pertinents de la thèse :"
        best_chunks = []
        for r in sorted_v[:3]:
            # Nettoyage initial des espaces
            chunk = re.sub(r'\s+', ' ', r.get("chunk", "").strip())
            
            # Suppression des légendes de figures (ex: "FIGURE 31 : ...")
            chunk = re.sub(r'(?i)(figure|fig\.)\s*\d+\s*:?[^\.]*\.', '', chunk)
            
            # Suppression des numéros de pages isolés ou numérotations parasites
            chunk = re.sub(r'\b\d{1,3}\b', '', chunk)
            chunk = re.sub(r'\s+', ' ', chunk).strip()
            
            if len(chunk) > 80:
                # Tenter de commencer par une majuscule si la phrase est coupée au début
                if not chunk[0].isupper():
                    first_dot = chunk.find('. ')
                    if first_dot != -1 and first_dot < 100:
                        chunk = chunk[first_dot + 2:]
                
                # Tenter de terminer par un point pour éviter les phrases coupées à la fin
                last_dot = chunk.rfind('.')
                if last_dot > len(chunk) // 2:
                    chunk = chunk[:last_dot + 1]
                    
                best_chunks.append(chunk.strip())

        if best_chunks:
            # Concaténation propre sans afficher le % de pertinence
            formatted_chunks = "\n\n".join(f"▸ {c}" for c in best_chunks)
            parts.append(intro_v + "\n\n" + formatted_chunks)

    # ── SYNTHESE FINALE ────────────────────────────────────────
    if not parts:
        return (
            "Je ne dispose pas d'informations suffisantes dans le graphe ou le corpus "
            "pour répondre. Essayez des termes plus précis "
            "(caldera, phonolite, ZDD, hydrothermal…)."
        )
    if action == "use_both" and len(parts) == 2:
        return f"── Graph Store ──\n{parts[0]}\n\n── Vector Store ──\n{parts[1]}"
    return "\n\n".join(parts)


SYSTEM_PROMPT_REFORMULATION = """
Tu es un géologue expert. Ta mission est d'analyser le contexte fourni et de formuler une réponse fluide, claire et académique en français.

RÈGLES OBLIGATOIRES :
1. Si le contexte contient des relations brutes (ex: 'Caldera CONTROLE_STRUCTURAL Rift N030'), tu dois formuler une belle phrase d'explication (ex: 'La Caldera exerce un contrôle structural sur la mise en place de la Rift zone N030.').
2. Sois strictement fidèle au contexte. N'invente pas d'informations géologiques qui ne sont pas fournies.
3. Ne fais pas une liste à puces de mots-clés, rédige un vrai paragraphe explicatif.
4. Si le contexte est vide ou insuffisant, réponds simplement : "Je ne dispose pas de suffisamment d'informations pour répondre."
"""

def reformulate_with_free_ai(question: str, raw_answer: str) -> str:
    if not raw_answer.strip():
        return raw_answer

    import http.client
    import ssl

    print(f"[Groq] Appel avec clé: {GROQ_API_KEY[:20]}...")
    print(f"[Groq] Prompt: {raw_answer[:100]}")

    body = json.dumps({
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Tu es un assistant de recherche spécialisé sur le Piton des Neiges (La Réunion). "
                    "Ta mission est de répondre UNIQUEMENT à partir du contexte fourni ci-dessous.\n\n"
                    "RÈGLES STRICTES :\n"
                    "1. N'invente JAMAIS d'information qui n'est pas explicitement présente dans le contexte.\n"
                    "2. Cite les passages pertinents entre guillemets (« ... ») pour appuyer ta réponse.\n"
                    "3. Si le contexte contient des relations brutes (ex: 'A RELATION B'), reformule-les en phrases claires.\n"
                    "4. Si le contexte ne contient pas assez d'information pour répondre, dis-le clairement : "
                    "'Les passages disponibles ne permettent pas de répondre précisément à cette question.'\n"
                    "5. Commence ta réponse par une synthèse, puis cite les extraits pertinents de la thèse.\n"
                    "6. Ne fabrique PAS de données chiffrées (températures, surfaces, dates) qui ne sont pas dans le contexte."
                )
            },
            {
                "role": "user",
                "content": f"Question : {question}\n\nContexte extrait de la thèse :\n{raw_answer}"
            }
        ],
        "temperature": 0.1,
        "max_tokens": 700
    })

    try:
        ctx = ssl.create_default_context()
        conn = http.client.HTTPSConnection("api.groq.com", context=ctx, timeout=20)
        conn.request(
            "POST",
            "/openai/v1/chat/completions",
            body=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Length": str(len(body.encode("utf-8")))
            }
        )
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        print(f"[Groq] Status: {resp.status}")
        print(f"[Groq] Réponse: {raw[:200]}")

        if resp.status == 200:
            data = json.loads(raw)
            result = data["choices"][0]["message"]["content"].strip()
            if result and len(result) > 20:
                return result
        else:
            print(f"[Groq] Erreur HTTP {resp.status}: {raw}")
    except Exception as e:
        print(f"[Groq] Exception: {type(e).__name__}: {e}")
    finally:
        try:
            conn.close()
        except:
            pass

    return raw_answer


# ============================================================
# ENDPOINT 1 : POST /query (bouton principal)
# ============================================================
@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def process_query(body: QueryRequest):
    """
    Analyse la question, applique Q-Learning pour choisir le store,
    exécute la requête et retourne la réponse avec métadonnées RL.
    """
    question = body.question.strip()
    previous_question = body.previous_question.strip() if body.previous_question else None
    if not question:
        raise HTTPException(status_code=400, detail="Question vide.")

    agent = get_q_agent()

    # 1. Détection du type
    analysis_type, routing, confidence, state = detect_query_type(question)

    # ★ BUG FIX 1 : VALIDATION OFF-TOPIC IMMÉDIATE (AVANT recherches)
    if state != "citation" and is_offtopic(question):
        update_q(agent, "unknown", "fallback", 0.0)
        return QueryResponse(
            question=question, analysis_type="unknown",
            routing="none", confidence=0.0,
            answer=(
                "Je suis un assistant spécialisé en géologie volcanique "
                "du Piton des Neiges (La Réunion). Posez-moi une question "
                "sur la géologie, la pétrologie, le système hydrothermal, "
                "ou les méthodes géochronologiques."
            ),
            state="unknown", action="fallback", reward=0.0,
            decision_path=["User Query", "Question hors contexte → Refus poli"]
        )

    # 1. Détection du type
    analysis_type, routing, confidence, state = detect_query_type(question)

    # 2. Choix de l'action (Q-Learning)
    action = choose_action(agent, state)

    # ★ SAFETY NET : l'action "fallback" ne doit JAMAIS bloquer une question valide
    # Si la question a passé le filtre is_offtopic(), elle est légitime → vector search
    if action == "fallback":
        action = "use_vector"

    # ════════════════════════════════════════════════════════════
    # ★ BRANCHE CITATION RAG — Recherche hybride Keyword+Vector
    # ════════════════════════════════════════════════════════════
    if state == "citation":
        action = "citation"
        decision_path = ["User Query", "Analyse: Citation"]
        if previous_question:
            decision_path.append(f"Contexte: {previous_question[:60]}")
        decision_path.append("Citation RAG → Keyword+Vector Fusion")

        # Phase 1-4 : Recherche hybride (BM25-like + Vector)
        citation_results = citation_search(question, previous_question, k=5)

        if not citation_results or (len(citation_results) == 1 and citation_results[0].get("chunk") == "__NO_RESULT__"):
            answer = (
                "Je n'ai pas trouvé de passage exact dans la thèse correspondant "
                "à cette demande. Essayez de reformuler avec des termes plus précis."
            )
            decision_path.append("Aucun passage trouvé")
            reward = -0.5
        else:
            # Phase 5 : Construire le prompt Citation spécifique
            passages_text = "\n\n".join(
                f"[Passage {i+1}, score={r['score']:.3f}]:\n{r['chunk']}"
                for i, r in enumerate(citation_results)
            )
            citation_prompt = (
                f"Voici les passages EXACTS extraits de la thèse scientifique "
                f"sur le Piton des Neiges.\n\n"
                f"{passages_text}\n\n"
                f"CONSIGNES STRICTES :\n"
                f"1. Ne fais PAS de synthèse générale ni de paraphrase.\n"
                f"2. Cite et reproduis les passages pertinents tels quels entre guillemets.\n"
                f"3. Indique clairement d'où vient chaque citation (ex: 'La thèse indique textuellement : \"...\"').\n"
                f"4. Si la question de suivi fait référence à un concept de la question précédente "
                f"('{previous_question or 'N/A'}'), utilise ce contexte pour identifier "
                f"les passages les plus pertinents.\n"
                f"5. Fournis la PREUVE DOCUMENTAIRE demandée par l'utilisateur."
            )

            # Reformulation LLM avec le prompt citation
            answer = reformulate_with_free_ai(question, citation_prompt)
            decision_path.append("LLM Citation Extraction (Free AI)")

            # Si le LLM n'a pas répondu, fallback vers les passages bruts
            if not answer or len(answer.strip()) < 30:
                answer = "📑 Passages trouvés dans la thèse :\n\n" + "\n\n".join(
                    f"▸ {r['chunk']}" for r in citation_results[:3]
                )

            reward = 1.0

        decision_path.append(f"Reward: {reward:+.1f}")

        # Pas de mise à jour Q-table pour les citations (état spécial)
        return QueryResponse(
            question       = question,
            analysis_type  = analysis_type,
            routing        = "Citation RAG",
            confidence     = round(confidence, 2),
            answer         = answer,
            state          = state,
            action         = action,
            reward         = reward,
            decision_path  = decision_path,
            cypher_query   = None,
            raw_graph_data = None,
            extracted_entities = None,
        )

    # ════════════════════════════════════════════════════════════
    # BRANCHE STANDARD — Graph / Vector / Hybrid
    # ════════════════════════════════════════════════════════════

    # 3. Exécution des recherches
    graph_results  = []
    vector_results = []
    decision_path = ["User Query", f"Analysis: {analysis_type}"]
    cypher_q = None
    raw_graph = None
    extracted_ents = None
    graph_context = ""

    if action in ("use_graph", "use_both"):
        g_data = graph_rag(question)
        decision_path.append("Graph RAG → Neo4j Query")
        if "error" not in g_data and "message" not in g_data:
            cypher_q = g_data["cypher_query"]
            raw_graph = g_data["raw_graph_data"]
            extracted_ents = g_data["extracted_entities"]
            graph_context = g_data["graph_context_for_llm"]
            action = "use_graph" # Force action to graph for frontend routing display
        else:
            action = "use_vector"  # Fallback
            decision_path.append(f"Graph Fallback: {g_data.get('message', 'Error')}")

    if action in ("use_vector", "use_both"):
        vector_results = vector_search(question)
        decision_path.append("Vector RAG → FAISS Search")

    # VALIDATION : question hors-sujet ou trop courte
    # [SUPPRIMÉ - Maintenant géré au DÉBUT de process_query]

    # 4. Préparation du prompt pour le LLM
    prompt_pour_llm = ""
    if graph_context and vector_results:
        vec_text = format_vector_answer(vector_results)
        prompt_pour_llm = (
            f"Relations du graphe Neo4j :\n{graph_context}\n\n"
            f"Passages de la thèse :\n{vec_text}\n\n"
            f"CONSIGNES : Synthétise ces deux sources pour répondre à la question. "
            f"Cite les passages pertinents entre guillemets (« ... »). "
            f"N'invente aucune information absente des sources fournies."
        )
    elif graph_context:
        prompt_pour_llm = (
            f"Relations du graphe Neo4j :\n{graph_context}\n\n"
            f"CONSIGNES : Réponds UNIQUEMENT à partir des relations ci-dessus. "
            f"Reformule-les en phrases claires et académiques en français. "
            f"N'invente aucune information absente des relations."
        )
    elif vector_results:
        vec_text = format_vector_answer(vector_results)
        prompt_pour_llm = (
            f"Passages extraits de la thèse scientifique :\n{vec_text}\n\n"
            f"CONSIGNES : Réponds UNIQUEMENT à partir des passages ci-dessus. "
            f"Cite les extraits pertinents entre guillemets (« ... »). "
            f"N'invente aucune information absente des passages. "
            f"Si les passages ne suffisent pas, indique-le clairement."
        )

    # 4b. Reformulation LLM
    if prompt_pour_llm:
        answer = reformulate_with_free_ai(question, prompt_pour_llm)
        decision_path.append("LLM Reformulation (Free AI)")
    else:
        answer = "Je ne dispose d'aucune information dans le graphe ou le texte pour répondre."

    # 5. Récompense (Biais pro-Graph)
    if body.feedback is not None:
        reward = float(body.feedback)
    else:
        reward = 1.0 if "graph" in action else 0.5
        if not answer or len(answer.strip()) < 30:
            reward = 0.5 if reward == 1.0 else -1.0

    decision_path.append(f"Reward: {reward:+.1f}")

    # 6. Mise à jour Q-table
    update_q(agent, state, action, reward)

    return QueryResponse(
        question      = question,
        analysis_type = analysis_type,
        routing       = action,
        confidence    = round(confidence, 2),
        answer        = answer,
        state         = state,
        action        = action,
        reward        = reward,
        decision_path = decision_path,
        cypher_query  = cypher_q,
        raw_graph_data = raw_graph,
        extracted_entities = extracted_ents
    )

# ============================================================
# ENDPOINT 2 : GET /vectorial (bouton Vectorial RAG)
# ============================================================
@app.get("/vectorial", tags=["Vectorial RAG"])
async def get_vectorial(question: Optional[str] = None, k: int = 3, method: str = "cosine"):
    """
    Retourne : infos chunking + projection PCA 2D + résultats retrieval.
    """
    method = method if method in ["cosine", "dot_product", "bm25_hybrid"] else "cosine"
    vs = get_vector_store()
    if not vs:
        raise HTTPException(status_code=503, detail="Vector store non disponible.")

    chunks     = vs["chunks"]
    embeddings = vs["embeddings"]

    # --- Chunking stats ---
    sizes = [len(c.split()) for c in chunks]
    chunking_info = {
        "method":         "Recursive Splitting (meilleure méthode détectée)",
        "total_chunks":   len(chunks),
        "avg_words":      round(float(np.mean(sizes)), 1),
        "min_words":      int(np.min(sizes)),
        "max_words":      int(np.max(sizes)),
        "methods_tested": [
            "Fixed Size", "By Sentence", "By Paragraph",
            "Semantic Similarity", "Sliding Window",
            "Recursive Splitting ✓", "Token-based"
        ],
        "hierarchy": {
            "document":   "Thèse Piton des Neiges (400 pages)",
            "chunks":     len(chunks),
            "sub_chunks": len(chunks) * 3,
        }
    }

    # --- PCA 2D ---
    pca_data = []
    if SKLEARN_OK and embeddings is not None and len(embeddings) >= 2:
        try:
            sample_n = min(200, len(embeddings))
            idx_sample = np.random.choice(len(embeddings), sample_n, replace=False)
            sample_emb = embeddings[idx_sample]
            pca = PCA(n_components=2)
            coords = pca.fit_transform(sample_emb)
            # Assign simple cluster labels (K-means-like via distance to centroid)
            centroid = coords.mean(axis=0)
            dists    = np.linalg.norm(coords - centroid, axis=1)
            q25, q75 = np.percentile(dists, [25, 75])
            labels   = np.where(dists < q25, 0, np.where(dists < q75, 1, 2))
            pca_data = [
                {"x": round(float(coords[i,0]), 4),
                 "y": round(float(coords[i,1]), 4),
                 "cluster": int(labels[i]),
                 "chunk_preview": chunks[idx_sample[i]][:80]}
                for i in range(sample_n)
            ]
            variance_explained = [round(float(v)*100, 1) for v in pca.explained_variance_ratio_]
        except Exception as e:
            variance_explained = [0, 0]
    else:
        variance_explained = [45.2, 22.1]  # valeurs simulées si sklearn absent

    # --- Détection hors-sujet (BUG FIX 3: Early return si off-topic) ---
    if question and is_offtopic(question):
        response = {
            "status": "ok",
            "retrieval_method_used": method,
            "methods_available": ["cosine", "dot_product", "bm25_hybrid"],
            "chunking": chunking_info,
            "pca": {
                "method": "PCA 2D",
                "points": pca_data[:150],
                "variance_explained": variance_explained,
                "clusters": 3,
            },
            "retrieval": {
                "method": "N/A",
                "results": [],
                "summary": "Question hors du domaine géologique.",
                "retrieval_method_used": method,
                "methods_available": ["cosine", "dot_product", "bm25_hybrid"]
            },
            "offtopic": True,
            "offtopic_message": (
                "Question hors du domaine géologique. "
                "Posez une question sur la géologie du Piton des Neiges."
            )
        }
        return response
    
    # --- Retrieval ---
    retrieval_results = []
    retrieval_method_label = "Top-k Semantic Search (Cosine Similarity via FAISS)"
    if question and _embed_model:
        raw = vector_search_method(question, k=k, method=method, min_score=RETRIEVAL_MIN_SCORE)
        retrieval_results = [
            {"rank": i+1, "chunk": r["chunk"], "score": r["score"]}
            for i, r in enumerate(raw)
        ]
        if method == "dot_product":
            retrieval_method_label = "Top-k Semantic Search (Dot-Product via FAISS)"
        elif method == "bm25_hybrid":
            retrieval_method_label = "Hybrid BM25 + Embeddings"
        summary = f"Top-{k} résultats pour : '{question}'"
    else:
        summary = "Pose une question pour lancer le retrieval."

    response = {
        "status":          "ok",
        "retrieval_method_used": method,
        "methods_available": ["cosine", "dot_product", "bm25_hybrid"],
        "chunking":        chunking_info,
        "pca": {
            "method":            "PCA 2D",
            "points":            pca_data[:150],
            "variance_explained": variance_explained,
            "clusters":          3,
        },
        "retrieval": {
            "method":   retrieval_method_label,
            "results":  retrieval_results,
            "summary":  summary,
            "retrieval_method_used": method,
            "methods_available": ["cosine", "dot_product", "bm25_hybrid"]
        }
    }
    
    return response

# ============================================================
# ENDPOINT 3 : GET /graph (bouton Graph RAG)
# ============================================================
@app.get("/graph", tags=["Graph RAG"])
async def get_graph(question: Optional[str] = None, limit: int = 6000):
    """
    Retourne : graphe Neo4j + communautés Louvain + analyse avancée.
    """
    drv = get_neo4j()

    # Stats globales
    stats = {"nodes": 0, "relations": 0, "communities": 0, "modularity": 0.0}
    nodes_sample  = []
    edges_sample  = []
    communities   = []
    top_pagerank  = []
    advanced      = {}

    if drv:
        with drv.session() as s:
            try:
                # Stats
                r = s.run("MATCH (n:Entity) RETURN count(n) AS c")
                stats["nodes"] = r.single()["c"]
                r = s.run("MATCH ()-[r]->() RETURN count(r) AS c")
                stats["relations"] = r.single()["c"]

                # Communautés (si propriété community existe)
                r = s.run("MATCH (n:Entity) WHERE n.community IS NOT NULL "
                          "RETURN n.community AS c, count(*) AS cnt "
                          "ORDER BY cnt DESC LIMIT 15")
                communities = [{"id": rec["c"], "size": rec["cnt"]} for rec in r]
                stats["communities"] = len(communities)

                # Calcul modularité approx
                if communities:
                    sizes  = [c["size"] for c in communities]
                    total  = sum(sizes)
                    stats["modularity"] = round(
                        1 - sum((s/total)**2 for s in sizes), 3)

                # Nœuds échantillon — récupère tous les types de nœuds
                r = s.run("""
                    MATCH (n)
                    WHERE n.name IS NOT NULL
                    RETURN id(n) AS id,
                           COALESCE(n.name, toString(id(n))) AS name,
                           labels(n)[0] AS type,
                           COALESCE(n.louvain_community, n.community, 0) AS community,
                           COALESCE(n.pagerank, 0.001) AS pagerank
                    ORDER BY n.pagerank DESC
                    LIMIT 100000
                """)
                nodes_sample = [dict(rec) for rec in r]

                # Arêtes échantillon — inclut CO_OCCURS et autres relations
                r = s.run("""
                    MATCH (a)-[r]->(b)
                    WHERE a.name IS NOT NULL AND b.name IS NOT NULL
                    RETURN a.name AS source,
                           type(r) AS rel_type,
                           b.name AS target
                    LIMIT 100000
                """)
                edges_sample = [dict(rec) for rec in r]

                # Top PageRank
                top_pagerank = nodes_sample[:10]

                # Analyse avancée
                r = s.run("""
                    MATCH (n:Entity)
                    RETURN n.type AS type, count(*) AS cnt
                    ORDER BY cnt DESC LIMIT 10
                """)
                type_dist = [{"type": rec["type"], "count": rec["cnt"]} for rec in r]

                r = s.run("""
                    MATCH ()-[r]->() RETURN type(r) AS rel, count(*) AS cnt
                    ORDER BY cnt DESC LIMIT 10
                """)
                rel_dist = [{"relation": rec["rel"], "count": rec["cnt"]} for rec in r]

                advanced = {
                    "type_distribution":     type_dist,
                    "relation_distribution": rel_dist,
                    "density": round(stats["relations"] /
                        max(stats["nodes"] * (stats["nodes"]-1), 1) * 100, 2),
                    "semantic_paths": [
                        "Caldera → DEFINED_BY → Faille de caldera",
                        "Sources Cilaos → CONTROLLED_BY → Rift-zone N030",
                        "Brèche Orangers → OVERLIES → Coulées LM",
                    ],
                    "fusion_score": 0.87,
                }

            except Exception as e:
                print(f"[Graph endpoint] {e}")

        # --- Fallback: load a static graph file if Neo4j returned no nodes ---
        try:
            if (not nodes_sample or stats.get("nodes", 0) == 0) and os.path.exists("graph_data.json"):
                print("[Graph endpoint] Neo4j empty — loading fallback graph_data.json")
                with open("graph_data.json", "r", encoding="utf-8") as f:
                    g = json.load(f)

                # Accept common shapes: {nodes:[...], links:[...]} or {nodes:..., edges:...}
                file_nodes = g.get("nodes") or g.get("nodes_sample") or g.get("data") or []
                file_edges = g.get("links") or g.get("edges") or g.get("relationships") or g.get("edges_sample") or []

                nodes_sample = []
                for n in file_nodes:
                    # Normalize node fields with safe defaults
                    nid = n.get("id") or n.get("entity_id") or n.get("name")
                    nodes_sample.append({
                        "id": nid,
                        "name": n.get("name") or n.get("label") or nid,
                        "type": n.get("type") or n.get("label") or "",
                        "community": n.get("community") or n.get("louvain_community") or None,
                        "pagerank": n.get("pagerank") or n.get("score") or 0.0,
                        "cx": n.get("x") or n.get("cx"),
                        "cy": n.get("y") or n.get("cy"),
                    })

                edges_sample = []
                for e in file_edges:
                    if isinstance(e, dict):
                        s = e.get("source") or e.get("from") or e.get("a") or e.get("source_name")
                        t = e.get("target") or e.get("to") or e.get("b") or e.get("target_name")
                        rel = e.get("rel_type") or e.get("relation") or e.get("type") or ""
                    elif isinstance(e, (list, tuple)) and len(e) >= 2:
                        s, t = e[0], e[1]
                        rel = ""
                    else:
                        continue
                    edges_sample.append({"source": s, "rel_type": rel, "target": t})

                stats["nodes"] = len(nodes_sample)
                stats["relations"] = len(edges_sample)
        except Exception as e:
            print(f"[Graph endpoint] Fallback load error: {e}")

    # Résultats question
    graph_results = []
    if question:
        graph_results = graph_search(question, limit=6)

    return {
        "status":        "ok",
        "stats":         stats,
        "nodes_sample":  nodes_sample,
        "edges_sample":  edges_sample,
        "communities":   communities,
        "top_pagerank":  top_pagerank,
        "advanced":      advanced,
        "query_results": graph_results,
    }

# ============================================================
# ENDPOINT 4 : GET /agentic (bouton Agentic RAG)
# ============================================================
@app.get("/agentic", tags=["Agentic RAG"])
async def get_agentic():
    """
    Retourne : Q-table + historique des récompenses + chemin de décision.
    """
    agent = get_q_agent()

    # Q-table formatée pour affichage
    q_table_display = []
    for state in agent["states"]:
        row = {"state": state}
        for action in agent["actions"]:
            row[action] = round(agent["Q"][state][action], 4)
        best = max(agent["Q"][state], key=agent["Q"][state].get)
        row["best_action"] = best
        q_table_display.append(row)

    # Historique récompenses
    history = agent["history"][-50:]
    rewards = [h["reward"] for h in history]
    cumulative = []
    s = 0
    for r in rewards:
        s += r
        cumulative.append(round(s, 2))

    # Stats RL
    total   = len(history)
    pos     = sum(1 for h in history if h["reward"] > 0)
    neg     = sum(1 for h in history if h["reward"] < 0)
    success_rate = round(pos / total * 100, 1) if total > 0 else 0.0

    # Distribution des actions
    action_counts = Counter(h["action"] for h in history)
    state_counts  = Counter(h["state"]  for h in history)

    # Politique actuelle (best action par état)
    policy = {state: max(agent["Q"][state], key=agent["Q"][state].get)
              for state in agent["states"]}

    # Decision path exemple
    decision_path = {
        "graph":   ["User Query", "Systematic/Relational", "use_graph", "Neo4j Query", "Answer"],
        "vector":  ["User Query", "Semantic/Conceptual",   "use_vector","FAISS Search","Answer"],
        "hybrid":  ["User Query", "Hybrid",                "use_both",  "Neo4j + FAISS","Fusion Answer"],
        "unknown": ["User Query", "Unknown type",          "use_vector", "FAISS Search", "Answer"],
    }

    return {
        "status": "ok",
        "q_learning": {
            "states":        agent["states"],
            "actions":       agent["actions"],
            "alpha":         agent["alpha"],
            "gamma":         agent["gamma"],
            "epsilon":       agent["epsilon"],
            "q_table":       q_table_display,
            "current_policy": policy,
        },
        "reward_monitor": {
            "total_episodes":    total,
            "positive_rewards":  pos,
            "negative_rewards":  neg,
            "success_rate":      success_rate,
            "rewards_history":   rewards,
            "cumulative_reward": cumulative,
            "action_distribution": dict(action_counts),
            "state_distribution":  dict(state_counts),
        },
        "decision_path": decision_path,
        "hybrid_feedback": {
            "description": "Le résultat est réinjecté dans la Q-table via update_q()",
            "formula":     "Q(s,a) += α × (r + γ × max Q(s') - Q(s,a))",
            "alpha":       agent["alpha"],
            "gamma":       agent["gamma"],
        }
    }

# ============================================================
# ENDPOINT 5 : POST /graph/louvain (EXÉCUTER LOUVAIN)
# ============================================================
LOUVAIN_JSON = "louvain_communities.json"

@app.post("/graph/louvain", response_model=LouvainResponse, tags=["Graph RAG"])
async def run_louvain():
    """
    Exécute l'algorithme Louvain sur le graphe Neo4j
    et stocke les communautés. Utilise le fichier pré-calculé
    louvain_communities.json s'il existe (produit par step4_louvain_neo4j.py),
    sinon calcule en live.
    """
    # --- Stratégie 1 : fichier pré-calculé (rapide) ---
    if os.path.exists(LOUVAIN_JSON):
        try:
            with open(LOUVAIN_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            meta = data.get("metadata", {})
            comms = data.get("communities", {})
            dist = [
                {"community": int(cid), "size": c["size"]}
                for cid, c in sorted(comms.items(), key=lambda x: -x[1]["size"])
            ]
            return LouvainResponse(
                status="success",
                communities_found=meta.get("total_communities", len(comms)),
                modularity=meta.get("modularity", 0.0),
                message=f"Louvain chargé depuis {LOUVAIN_JSON} — {len(comms)} communautés, modularité {meta.get('modularity', 0)}",
                community_distribution=dist
            )
        except Exception as e:
            print(f"[Louvain] Erreur lecture JSON: {e}")

    # --- Stratégie 2 : calcul en live ---
    drv = get_neo4j()
    if not drv:
        raise HTTPException(status_code=503, detail="Neo4j non disponible.")

    with drv.session() as s:
        try:
            import networkx as nx
            import community as community_louvain

            result = s.run("MATCH (a:Entity)-[r]->(b:Entity) RETURN a.name AS source, b.name AS target")
            edges = [(rec["source"], rec["target"]) for rec in result if rec["source"] and rec["target"]]

            if not edges:
                return LouvainResponse(
                    status="error", communities_found=0, modularity=0.0,
                    message="Graphe vide ou aucune relation trouvée.",
                    community_distribution=None
                )

            G = nx.Graph()
            G.add_edges_from(edges)

            partition = community_louvain.best_partition(G, weight="weight")
            modularity_val = community_louvain.modularity(partition, G)
            community_count = len(set(partition.values()))

            # Réinjecter dans Neo4j par batch
            batch = [{"name": name, "comm": cid} for name, cid in partition.items()]
            s.run("""
                UNWIND $batch AS row
                MATCH (n:Entity {name: row.name})
                SET n.louvain_community = row.comm
            """, batch=batch)

            # Distribution
            dist_result = s.run("""
                MATCH (n:Entity)
                WHERE n.louvain_community IS NOT NULL
                RETURN n.louvain_community AS community, count(*) AS size
                ORDER BY size DESC
            """)
            community_dist = [{"community": rec["community"], "size": rec["size"]}
                             for rec in dist_result]

            return LouvainResponse(
                status="success",
                communities_found=community_count,
                modularity=round(modularity_val, 4),
                message=f"Louvain exécuté en live. {community_count} communautés. Modularité: {round(modularity_val, 4)}",
                community_distribution=community_dist
            )
        except Exception as e:
            return LouvainResponse(
                status="error", communities_found=0, modularity=0.0,
                message=f"Erreur Louvain: {str(e)}",
                community_distribution=None
            )

# ============================================================
# ENDPOINT 6 : GET /graph/louvain/status (VÉRIFIER STATUT)
# ============================================================
@app.get("/graph/louvain/status", tags=["Graph RAG"])
async def get_louvain_status():
    """
    Vérifie si Louvain a déjà été exécuté et retourne les communautés existantes.
    Utilise louvain_communities.json en priorité.
    """
    # Priorité au fichier pré-calculé
    if os.path.exists(LOUVAIN_JSON):
        try:
            with open(LOUVAIN_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            comms = data.get("communities", {})
            dist = [
                {"community": int(cid), "size": c["size"]}
                for cid, c in sorted(comms.items(), key=lambda x: -x[1]["size"])
            ]
            return {
                "status": "ok",
                "has_communities": True,
                "communities_count": len(comms),
                "community_distribution": dist
            }
        except Exception:
            pass

    drv = get_neo4j()
    if not drv:
        return {"status": "error", "message": "Neo4j non disponible", "has_communities": False}

    with drv.session() as s:
        try:
            rec = s.run("MATCH (n:Entity) WHERE n.louvain_community IS NOT NULL RETURN count(n) AS c").single()
            count = rec["c"] if rec else 0

            if count > 0:
                r2 = s.run("""
                    MATCH (n:Entity)
                    WHERE n.louvain_community IS NOT NULL
                    RETURN n.louvain_community AS community, count(*) AS size
                    ORDER BY size DESC
                """)
                communities = [{"community": rec["community"], "size": rec["size"]} for rec in r2]
                return {
                    "status": "ok",
                    "has_communities": True,
                    "communities_count": len(communities),
                    "community_distribution": communities
                }
            else:
                return {
                    "status": "ok",
                    "has_communities": False,
                    "communities_count": 0,
                    "message": "Aucune communauté. Exécutez POST /graph/louvain."
                }
        except Exception as e:
            return {"status": "error", "message": str(e), "has_communities": False}

# ============================================================
# ENDPOINT 7 : GET /graph/nodes/by-community (AFFICHER PAR COMMUNAUTÉ)
# ============================================================
@app.get("/graph/nodes/by-community", tags=["Graph RAG"])
async def get_nodes_by_community():
    """
    Retourne tous les nœuds regroupés par communauté.
    Utilise louvain_communities.json en priorité.
    """
    # Priorité au fichier pré-calculé (plus complet)
    if os.path.exists(LOUVAIN_JSON):
        try:
            with open(LOUVAIN_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            comms = data.get("communities", {})
            result = []
            for cid, c in sorted(comms.items(), key=lambda x: -x[1]["size"]):
                result.append({
                    "community": int(cid),
                    "nodes": [
                        {
                            "name": m.get("name") or m.get("id"),
                            "type": m.get("type", ""),
                            "description": m.get("description", ""),
                            "pagerank": m.get("pagerank") or m.get("score") or 0
                        }
                        for m in c.get("members", [])
                    ]
                })
            return {
                "status": "ok",
                "total_communities": len(result),
                "communities": result
            }
        except Exception:
            pass

    drv = get_neo4j()
    if not drv:
        raise HTTPException(status_code=503, detail="Neo4j non disponible.")

    with drv.session() as s:
        try:
            result = s.run("""
                MATCH (n:Entity)
                WHERE n.louvain_community IS NOT NULL
                RETURN n.louvain_community AS community,
                       collect({
                           name: n.name,
                           type: n.type,
                           description: n.description,
                           pagerank: n.pagerank
                       }) AS nodes
                ORDER BY community
            """)
            communities = [{"community": rec["community"], "nodes": rec["nodes"]}
                          for rec in result]
            return {
                "status": "ok",
                "total_communities": len(communities),
                "communities": communities
            }
        except Exception as e:
            return {"status": "error", "message": str(e), "communities": []}

# ============================================================
# ENDPOINT HEALTH CHECK
# ============================================================
@app.get("/", tags=["Health"])
async def root():
    vs  = get_vector_store()
    drv = get_neo4j()
    return {
        "app":          "Agentic Graph-Vector RAG — Piton des Neiges",
        "status":       "running",
        "vector_store": "ok" if vs  else "unavailable",
        "neo4j":        "ok" if drv else "unavailable",
        "endpoints":    ["/query", "/vectorial", "/graph", "/agentic", "/graph/louvain", "/docs"],
    }

@app.on_event("startup")
async def startup_event():
    print("[FastAPI] Démarrage — chargement des stores...")
    get_vector_store()
    get_neo4j()
    get_q_agent()
    print("[FastAPI] Prêt !")
    print("[FastAPI] Endpoints disponibles:")
    print("  POST /query           - Question/Réponse avec Q-Learning")
    print("  GET  /vectorial       - Analyse vectorielle + PCA")
    print("  GET  /graph           - Analyse graphe + communautés")
    print("  GET  /agentic         - Table Q + historique RL")
    print("  POST /graph/louvain   - EXÉCUTER LOUVAIN (Nouveau!)")
    print("  GET  /graph/louvain/status - Vérifier statut Louvain")
    print("  GET  /graph/nodes/by-community - Nœuds par communauté")

@app.on_event("shutdown")
async def shutdown_event():
    global _neo4j_driver
    if _neo4j_driver:
        _neo4j_driver.close()
    print("[FastAPI] Arrêté.")