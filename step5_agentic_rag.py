"""
step5_agentic_rag_FINAL.py — Version CORRIGÉE (exploitation pure)
======================================================================
CORRECTIONS :
  1. epsilon = 0 (pas d'exploration aléatoire)
  2. Meilleure détection des réponses pertinentes
  3. Récompenses plus généreuses pour les bonnes réponses
  4. Initialisation des Q-valeurs favorisant les bonnes actions

Usage : python step5_agentic_rag_FINAL.py
"""

import os
import re
import pickle
import random
import json
import numpy as np
from collections import defaultdict

# ============================================================
# IMPORTS
# ============================================================
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    VECTOR_OK = True
except ImportError:
    VECTOR_OK = False
    print("[WARN] FAISS ou SentenceTransformer non installé")

try:
    from neo4j import GraphDatabase
    NEO4J_OK = True
except ImportError:
    NEO4J_OK = False
    print("[WARN] neo4j non installé")

# ============================================================
# CONFIGURATION
# ============================================================
NEO4J_URI = "neo4j+s://acfc5ab2.databases.neo4j.io"
NEO4J_USER = "acfc5ab2"
NEO4J_PASSWORD = "XVH4uQXBtkf_FSMrK4T16vcTbJZ8QQu_7-HblC-hJgQ"
VECTOR_STORE_PATH = "vector_store.pkl"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ============================================================
# PARAMETRES Q-LEARNING - MODE EXPLOITATION PURE
# ============================================================
ALPHA = 0.3
GAMMA = 0.9
EPSILON = 0.0           # ← PAS D'EXPLORATION ALÉATOIRE
EPSILON_DECAY = 1.0
MIN_EPSILON = 0.0
Q_SAVE_PATH = "q_table.json"

# ============================================================
# MOTS-CLES PAR CATÉGORIE
# ============================================================
GRAPH_KEYWORDS = [
    "relation", "lien", "rapport", "connecté", "relie",
    "repose sur", "surmonte", "recoupe", "intrude",
    "présent dans", "localisé", "situé",
    "caldera", "rift", "faille", "brèche", "cirque",
    "mafate", "salazie", "cilaos", "pn0", "pn1", "pn2", "pn3", "pn4",
]

VECTOR_KEYWORDS = [
    "explique", "explication", "décris", "description",
    "comment", "pourquoi", "qu'est-ce", "définition",
    "principe", "mécanisme", "processus", "fonctionne",
    "hydrothermal", "géothermie", "géochronologie",
    "thermochronologie", "volcanisme", "éruption",
    "différenciation", "magma", "point chaud",
    "u-pb", "ar-ar", "k-ar", "zdd",
]

# Mots-clés pour valider les réponses
VALIDATION_KEYWORDS = {
    "zdd": ["zdd", "zircon", "double dating", "u-th/he", "u-pb", "188 ka", "188.2"],
    "caldera": ["caldera", "effondrement", "ignimbrite", "188 ka", "rift", "n030"],
    "hydrothermal": ["hydrothermal", "source thermale", "chambre magmatique", "27 ka"],
    "differentiation": ["différenciation", "magmatique", "cristallisation", "fractionnée"],
    "brèche": ["brèche", "avalanche", "débris", "br.b1", "orangers"],
}


# ============================================================
# 1. VECTOR STORE
# ============================================================
class VectorStore:
    def __init__(self, path=VECTOR_STORE_PATH):
        self.chunks = []
        self.index = None
        self.model = None
        self.available = False
        self.dim = None

        if not VECTOR_OK:
            return
        if not os.path.exists(path):
            print(f"[VectorStore] {path} introuvable.")
            return

        print(f"[VectorStore] Chargement...")
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.chunks = data.get("chunks", [])
        embeddings = data.get("embeddings", None)
        
        try:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
        except Exception as e:
            print(f"[VectorStore] Erreur modèle: {e}")
            return

        if embeddings is not None:
            vecs = np.array(embeddings, dtype="float32")
        else:
            vecs = self.model.encode(self.chunks, show_progress_bar=True)
            vecs = np.array(vecs, dtype="float32")

        if len(vecs) == 0:
            return

        self.dim = vecs.shape[1]
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(vecs)
        self.available = True
        print(f"[VectorStore] OK ({self.index.ntotal} vecteurs)")

    def search(self, question, k=3):
        if not self.available or self.model is None or self.index is None:
            return []
        try:
            q_vec = self.model.encode([question])
            q_vec = np.array(q_vec, dtype="float32")
            if q_vec.shape[1] != self.dim:
                if q_vec.shape[1] > self.dim:
                    q_vec = q_vec[:, :self.dim]
                else:
                    pad = np.zeros((q_vec.shape[0], self.dim - q_vec.shape[1]), dtype="float32")
                    q_vec = np.hstack([q_vec, pad])
            D, I = self.index.search(q_vec, min(k, self.index.ntotal))
            results = []
            for i, d in zip(I[0], D[0]):
                if 0 <= i < len(self.chunks):
                    results.append({"chunk": self.chunks[i], "distance": float(d)})
            return results
        except Exception as e:
            print(f"[VectorStore] Erreur: {e}")
            return []


# ============================================================
# 2. GRAPH STORE
# ============================================================
class GraphStore:
    def __init__(self):
        self.driver = None
        self.available = False
        if not NEO4J_OK:
            return
        try:
            self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
            self.driver.verify_connectivity()
            self.available = True
            print("[GraphStore] OK")
        except Exception as e:
            print(f"[GraphStore] Erreur: {e}")

    def close(self):
        if self.driver:
            self.driver.close()

    def query_entities(self, question, limit=8):
        if not self.available:
            return []
        keywords = [w for w in re.findall(r'\b[a-zA-ZÀ-ÿ]{4,}\b', question)
                    if w.lower() not in {"dans","pour","avec","entre","sont","est","les","des"}]
        if not keywords:
            return self._top_entities(limit)
        
        results = []
        with self.driver.session() as session:
            for kw in keywords[:4]:
                try:
                    query = """
                        MATCH (e:Entity)
                        WHERE toLower(e.name) CONTAINS toLower($kw)
                        OPTIONAL MATCH (e)-[r]->(n:Entity)
                        RETURN e.name AS name, e.etype AS type, e.community AS community,
                               collect(DISTINCT {rel: type(r), neighbor: n.name})[..3] AS relations
                        LIMIT $lim
                    """
                    result = session.run(query, kw=kw, lim=limit)
                    for record in result:
                        results.append({
                            "name": record["name"],
                            "type": record["type"],
                            "community": record["community"],
                            "relations": record["relations"]
                        })
                except Exception:
                    pass
        
        if not results:
            return self._top_entities(limit)
        
        seen = set()
        unique = []
        for r in results:
            if r["name"] not in seen:
                seen.add(r["name"])
                unique.append(r)
        return unique[:limit]

    def _top_entities(self, limit):
        if not self.available:
            return []
        with self.driver.session() as session:
            try:
                result = session.run("""
                    MATCH (e:Entity)
                    RETURN e.name AS name, e.etype AS type, e.community AS community
                    ORDER BY e.pagerank DESC LIMIT $lim
                """, lim=limit)
                return [{"name": record["name"], "type": record["type"], 
                         "community": record["community"], "relations": []} 
                        for record in result]
            except Exception:
                return []


# ============================================================
# 3. Q-LEARNING - MODE EXPLOITATION PURE
# ============================================================
class QLearningAgent:
    STATES = ["graph", "vector", "hybrid", "unknown"]
    ACTIONS = ["use_graph", "use_vector", "use_both", "fallback"]

    def __init__(self, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_initial = epsilon
        self.Q = {s: {a: 0.0 for a in self.ACTIONS} for s in self.STATES}
        
        # Initialisation favorisant les bonnes actions
        self.Q["graph"]["use_graph"] = 1.0      # Graph RAG pour questions relationnelles
        self.Q["graph"]["use_vector"] = 0.2
        self.Q["graph"]["use_both"] = 0.5
        
        self.Q["vector"]["use_vector"] = 1.0    # Vector RAG pour questions sémantiques
        self.Q["vector"]["use_graph"] = 0.2
        self.Q["vector"]["use_both"] = 0.5
        
        self.Q["hybrid"]["use_both"] = 1.0      # Hybride pour les deux
        self.Q["hybrid"]["use_graph"] = 0.3
        self.Q["hybrid"]["use_vector"] = 0.3
        
        self.Q["unknown"]["fallback"] = 1.0
        self.Q["unknown"]["use_vector"] = 0.5
        
        self.history = []
        self.episodes = 0
        self._load()

    def detect_state(self, question):
        q = question.lower()
        g = sum(1 for kw in GRAPH_KEYWORDS if kw in q)
        v = sum(1 for kw in VECTOR_KEYWORDS if kw in q)
        if g > 0 and v > 0:
            return "hybrid"
        if g > 0:
            return "graph"
        if v > 0:
            return "vector"
        return "unknown"

    def choose_action(self, state):
        """Choix déterministe (pas d'exploration aléatoire)"""
        q_values = self.Q[state]
        max_q = max(q_values.values())
        best_actions = [a for a, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    def update(self, state, action, reward):
        current_q = self.Q[state][action]
        best_next = max(self.Q[state].values())
        
        new_q = current_q + self.alpha * (reward + self.gamma * best_next - current_q)
        self.Q[state][action] = new_q
        
        self.history.append({"state": state, "action": action, "reward": reward})
        self.episodes += 1
        
        self._save()
        print(f"  [RL] {state} → {action} | reward={reward:+.1f} | Q={new_q:.3f}")

    def print_qtable(self):
        print("\n  " + "=" * 65)
        print("  Q-TABLE APPRISE")
        print("  " + "=" * 65)
        print(f"  {'État':<10} " + "".join(f"{a:^14}" for a in self.ACTIONS))
        print("  " + "-" * 65)
        for s in self.STATES:
            row = f"  {s:<10}"
            for a in self.ACTIONS:
                row += f"  {self.Q[s][a]:>+8.3f}   "
            print(row)
        print("  " + "-" * 65)

    def _save(self):
        try:
            with open(Q_SAVE_PATH, "w", encoding="utf-8") as f:
                json.dump({
                    "Q": self.Q,
                    "epsilon": self.epsilon,
                    "episodes": self.episodes,
                    "history": self.history[-200:]
                }, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _load(self):
        if os.path.exists(Q_SAVE_PATH):
            try:
                with open(Q_SAVE_PATH, "r", encoding="utf-8") as f:
                    d = json.load(f)
                self.Q = d.get("Q", self.Q)
                self.epsilon = d.get("epsilon", self.epsilon_initial)
                self.episodes = d.get("episodes", 0)
                self.history = d.get("history", [])
            except Exception:
                pass


# ============================================================
# 4. AGENT PRINCIPAL CORRIGÉ
# ============================================================
class AgenticGraphVectorRAG:
    def __init__(self):
        print("=" * 65)
        print("  AGENTIC GRAPH-VECTOR RAG (VERSION CORRIGÉE)")
        print("  Mode: Exploitation pure (pas d'actions aléatoires)")
        print("=" * 65)
        self.vector_store = VectorStore()
        self.graph_store = GraphStore()
        self.rl_agent = QLearningAgent()
        print(f"\n  Vector Store : {'✓' if self.vector_store.available else '✗'}")
        print(f"  Graph Store  : {'✓' if self.graph_store.available else '✗'}")
        print(f"  Q-Learning   : ✓ (ε={self.rl_agent.epsilon})\n")

    def _evaluate_answer_quality(self, answer, question):
        """Évalue la qualité de la réponse - plus généreuse"""
        if not answer or len(answer.strip()) < 20:
            return -0.5
        
        answer_lower = answer.lower()
        
        # Bonus si la réponse contient des informations substantielles
        if len(answer.split()) > 50:
            return 1.0
        
        # Vérification des concepts clés
        question_lower = question.lower()
        
        if "zdd" in question_lower:
            keywords = VALIDATION_KEYWORDS["zdd"]
        elif "caldera" in question_lower:
            keywords = VALIDATION_KEYWORDS["caldera"]
        elif "hydrothermal" in question_lower or "thermale" in question_lower:
            keywords = VALIDATION_KEYWORDS["hydrothermal"]
        elif "différenciation" in question_lower or "magmatique" in question_lower:
            keywords = VALIDATION_KEYWORDS["differentiation"]
        else:
            keywords = []
        
        if keywords:
            found = sum(1 for kw in keywords if kw in answer_lower)
            if found >= 2:
                return 1.0
            elif found == 1:
                return 0.5
        
        return 0.0

    def _format_graph(self, results):
        if not results:
            return ""
        lines = []
        for r in results[:5]:
            line = f"• {r.get('name', '?')}"
            if r.get("type"):
                line += f" [{r['type']}]"
            if r.get("community") is not None:
                line += f" (communauté {r['community']})"
            lines.append(line)
            for rel in (r.get("relations") or [])[:2]:
                if rel.get("rel") and rel.get("neighbor"):
                    lines.append(f"    └── {rel['rel']} → {rel['neighbor']}")
        return "\n".join(lines) if lines else ""

    def _format_vector(self, results):
        if not results:
            return ""
        formatted = []
        for i, r in enumerate(results[:2]):
            chunk = r.get("chunk", "")
            if chunk:
                formatted.append(f"[Source {i+1}] {chunk[:400]}...")
        return "\n\n".join(formatted)

    def answer(self, question, verbose=True):
        if verbose:
            print(f"\n{'─'*65}")
            print(f"  ❓ {question}")

        state = self.rl_agent.detect_state(question)
        action = self.rl_agent.choose_action(state)

        if verbose:
            print(f"  🧠 État={state} → Action={action}")

        graph_text = ""
        vector_text = ""

        if action in ("use_graph", "use_both"):
            graph_text = self._format_graph(self.graph_store.query_entities(question))

        if action in ("use_vector", "use_both"):
            vector_text = self._format_vector(self.vector_store.search(question, k=2))

        # Construction de la réponse finale
        if action == "use_graph" and graph_text:
            final = graph_text
        elif action == "use_vector" and vector_text:
            final = vector_text
        elif action == "use_both":
            parts = []
            if graph_text:
                parts.append("📊 ENTITÉS CONNEXES:\n" + graph_text)
            if vector_text:
                parts.append("📄 EXTRAITS DE LA THÈSE:\n" + vector_text)
            final = "\n\n".join(parts) if parts else ""
        else:
            final = ""

        auto_reward = self._evaluate_answer_quality(final, question)
        
        if verbose:
            print(f"  🤖 Réponse:\n    {final[:400]}{'...' if len(final)>400 else ''}")
            print(f"  🎁 Récompense: {auto_reward:+.1f}")

        return {
            "question": question,
            "state": state,
            "action": action,
            "reward": auto_reward,
            "answer": final
        }

    def interactive_session(self):
        print("\n" + "=" * 65)
        print("  MODE INTERACTIF")
        print("  L'agent répond directement (pas d'exploration aléatoire)")
        print("=" * 65)

        while True:
            q = input("\n  ❓ Question: ").strip()
            if q.lower() in ("exit", "quit", "q"):
                break
            if q.lower() == "stats":
                self.rl_agent.print_qtable()
                continue
            if not q:
                continue
            
            result = self.answer(q)
            self.rl_agent.update(result["state"], result["action"], result["reward"])
            
            print(f"\n  📝 Réponse finale:\n{result['answer']}")

    def run_demo(self):
        print("\n" + "=" * 65)
        print("  DÉMONSTRATION AUTOMATIQUE")
        print("=" * 65)
        
        test_questions = [
            "Qu'est-ce que la méthode ZDD ?",
            "Quelle est la relation entre caldera et rift zone N030 ?",
            "Explique le fonctionnement du système hydrothermal",
        ]
        
        for q in test_questions:
            result = self.answer(q)
            self.rl_agent.update(result["state"], result["action"], result["reward"])
            print(f"\n  📝 Réponse:\n{result['answer'][:400]}")
            print()
        
        self.rl_agent.print_qtable()

    def close(self):
        self.graph_store.close()


# ============================================================
# 5. MAIN
# ============================================================
def main():
    agent = AgenticGraphVectorRAG()
    
    print("\n" + "=" * 65)
    print("  MENU PRINCIPAL")
    print("=" * 65)
    print("  1. Lancer la démonstration automatique")
    print("  2. Mode interactif")
    print("  3. Quitter")
    print("=" * 65)
    
    choix = input("\n  Votre choix (1/2/3): ").strip()
    
    if choix == "1":
        agent.run_demo()
    elif choix == "2":
        agent.interactive_session()
    else:
        print("  Au revoir !")
    
    agent.close()
    print(f"\n✅ Terminé. Q-table sauvegardée → {Q_SAVE_PATH}")


if __name__ == "__main__":
    main()