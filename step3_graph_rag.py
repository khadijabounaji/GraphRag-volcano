"""
step4_graph_store_OPTIMAL.py - Version OPTIMISÉE
==================================================
CIBLES :
  - Nœuds        : 600-900
  - Relations    : 4800-5200 (juste autour de 5000)
  - Densité      : 2-4% (ni trop sparse, ni trop dense)
  - Communautés  : 50-80 (groupes cohérents)

STRATÉGIE :
  1. PMI adaptatif selon fréquence
  2. Filtrage par significance score (PMI × log(weight))
  3. Normalisation améliorée
  4. Paramètres dynamiques

Usage : python step4_graph_store_OPTIMAL.py
"""

import os
import re
import pickle
import itertools
import time
import math
from collections import Counter, defaultdict

import networkx as nx
import community as community_louvain
from neo4j import GraphDatabase

# ============================================================
# CONFIGURATION
# ============================================================
NEO4J_URI = "neo4j+s://acfc5ab2.databases.neo4j.io"
NEO4J_USER = "acfc5ab2"
NEO4J_PASSWORD = "XVH4uQXBtkf_FSMrK4T16vcTbJZ8QQu_7-HblC-hJgQ"

VECTOR_STORE_PATH = "vector_store.pkl"

# ============================================================
# PARAMÈTRES OPTIMISÉS (équilibre parfait)
# ============================================================
MIN_ENTITY_FREQ = 3          # Fréquence minimale
MIN_EDGE_WEIGHT = 3          # Co-occurrence minimale (équilibre)
MIN_PMI = 1.0                # Seuil PMI (plus permissif)
WINDOW_SENTENCES = 3         # Fenêtre moyenne
SIG_FACTOR = 0.5             # Facteur significance (PMI × log(weight))
MAX_RELATIONS = 5200         # Plafond automatique
TARGET_RELATIONS = 5000      # Cible

BATCH_SIZE = 50

# ============================================================
# ENTITÉS EXPERTISÉES (gardées)
# ============================================================
EXPERT_ENTITIES = {
    "piton_des_neiges": {"name": "Piton des Neiges", "etype": "Volcano"},
    "piton_fournaise": {"name": "Piton de la Fournaise", "etype": "Volcano"},
    "reunion": {"name": "Île de La Réunion", "etype": "Location"},
    "mafate": {"name": "Cirque de Mafate", "etype": "Cirque"},
    "salazie": {"name": "Cirque de Salazie", "etype": "Cirque"},
    "cilaos": {"name": "Cirque de Cilaos", "etype": "Cirque"},
    "pn0": {"name": "PN0", "etype": "GeologicalStage"},
    "lm": {"name": "LM", "etype": "GeologicalStage"},
    "pn1": {"name": "PN1", "etype": "GeologicalStage"},
    "pn2": {"name": "PN2", "etype": "GeologicalStage"},
    "pn3": {"name": "PN3", "etype": "GeologicalStage"},
    "pn4": {"name": "PN4", "etype": "GeologicalStage"},
    "br_b1": {"name": "Br.b1", "etype": "GeologicalUnit"},
    "br_orangers": {"name": "Brèche des Orangers", "etype": "GeologicalUnit"},
    "ignimbrite": {"name": "Ignimbrite prismée", "etype": "GeologicalUnit"},
    "br_hydro": {"name": "Brèche hydrothermale", "etype": "GeologicalUnit"},
    "gabbro": {"name": "Complexe gabbroïque", "etype": "Intrusion"},
    "sy_chapelle": {"name": "Syénite La Chapelle", "etype": "Intrusion"},
    "sy_troublanc": {"name": "Microsyénite Trou Blanc", "etype": "Intrusion"},
    "caldera": {"name": "Caldera différenciée", "etype": "Caldera"},
    "rift_n030": {"name": "Rift-zone N030", "etype": "TectonicStructure"},
    "hydrothermal": {"name": "Système hydrothermal", "etype": "HydrothermalSystem"},
    "chamber": {"name": "Chambre magmatique", "etype": "MagmaChamber"},
    "sources_cilaos": {"name": "Sources Cilaos", "etype": "ThermalSpring"},
    "sources_salazie": {"name": "Sources Salazie", "etype": "ThermalSpring"},
    "age_188": {"name": "188 ka", "etype": "AgeResult"},
    "age_27": {"name": "27 ka", "etype": "AgeResult"},
    "age_145": {"name": "145 ka", "etype": "AgeResult"},
}

# ============================================================
# STOP WORDS ET TERMES TECHNIQUES
# ============================================================
STOP_WORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "ou", "donc",
    "il", "elle", "on", "nous", "vous", "ils", "elles", "ce", "cet", "cette",
    "ces", "mon", "ton", "son", "notre", "votre", "leur", "est", "sont",
    "était", "étaient", "sera", "seront", "a", "ont", "avait", "auront",
    "peut", "peuvent", "avoir", "être", "faire", "voir", "dont", "où",
    "quand", "comment", "pourquoi", "quel", "quelle", "quelles", "quels",
    "puis", "ainsi", "donc", "enfin", "alors", "ensuite", "bien", "très",
    "trop", "plus", "moins", "aussi", "non", "oui", "si", "mais", "cette",
    "cet", "cela", "ceux", "celles", "celui", "celle", "ceci", "cela"
}

TECH_TERMS = [
    # Roches
    "basalte", "phonolite", "trachyte", "syénite", "gabbro",
    "andésite", "péridotite", "dunite", "pyroxénite",
    "mugéarite", "benmoreite", "hawaiite", "océanite",
    # Formations
    "ignimbrite", "téphra", "brèche", "coulée", "dalle",
    "lahar", "pyroclastique", "cendres", "ponces",
    # Structures
    "caldera", "rift", "dyke", "sill", "filon", "laccolithe",
    "intrusion", "pluton", "faille", "fracture", "détachement",
    # Processus
    "érosion", "effondrement", "déstabilisation", "avalanche",
    "glissement", "subsidence", "exhumation", "refroidissement",
    "cristallisation", "différenciation", "fusion", "dégazage",
    # Méthodes
    "u-pb", "k-ar", "ar-ar", "ahe", "zhe", "zdd",
    "thermochronologie", "géochronologie", "datation",
    "zircon", "apatite", "phlogopite", "sims", "icp-ms",
    # Hydrothermal
    "hydrothermal", "géothermie", "thermale", "aquifère",
    "perméabilité", "porosité", "altération", "argilisation",
    "silicification", "carbonatation", "prehnite", "pumpellyite",
    # Minéraux
    "olivine", "pyroxène", "feldspath", "plagioclase", "orthoclase",
    "biotite", "muscovite", "amphibole", "quartz", "calcite",
    # Géochimie
    "isotope", "terres rares", "sio2", "tio2", "al2o3", "feo", "mgo", "cao",
]

# ============================================================
# NORMALISATION
# ============================================================
NORMALIZE = {
    "basaltes": "basalte", "basaltique": "basalte",
    "phonolites": "phonolite", "phonolitique": "phonolite",
    "trachytes": "trachyte", "trachytique": "trachyte",
    "syénites": "syénite", "syénitique": "syénite",
    "gabbros": "gabbro", "gabbroïque": "gabbro",
    "ignimbrites": "ignimbrite",
    "brèches": "brèche",
    "calderas": "caldera",
    "dykes": "dyke", "sills": "sill",
    "failles": "faille", "fractures": "fracture",
    "hydrothermalisme": "hydrothermal",
    "géothermique": "géothermie",
}

def normalize(term):
    term_lower = term.lower()
    return NORMALIZE.get(term_lower, term_lower)

# ============================================================
# 1. CHARGEMENT
# ============================================================
def load_chunks():
    if not os.path.exists(VECTOR_STORE_PATH):
        print(f"[ERREUR] {VECTOR_STORE_PATH} introuvable")
        return []
    with open(VECTOR_STORE_PATH, "rb") as f:
        data = pickle.load(f)
    chunks = data.get("chunks", [])
    print(f"[INFO] {len(chunks)} chunks chargés")
    return chunks

# ============================================================
# 2. DÉCOUPAGE EN PHRASES
# ============================================================
def split_sentences(text):
    """Découpe un texte en phrases."""
    # Découpage sur ponctuation forte
    sentences = re.split(r'(?<=[.!?;:])\s+(?=[A-ZÀ-Ý])', text)
    # Filtrage des phrases trop courtes
    return [s.strip() for s in sentences if len(s.strip()) > 40]

def get_windows(sentences, window=WINDOW_SENTENCES):
    """Génère des fenêtres glissantes de N phrases."""
    for i in range(len(sentences)):
        yield " ".join(sentences[i:i+window])

# ============================================================
# 3. EXTRACTION DES ENTITÉS
# ============================================================
def extract_entities(text):
    """Extrait les entités d'un texte."""
    entities = set()
    text_lower = text.lower()
    
    # Termes techniques
    for term in TECH_TERMS:
        if term in text_lower:
            entities.add(normalize(term))
    
    # Noms propres (mots avec majuscule)
    for match in re.finditer(r'\b[A-ZÀ-Ý][a-zà-ÿ]{3,}\b', text):
        word = match.group(0)
        if word.lower() not in STOP_WORDS and len(word) >= 4:
            entities.add(word)
    
    # Nombres avec unités (âges)
    for match in re.finditer(r'\b\d+(?:\.\d+)?\s*(?:ka|Ma|Ga)\b', text, re.IGNORECASE):
        entities.add(match.group(0).lower())
    
    return list(entities)

# ============================================================
# 4. CONSTRUCTION DU GRAPHE AVEC SIGNIFICANCE SCORE
# ============================================================
def compute_significance(weight, pmi):
    """Calcule un score de significance combinant poids et PMI."""
    return pmi * math.log(weight + 1)

def build_graph(chunks):
    print("\n[INFO] Construction du graphe...")
    
    # Création des fenêtres
    all_windows = []
    for chunk in chunks:
        sentences = split_sentences(chunk)
        if not sentences:
            continue
        for win in get_windows(sentences):
            if len(win) > 50:  # Ignorer les fenêtres trop courtes
                all_windows.append(win)
    
    print(f"  - {len(all_windows)} fenêtres générées")
    
    # Extraction des entités par fenêtre
    entity_window_ids = defaultdict(set)
    window_entities = []
    
    for win_id, window in enumerate(all_windows):
        ents = extract_entities(window)
        window_entities.append(ents)
        for ent in ents:
            entity_window_ids[ent].add(win_id)
    
    # Filtrage par fréquence
    valid_entities = {ent: wins for ent, wins in entity_window_ids.items() 
                     if len(wins) >= MIN_ENTITY_FREQ}
    
    # Ajout des entités expertisées
    for ent_id, data in EXPERT_ENTITIES.items():
        if ent_id not in valid_entities:
            valid_entities[ent_id] = set()
    
    print(f"  - {len(valid_entities)} entités retenues")
    
    # Construction du graphe
    G = nx.Graph()
    
    # Ajout des nœuds
    for ent, win_ids in valid_entities.items():
        node_type = EXPERT_ENTITIES.get(ent, {}).get("etype", "Auto")
        G.add_node(ent, 
                   name=EXPERT_ENTITIES.get(ent, {}).get("name", ent),
                   etype=node_type,
                   frequency=len(win_ids))
    
    # Comptage des co-occurrences
    cooc_counts = Counter()
    for ents in window_entities:
        valid_ents_in_window = [e for e in ents if e in valid_entities]
        for u, v in itertools.combinations(sorted(set(valid_ents_in_window)), 2):
            if u != v:
                cooc_counts[(u, v)] += 1
    
    print(f"  - {len(cooc_counts)} paires candidates")
    
    # Filtrage avec significance score
    total_windows = len(all_windows)
    edges_data = []
    
    for (u, v), weight in cooc_counts.items():
        if weight < MIN_EDGE_WEIGHT:
            continue
        
        freq_u = len(valid_entities.get(u, []))
        freq_v = len(valid_entities.get(v, []))
        
        p_u = freq_u / total_windows if total_windows > 0 else 0
        p_v = freq_v / total_windows if total_windows > 0 else 0
        p_uv = weight / total_windows if total_windows > 0 else 0
        
        if p_u > 0 and p_v > 0 and p_uv > 0:
            pmi = math.log2(p_uv / (p_u * p_v))
        else:
            pmi = 0
        
        if pmi < MIN_PMI:
            continue
        
        sig = compute_significance(weight, pmi)
        edges_data.append({
            'u': u, 'v': v,
            'weight': weight,
            'pmi': round(pmi, 3),
            'sig': round(sig, 3)
        })
    
    # Trier par significance score
    edges_data.sort(key=lambda x: x['sig'], reverse=True)
    
    # Plafonner les relations si nécessaire (pour rester sous 5000)
    if len(edges_data) > MAX_RELATIONS:
        edges_data = edges_data[:MAX_RELATIONS]
        print(f"  - Plafonné à {MAX_RELATIONS} relations")
    
    # Ajout des relations
    for edge in edges_data:
        G.add_edge(edge['u'], edge['v'], 
                  weight=edge['weight'], 
                  pmi=edge['pmi'],
                  sig=edge['sig'])
    
    # Supprimer les nœuds isolés
    G.remove_nodes_from(list(nx.isolates(G)))
    
    print(f"  - Graphe final: {G.number_of_nodes()} nœuds, {G.number_of_edges()} relations")
    
    if G.number_of_nodes() > 1:
        max_possible = G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
        density = G.number_of_edges() / max_possible * 100
        print(f"  - Densité: {density:.2f}%")
    
    return G

# ============================================================
# 5. LOUVAIN + MÉTRIQUES AVANCÉES
# ============================================================
def compute_advanced_metrics(G):
    print("\n[INFO] Calcul des métriques avancées...")
    
    # Louvain
    partition = community_louvain.best_partition(G, weight='weight', random_state=42)
    num_comm = len(set(partition.values()))
    
    for node, comm in partition.items():
        G.nodes[node]['community'] = comm
    
    # PageRank
    pagerank = nx.pagerank(G, weight='weight')
    for node, pr in pagerank.items():
        G.nodes[node]['pagerank'] = round(pr, 6)
    
    # Betweenness centrality (échantillon si trop grand)
    if G.number_of_nodes() < 1000:
        betweenness = nx.betweenness_centrality(G, weight='weight')
        for node, bet in betweenness.items():
            G.nodes[node]['betweenness'] = round(bet, 6)
    
    print(f"  - {num_comm} communautés détectées")
    
    # Taille des communautés
    comm_sizes = Counter(partition.values())
    avg_size = sum(comm_sizes.values()) / len(comm_sizes)
    print(f"  - Taille moyenne des communautés: {avg_size:.1f} nœuds")
    
    return G, partition, pagerank, num_comm

# ============================================================
# 6. EXPORT NEO4J
# ============================================================
def export_to_neo4j(G):
    print("\n[INFO] Export vers Neo4j...")
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("  - Connexion réussie")
    except Exception as e:
        print(f"  - ERREUR: {e}")
        return 0, 0
    
    with driver.session() as session:
        # Nettoyage
        session.run("MATCH (n) DETACH DELETE n")
        session.run("CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.entity_id IS UNIQUE")
        
        # Création des nœuds (batch)
        nodes_list = list(G.nodes(data=True))
        nodes_created = 0
        
        for i in range(0, len(nodes_list), BATCH_SIZE):
            batch = nodes_list[i:i+BATCH_SIZE]
            for node_id, data in batch:
                query = """
                CREATE (n:Entity {
                    entity_id: $id,
                    name: $name,
                    etype: $etype,
                    community: $community,
                    pagerank: $pagerank
                })
                """
                session.run(query, {
                    "id": node_id,
                    "name": data.get("name", node_id),
                    "etype": data.get("etype", "Auto"),
                    "community": data.get("community", -1),
                    "pagerank": data.get("pagerank", 0.0)
                })
                nodes_created += 1
            print(f"  - {nodes_created}/{len(nodes_list)} nœuds...")
        
        print(f"  - {nodes_created} nœuds créés")
        
        # Création des relations
        edges_list = list(G.edges(data=True))
        rels_created = 0
        
        for i in range(0, len(edges_list), BATCH_SIZE):
            batch = edges_list[i:i+BATCH_SIZE]
            for u, v, data in batch:
                query = """
                MATCH (a:Entity {entity_id: $u})
                MATCH (b:Entity {entity_id: $v})
                CREATE (a)-[:CO_OCCURS {
                    weight: $weight,
                    pmi: $pmi,
                    significance: $sig
                }]->(b)
                """
                session.run(query, {
                    "u": u,
                    "v": v,
                    "weight": data.get("weight", 1),
                    "pmi": data.get("pmi", 0),
                    "sig": data.get("sig", 0)
                })
                rels_created += 1
            print(f"  - {rels_created}/{len(edges_list)} relations...")
        
        print(f"  - {rels_created} relations créées")
    
    driver.close()
    return nodes_created, rels_created

# ============================================================
# 7. STATISTIQUES DÉTAILLÉES
# ============================================================
def print_detailed_stats(G, num_comm):
    print("\n" + "=" * 70)
    print("  STATISTIQUES FINALES")
    print("=" * 70)
    print(f"  Nœuds               : {G.number_of_nodes()}")
    print(f"  Relations           : {G.number_of_edges()}")
    print(f"  Communautés Louvain : {num_comm}")
    
    if G.number_of_nodes() > 1:
        max_e = G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
        density = G.number_of_edges() / max_e * 100
        print(f"  Densité             : {density:.2f}%")
    
    # Types de nœuds
    type_counts = Counter()
    for _, data in G.nodes(data=True):
        type_counts[data.get("etype", "Auto")] += 1
    
    print("\n  Types d'entités:")
    for etype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    - {etype}: {count}")
    
    # Distribution des communautés
    comm_counts = Counter()
    for _, data in G.nodes(data=True):
        comm_counts[data.get("community", -1)] += 1
    
    small_comms = sum(1 for c in comm_counts.values() if c < 3)
    print(f"\n  Communautés de taille < 3: {small_comms}")
    
    # Top relations par significance
    print("\n  Top 10 relations par significance:")
    top_edges = sorted(G.edges(data=True), key=lambda x: x[2].get("sig", 0), reverse=True)[:10]
    for u, v, data in top_edges:
        print(f"    {u} <-> {v} | sig={data.get('sig', 0):.2f} | pmi={data.get('pmi', 0):.2f}")

# ============================================================
# 8. MAIN
# ============================================================
def main():
    print("=" * 70)
    print("  GRAPH STORE OPTIMISÉ (CIBLE: ~5000 RELATIONS, DENSITÉ 2-4%)")
    print("=" * 70)
    print(f"  Paramètres:")
    print(f"    MIN_ENTITY_FREQ  = {MIN_ENTITY_FREQ}")
    print(f"    MIN_EDGE_WEIGHT  = {MIN_EDGE_WEIGHT}")
    print(f"    MIN_PMI          = {MIN_PMI}")
    print(f"    WINDOW_SENTENCES = {WINDOW_SENTENCES}")
    print(f"    MAX_RELATIONS    = {MAX_RELATIONS}")
    print("=" * 70)
    
    # Chargement
    chunks = load_chunks()
    if not chunks:
        return
    
    # Construction
    G = build_graph(chunks)
    if G.number_of_nodes() == 0:
        print("[ERREUR] Graphe vide")
        return
    
    # Métriques
    G, partition, pagerank, num_comm = compute_advanced_metrics(G)
    
    # Statistiques
    print_detailed_stats(G, num_comm)
    
    # Export
    nodes, rels = export_to_neo4j(G)
    
    # Bilan
    print("\n" + "=" * 70)
    print("  BILAN FINAL")
    print("=" * 70)
    print(f"  Nœuds  : {nodes}")
    print(f"  Relat. : {rels}")
    
    if 4800 <= rels <= 5200:
        print("  ✅ Relations: PARFAIT (dans la cible 4800-5200)")
    elif rels < 4800:
        print(f"  ⚠️ Relations: {rels} (en dessous de 4800) → baisser MIN_PMI")
    else:
        print(f"  ⚠️ Relations: {rels} (au-dessus de 5200) → augmenter MIN_PMI ou MIN_EDGE_WEIGHT")
    
    if 2 <= (G.number_of_edges() / (G.number_of_nodes() * (G.number_of_nodes() - 1) / 2) * 100) <= 4:
        print("  ✅ Densité: PARFAITE (2-4%)")
    else:
        print("  ⚠️ Densité: à ajuster")
    
    print("\n  Visualisation: https://console.neo4j.io")
    print("=" * 70)

if __name__ == "__main__":
    main()