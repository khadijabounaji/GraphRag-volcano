#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
╔══════════════════════════════════════════════════════════════╗
║  STEP 4 — Louvain Community Detection  (Neo4j Aura)        ║
║                                                              ║
║  Pipeline :                                                  ║
║    1. Extraire le graphe Entity depuis Neo4j Aura            ║
║    2. Appliquer Louvain (networkx + python-louvain)          ║
║    3. Réinjecter les communautés dans Neo4j                  ║
║    4. Créer des nœuds Community + relations BELONGS_TO       ║
║    5. Exporter en JSON pour Agentic RAG                      ║
║                                                              ║
║  Contraintes :                                               ║
║    - SANS gds.* (compatible Neo4j Aura Free)                 ║
║    - Idempotent (réexécutable sans casser)                   ║
║    - Batch mode pour gros graphes                            ║
║                                                              ║
║  Dépendances :                                               ║
║    pip install neo4j networkx python-louvain                 ║
╚══════════════════════════════════════════════════════════════╝
"""

import json
import time
from collections import Counter, defaultdict

import networkx as nx
import community as community_louvain
from neo4j import GraphDatabase

# ============================================================
# 1. CONFIGURATION NEO4J AURA
# ============================================================
NEO4J_URI      = "neo4j+s://acfc5ab2.databases.neo4j.io"
NEO4J_USER     = "acfc5ab2"
NEO4J_PASSWORD = "XVH4uQXBtkf_FSMrK4T16vcTbJZ8QQu_7-HblC-hJgQ"

BATCH_SIZE     = 500          # Taille des batchs pour la réinjection
OUTPUT_JSON    = "louvain_communities.json"

print("=" * 65)
print("  STEP 4 — Louvain Community Detection (Neo4j Aura)")
print("=" * 65)

# ============================================================
# 2. CONNEXION NEO4J
# ============================================================
print("\n[1/6] Connexion à Neo4j Aura...")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

try:
    driver.verify_connectivity()
    print("  ✓ Connexion réussie.")
except Exception as e:
    print(f"  ✗ Échec de connexion : {e}")
    raise SystemExit(1)

# ============================================================
# 3. EXTRACTION DU GRAPHE VERS NETWORKX
# ============================================================
print("\n[2/6] Extraction du graphe Entity depuis Neo4j...")

def load_graph_from_neo4j():
    """
    Charge toutes les relations entre nœuds Entity.
    Retourne un graphe NetworkX non-dirigé avec poids optionnel.
    """
    G = nx.Graph()

    # Récupérer les nœuds avec leurs propriétés utiles
    node_query = """
    MATCH (e:Entity)
    RETURN e.name AS name,
           e.id   AS id,
           e.type AS type,
           e.category AS category,
           e.description AS description
    """

    # Récupérer les arêtes avec type de relation (pour pondération)
    edge_query = """
    MATCH (a:Entity)-[r]->(b:Entity)
    RETURN a.name AS source,
           b.name AS target,
           type(r) AS rel_type
    """

    with driver.session() as session:
        # Nœuds
        node_count = 0
        for rec in session.run(node_query):
            node_name = rec["name"] or rec["id"]
            if node_name:
                G.add_node(node_name, **{
                    k: v for k, v in {
                        "type": rec["type"],
                        "category": rec["category"],
                        "description": rec["description"]
                    }.items() if v is not None
                })
                node_count += 1

        # Arêtes (avec pondération par nombre de relations)
        edge_count = 0
        edge_weights = defaultdict(int)
        for rec in session.run(edge_query):
            src, tgt = rec["source"], rec["target"]
            if src and tgt and src != tgt:
                edge_weights[(src, tgt)] += 1
                edge_count += 1

        for (src, tgt), weight in edge_weights.items():
            G.add_edge(src, tgt, weight=weight)

    print(f"  ✓ Graphe chargé : {G.number_of_nodes()} nœuds / {G.number_of_edges()} arêtes")
    return G


G = load_graph_from_neo4j()

if G.number_of_nodes() == 0:
    print("  ✗ Graphe vide — rien à analyser.")
    driver.close()
    raise SystemExit(0)

# ============================================================
# 4. ALGORITHME DE LOUVAIN
# ============================================================
print("\n[3/6] Application de l'algorithme de Louvain...")

t0 = time.time()

# Utiliser la pondération des arêtes si disponible
partition = community_louvain.best_partition(G, weight="weight", resolution=1.0)
modularity = community_louvain.modularity(partition, G, weight="weight")
num_communities = len(set(partition.values()))

elapsed = time.time() - t0

print(f"  ✓ {num_communities} communautés détectées")
print(f"  ✓ Modularité : {modularity:.4f}")
print(f"  ✓ Temps de calcul : {elapsed:.2f}s")

# Statistiques par communauté
community_sizes = Counter(partition.values())
print(f"\n  Distribution des communautés :")
for cid, size in community_sizes.most_common(10):
    print(f"    Communauté {cid} : {size} nœuds")
if num_communities > 10:
    print(f"    ... ({num_communities - 10} communautés supplémentaires)")

# ============================================================
# 5. RÉINJECTION DANS NEO4J (BATCH MODE)
# ============================================================
print(f"\n[4/6] Réinjection des communautés dans Neo4j (batch={BATCH_SIZE})...")

def write_communities_batch(partition, batch_size=BATCH_SIZE):
    """
    Écrit louvain_community sur chaque nœud Entity, par batch.
    Idempotent : écrase la valeur précédente.
    """
    items = list(partition.items())
    total = len(items)
    written = 0

    with driver.session() as session:
        for i in range(0, total, batch_size):
            batch = items[i:i + batch_size]
            params = [{"name": name, "community": cid} for name, cid in batch]

            session.run("""
                UNWIND $batch AS item
                MATCH (e:Entity {name: item.name})
                SET e.louvain_community = item.community
            """, batch=params)

            written += len(batch)
            print(f"    Batch {i // batch_size + 1} : {written}/{total} nœuds mis à jour")

    print(f"  ✓ {written} nœuds annotés avec louvain_community.")


write_communities_batch(partition)

# ============================================================
# 6. CRÉATION DES NŒUDS COMMUNITY + RELATIONS BELONGS_TO
# ============================================================
print("\n[5/6] Création des nœuds Community et relations BELONGS_TO...")

def create_community_nodes():
    """
    Crée/fusionne les nœuds Community à partir des propriétés louvain_community.
    Idempotent grâce à MERGE.
    """
    with driver.session() as session:
        # Supprimer les anciennes relations BELONGS_TO (idempotence)
        session.run("MATCH ()-[r:BELONGS_TO]->(:Community) DELETE r")
        # Supprimer les anciens nœuds Community (idempotence)
        session.run("MATCH (c:Community) DELETE c")

        # Recréer proprement
        session.run("""
            MATCH (e:Entity)
            WHERE e.louvain_community IS NOT NULL
            WITH e.louvain_community AS cid,
                 collect(e) AS members,
                 collect(DISTINCT e.category) AS cats,
                 collect(DISTINCT e.type) AS types
            CREATE (com:Community {
                community_id: cid,
                size: size(members),
                domains: [c IN cats WHERE c IS NOT NULL],
                types: [t IN types WHERE t IS NOT NULL],
                member_names: [m IN members | m.name]
            })
            WITH com, members
            UNWIND members AS e
            CREATE (e)-[:BELONGS_TO]->(com)
        """)

    # Vérification
    with driver.session() as session:
        result = session.run("MATCH (c:Community) RETURN count(c) AS cnt").single()
        rel_result = session.run("MATCH ()-[r:BELONGS_TO]->() RETURN count(r) AS cnt").single()
        print(f"  ✓ {result['cnt']} nœuds Community créés")
        print(f"  ✓ {rel_result['cnt']} relations BELONGS_TO créées")


create_community_nodes()

# ============================================================
# 7. EXPORT JSON POUR AGENTIC RAG
# ============================================================
print(f"\n[6/6] Export JSON → {OUTPUT_JSON}...")

def export_communities_json():
    """
    Exporte les communautés avec métadonnées pour intégration
    avec le pipeline Agentic RAG et le vector store.
    """
    communities = {}

    for node_name, cid in partition.items():
        cid_str = str(cid)
        if cid_str not in communities:
            communities[cid_str] = {
                "community_id": cid,
                "members": [],
                "types": set(),
                "categories": set(),
                "size": 0
            }
        node_data = G.nodes.get(node_name, {})
        communities[cid_str]["members"].append({
            "name": node_name,
            "type": node_data.get("type", ""),
            "category": node_data.get("category", ""),
            "description": node_data.get("description", "")
        })
        if node_data.get("type"):
            communities[cid_str]["types"].add(node_data["type"])
        if node_data.get("category"):
            communities[cid_str]["categories"].add(node_data["category"])
        communities[cid_str]["size"] += 1

    # Convertir les sets en listes pour JSON
    for c in communities.values():
        c["types"] = sorted(c["types"])
        c["categories"] = sorted(c["categories"])

    output = {
        "metadata": {
            "algorithm": "Louvain (python-louvain)",
            "modularity": round(modularity, 4),
            "total_communities": num_communities,
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "weighted": True
        },
        "communities": communities
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"  ✓ Exporté : {OUTPUT_JSON} ({len(communities)} communautés)")


export_communities_json()

# ============================================================
# 8. VÉRIFICATION FINALE
# ============================================================
print("\n" + "=" * 65)
print("  VÉRIFICATION FINALE")
print("=" * 65)

with driver.session() as session:
    results = session.run("""
        MATCH (c:Community)
        OPTIONAL MATCH (c)<-[:BELONGS_TO]-(e:Entity)
        RETURN c.community_id AS id,
               c.size AS size,
               c.domains AS domains,
               count(e) AS actual_members
        ORDER BY c.size DESC
        LIMIT 10
    """)
    for rec in results:
        domains = rec["domains"] or []
        domain_str = ", ".join(d for d in domains if d) or "—"
        print(f"  Communauté {rec['id']:>2} │ {rec['size']:>3} nœuds │ domaines: {domain_str}")

driver.close()

print(f"\n{'=' * 65}")
print("  ✓ PIPELINE TERMINÉ AVEC SUCCÈS")
print(f"  ✓ Fichier JSON : {OUTPUT_JSON}")
print(f"  ✓ Neo4j : communautés injectées + nœuds Community créés")
print(f"{'=' * 65}")
