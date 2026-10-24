import os
import re
import unicodedata
from neo4j import GraphDatabase

# Essayer d'importer les credentials depuis backend.py
try:
    from backend import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
except ImportError:
    NEO4J_URI = "neo4j+s://acfc5ab2.databases.neo4j.io"
    NEO4J_USER = "acfc5ab2"
    NEO4J_PASSWORD = "XVH4uQXBtkf_FSMrK4T16vcTbJZ8QQu_7-HblC-hJgQ"

CHUNKS_FILE = "chunks_export.txt"

# Liste d'entités canoniques
ENTITIES = [
    "Piton des Neiges", "Caldera", "Syenite", "Cilaos", "Mafate", "Salazie",
    "Basalte", "Rift-zone N030", "Breche Orangers", "Phonolite", "Source Thermale", "Gabbro"
]

# Dictionnaire de variations -> Nom canonique
SYNONYMS = {
    "rift zone n030": "Rift-zone N030",
    "rift n030": "Rift-zone N030",
    "zone n030": "Rift-zone N030",
    "n030": "Rift-zone N030",
    "n030e": "Rift-zone N030",
    "caldeira": "Caldera",
    "caldera": "Caldera"
}

# Règles d'extraction de relations (Mots-clés -> Relation Neo4j)
RELATION_RULES = {
    "LOCALISE_LE_LONG_DE": ["le long de", "situé sur", "le long du", "proximité", "dans"],
    "CONTROLE_STRUCTURAL": ["contrôle structural", "contrôle", "influence", "dirige", "structuration"],
    "POSTERIEUR_A": ["postérieur", "après", "coupe", "recoupe", "plus récent"],
    "COMPOSE_DE": ["composé", "constitué", "formé", "contient"]
}

def normalize_string(s: str) -> str:
    s = unicodedata.normalize('NFD', s).encode('ascii', 'ignore').decode('utf-8').lower()
    return s.replace("°e", "e").replace("°", "")

def find_entities_in_text(text: str) -> list:
    found = set()
    norm_text = normalize_string(text)
    
    # 1. Chercher les synonymes
    for syn, canonical in SYNONYMS.items():
        if syn in norm_text:
            found.add(canonical)
            
    # 2. Chercher les entités exactes
    for e in ENTITIES:
        if normalize_string(e) in norm_text:
            found.add(e)
            
    return list(found)

def extract_relations_from_chunk(text: str):
    """
    Retourne une liste de tuples (source, relation, target) détectés dans le texte.
    Utilise une approche par phrase (sentence-level) pour garantir la proximité
    entre les entités et le mot-clé de relation.
    """
    relations_found = set()
    
    # Découper le chunk en phrases
    sentences = re.split(r'(?<=[.!?]) +', text)
    
    for sentence in sentences:
        if len(sentence) < 10:
            continue
            
        entities = find_entities_in_text(sentence)
        if len(entities) >= 2:
            norm_sentence = normalize_string(sentence)
            
            # Comparer toutes les paires possibles dans la même phrase
            for i in range(len(entities)):
                for j in range(i+1, len(entities)):
                    e1 = entities[i]
                    e2 = entities[j]
                    
                    # Chercher le type de relation dans CETTE phrase
                    for rel_type, keywords in RELATION_RULES.items():
                        for kw in keywords:
                            if kw in norm_sentence:
                                relations_found.add((e1, rel_type, e2))
                                break # On arrête pour ce type de relation pour cette paire
                                
    return list(relations_found)

def main():
    print("[1] Chargement des textes...")
    if not os.path.exists(CHUNKS_FILE):
        print(f"Erreur: {CHUNKS_FILE} introuvable.")
        return
        
    with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = f.read().split("\n\n--- CHUNK")
        
    print(f"  > {len(chunks)} chunks chargés.")
    
    print("\n[2] Extraction des relations sémantiques...")
    all_relations = set()
    for chunk in chunks:
        rels = extract_relations_from_chunk(chunk)
        for r in rels:
            all_relations.add(r)
            
    print(f"  > {len(all_relations)} relations uniques détectées.")
    for r in all_relations:
        print(f"      {r[0]} -[{r[1]}]-> {r[2]}")
        
    if not all_relations:
        print("Aucune relation détectée. Fin.")
        return
        
    print("\n[3] Connexion à Neo4j Aura...")
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
    except Exception as e:
        print(f"Erreur connexion Neo4j: {e}")
        return
        
    print("\n[4] Injection dans Neo4j (MERGE)...")
    success_count = 0
    with driver.session() as s:
        for source, rel_type, target in all_relations:
            # Cypher robuste pour lier les noeuds s'ils existent et s'ils matchent
            cypher = f"""
            MATCH (c:Entity), (r:Entity)
            WHERE toLower(c.name) CONTAINS toLower($source) AND toLower(r.name) CONTAINS toLower($target)
            MERGE (c)-[:{rel_type}]->(r)
            """
            try:
                s.run(cypher, source=source, target=target)
                success_count += 1
            except Exception as e:
                print(f"Erreur sur {source}-{rel_type}-{target}: {e}")
                
    driver.close()
    print(f"\n[5] Terminé ! {success_count} nouvelles requêtes de relations exécutées dans Neo4j.")

if __name__ == "__main__":
    main()
