from neo4j import GraphDatabase

NEO4J_URI = "neo4j+s://acfc5ab2.databases.neo4j.io"
NEO4J_USER = "acfc5ab2"
NEO4J_PASSWORD = "XVH4uQXBtkf_FSMrK4T16vcTbJZ8QQu_7-HblC-hJgQ"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# On va supprimer uniquement les relations qu'on a ajoutées récemment pour ne pas casser le reste
relations_to_delete = ["LOCALISE_LE_LONG_DE", "CONTROLE_STRUCTURAL", "POSTERIEUR_A", "COMPOSE_DE"]

with driver.session() as s:
    for rel in relations_to_delete:
        cypher = f"MATCH ()-[r:{rel}]->() DELETE r"
        try:
            r = s.run(cypher)
            summary = r.consume()
            print(f"Purge de {rel} : {summary.counters.relationships_deleted} relations supprimées.")
        except Exception as e:
            print(f"Erreur purge {rel}: {e}")

print("Purge terminée.")
