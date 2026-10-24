from neo4j import GraphDatabase

NEO4J_URI = "neo4j+s://acfc5ab2.databases.neo4j.io"
NEO4J_USER = "acfc5ab2"
NEO4J_PASSWORD = "XVH4uQXBtkf_FSMrK4T16vcTbJZ8QQu_7-HblC-hJgQ"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

with driver.session() as s:
    r = s.run("MATCH (n:Entity)-[rel]->(m:Entity) WHERE type(rel) <> 'CO_OCCURS' RETURN n.name, type(rel), m.name LIMIT 20")
    print("All non-co-occur relations:")
    for rec in r:
        print(rec)

    r2 = s.run("MATCH (n:Entity) WHERE toLower(n.name) CONTAINS 'caldera' RETURN n.name")
    print("Caldera nodes:")
    for rec in r2:
        print(rec)

    r3 = s.run("MATCH (n:Entity) WHERE toLower(n.name) CONTAINS 'rift' RETURN n.name")
    print("Rift nodes:")
    for rec in r3:
        print(rec)
