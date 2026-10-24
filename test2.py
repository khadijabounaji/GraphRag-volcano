from neo4j import GraphDatabase

NEO4J_URI = "neo4j+s://acfc5ab2.databases.neo4j.io"
NEO4J_USER = "acfc5ab2"
NEO4J_PASSWORD = "XVH4uQXBtkf_FSMrK4T16vcTbJZ8QQu_7-HblC-hJgQ"

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

cypher = """
MATCH (c:Entity)-[r]->(x:Entity)
WHERE ANY(e IN $entities WHERE toLower(c.name) CONTAINS e)
AND ANY(e IN $entities WHERE toLower(x.name) CONTAINS e)
AND type(r) <> 'CO_OCCURS'
RETURN c.name AS source, type(r) AS relation, x.name AS target
LIMIT 30
"""

entities = ["caldera", "rift-zone n030"]

with driver.session() as s:
    r = s.run(cypher, entities=entities)
    results = [dict(rec) for rec in r]
    
print("Entities list:", entities)
print("Graph Results:", results)
