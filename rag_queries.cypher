
// ===========================================================
// REQUETES GRAPH RAG — A COPIER DANS TON PIPELINE PYTHON
// ===========================================================

// --- RAG Query A : Expansion depuis mots-clés ---
// Usage: donner des seed_names issus de l'embedding search
UNWIND $seed_names AS seedName
MATCH (seed:Entity) WHERE toLower(seed.name) CONTAINS toLower(seedName)
WITH collect(DISTINCT seed.community) AS relevantCommunities
MATCH (e:Entity) WHERE e.community IN relevantCommunities
OPTIONAL MATCH (e)<-[:MENTIONS]-(c:Chunk)
RETURN e.name AS entite, e.type AS type, e.community AS comm,
       collect(DISTINCT c.text)[..3] AS contexte_chunks
ORDER BY e.pagerank DESC LIMIT 50

// --- RAG Query B : Chemin entre deux entités ---
MATCH path = shortestPath(
  (a:Entity {name: $entity_a})-[*..5]-(b:Entity {name: $entity_b})
)
RETURN path

// --- RAG Query C : Contexte d'une communauté ---
MATCH (e:Entity {community: $community_id})
OPTIONAL MATCH (e)<-[:MENTIONS]-(c:Chunk)
RETURN e.name, e.type, collect(DISTINCT c.text)[..2] AS passages
ORDER BY e.pagerank DESC

// --- RAG Query D : Sous-graphe hydrothermal ---
MATCH (root:Entity {name: "Système hydrothermal du Piton des Neiges"})
MATCH (root)-[r*..3]-(related:Entity)
RETURN root, r, related LIMIT 50

// --- RAG Query E : Timeline géologique ---
MATCH p = (a:GeologicalStage)-[:PRECEDES*]->(z:GeologicalStage)
RETURN p ORDER BY a.age_range
