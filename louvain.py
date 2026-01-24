from neo4j import GraphDatabase
import networkx as nx
from community import community_louvain

URI = "neo4j://127.0.0.1:7687"  # Example local instance
USER = "neo4j"
PASSWORD = "JohnGoat1000"
DB = "neo4j"
PATH = "/Users/seanmaniti/neo4j/neo4j_and/cache/David Nathan_data.json"

def load_pub_graph_from_neo4j(uri, user, password, db):
    """
    Build an undirected NetworkX graph from Neo4j by combining:
      - COAUTHOR.weight
      - COVENUE.weight (default 1 if missing)
      - COTITLE.similarity
    into a single edge weight = sum of available weights.
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    G = nx.Graph()

    # Coauthor relationship is weighted by # of shared co-authors, cotitle is weighted by similarity score
    q = """
    MATCH (p1:PUBLICATION)-[r:COAUTHOR|COVENUE|COTITLE]-(p2:PUBLICATION)
    WITH p1.id AS a, p2.id AS b, type(r) AS t, r
    WITH a,b,
      CASE t
        WHEN 'COAUTHOR' THEN coalesce(r.weight, 1.0)
        WHEN 'COVENUE'  THEN coalesce(r.weight, 1.0)
        WHEN 'COTITLE'  THEN coalesce(r.similarity, 0.0)
      END AS w
    RETURN a,b,w
    """
    with driver.session(database=db) as session:
        for rec in session.run(q):
            a, b, w = rec["a"], rec["b"], float(rec["w"])
            if a == b:
                continue
            # undirected merge with summed weight
            if G.has_edge(a, b):
                G[a][b]["weight"] += w
            else:
                G.add_edge(a, b, weight=w)
    driver.close()
    return G

def run_louvain_and_write(uri, user, password, db, resolution=1.0, seed=42):
    G = load_pub_graph_from_neo4j(uri, user, password, db)
    if G.number_of_nodes() == 0:
        print("Graph is empty; nothing to cluster.")
        return

    partition = community_louvain.best_partition(
        G, weight="weight", resolution=resolution, random_state=seed
    )
    Q = community_louvain.modularity(partition, G, weight="weight")
    print(f"Louvain modularity: {Q:.4f} | nodes: {G.number_of_nodes()} | edges: {G.number_of_edges()}")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    # Batch updates for speed
    updates = {}
    for node_id, comm in partition.items():
        updates.setdefault(comm, []).append(node_id)

    with driver.session(database=db) as session:
        for comm, ids in updates.items():
            # write community for a batch of ids
            session.run(
                """
                UNWIND $ids AS pid
                MATCH (p:PUBLICATION {id: pid})
                SET p.community = $comm
                """,
                ids=ids, comm=int(comm)
            )
    driver.close()

    # Optional: print sizes
    from collections import Counter
    sizes = Counter(partition.values())
    for comm, sz in sizes.most_common():
        print(f"Community {comm}: {sz} nodes")

if __name__ == "__main__":
    run_louvain_and_write(URI, USER, PASSWORD, DB, resolution=1.0)