from __future__ import annotations

import argparse
import json
from pathlib import Path

from neo4j import GraphDatabase
import networkx as nx

# Louvain (python-louvain)
try:
    import community as community_louvain  # pip install python-louvain
except Exception:
    community_louvain = None

# Leiden (igraph + leidenalg)
try:
    import igraph as ig
    import leidenalg as la
except Exception:
    ig = None
    la = None


def load_pub_graph_from_neo4j(uri, user, password, db,
                              coauthor_scale: float = 1.0,
                              covenue_scale: float = 1.0,
                              cotitle_scale: float = 1.2,
                              use_log_coauthor: bool = True) -> nx.Graph:
    """
    Build a weighted, undirected NetworkX graph from Neo4j PUBLICATION relationships.

    Weights:
      - COAUTHOR: coauthor_scale * (log(1 + weight) if use_log_coauthor else weight; defaults to 1.0 if missing)
      - COVENUE : covenue_scale * 1.0 (if r.weight is null or 0) else r.weight
      - COTITLE : cotitle_scale * coalesce(r.similarity, 0.0)
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))

    q = """
    MATCH (p1:PUBLICATION)-[r:COAUTHOR|COVENUE|COTITLE]-(p2:PUBLICATION)
    WITH p1.id AS a, p2.id AS b, type(r) AS t, r
    WITH a, b, t, r,
         CASE
           WHEN t = 'COAUTHOR' THEN $coauthorScale *
                CASE
                  WHEN r.weight IS NULL THEN 1.0
                  ELSE (CASE WHEN $useLog THEN log(1 + toFloat(r.weight))
                             ELSE toFloat(r.weight) END)
                END
           WHEN t = 'COVENUE'  THEN $covenueScale *
                CASE
                  WHEN r.weight IS NULL OR toFloat(r.weight) = 0 THEN 1.0
                  ELSE toFloat(r.weight)
                END
           WHEN t = 'COTITLE'  THEN $cotitleScale * coalesce(toFloat(r.similarity), 0.0)
           ELSE 0.0
         END AS w
    RETURN a, b, w
    """

    G = nx.Graph()
    with driver.session(database=db) as session:
        for rec in session.run(q,
                               coauthorScale=coauthor_scale,
                               covenueScale=covenue_scale,
                               cotitleScale=cotitle_scale,
                               useLog=use_log_coauthor):
            a, b, w = rec["a"], rec["b"], float(rec["w"])
            if not a or not b or a == b or w <= 0.0:
                continue
            if G.has_edge(a, b):
                G[a][b]["weight"] += w
            else:
                G.add_edge(a, b, weight=w)
    driver.close()
    return G


def run_louvain(G: nx.Graph, resolution: float = 1.0, seed: int = 42):
    """
    Run Louvain on a NetworkX graph.
    Returns:
      (partition_dict, modularity)
    """
    if community_louvain is None:
        raise RuntimeError("python-louvain not installed. `pip install python-louvain`")
    part = community_louvain.best_partition(G, weight="weight",
                                            resolution=resolution,
                                            random_state=seed)
    Q = community_louvain.modularity(part, G, weight="weight")
    return part, Q


def run_leiden(G: nx.Graph, resolution: float = 1.0, seed: int = 42):
    """
    Run Leiden (CPM objective) on a NetworkX graph.
    Returns:
      (partition_dict, quality)
    """
    if ig is None or la is None:
        raise RuntimeError("Leiden not installed. `pip install python-igraph leidenalg`")

    # Map NetworkX nodes -> indices
    nodes = list(G.nodes())
    index_of = {n: i for i, n in enumerate(nodes)}

    # Build igraph from NetworkX
    edges = [(index_of[u], index_of[v]) for u, v in G.edges()]
    weights = [float(G[u][v].get("weight", 1.0)) for u, v in G.edges()] #HEREE
    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    g.es["weight"] = weights

    # Leiden with Constant Potts Model (CPM) resolution parameter
    part = la.find_partition(
        g,
        la.CPMVertexPartition,
        weights=g.es["weight"],
        resolution_parameter=resolution,
        seed=seed
    )
    membership = part.membership
    partition = {nodes[i]: int(membership[i]) for i in range(len(nodes))}
    quality = part.quality()
    return partition, quality


def write_predictions(partition: dict, output_path: Path) -> None:
    # Emit mention_id/pred_cluster_id JSONL for evaluate_b3.py.
    with output_path.open("w", encoding="utf-8") as handle:
        for node_id, cluster_id in partition.items():
            handle.write(
                json.dumps(
                    {"mention_id": node_id, "pred_cluster_id": str(cluster_id)},
                    ensure_ascii=True,
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run community detection on the Neo4j graph.")
    parser.add_argument("--uri", default="neo4j://127.0.0.1:7687", help="Neo4j URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j username")
    parser.add_argument("--password", default="JohnGoat1000", help="Neo4j password")
    parser.add_argument("--db", default="neo4j", help="Neo4j database name")
    parser.add_argument("--method", default="leiden", choices=["leiden", "louvain"], help="Clustering method")
    parser.add_argument("--resolution", type=float, default=1.0, help="Resolution parameter")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="pred_clusters.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    G = load_pub_graph_from_neo4j(
        args.uri,
        args.user,
        args.password,
        args.db,
    )

    if args.method == "louvain":
        partition, modularity = run_louvain(G, resolution=args.resolution, seed=args.seed)
        print(f"Louvain partition size: {len(set(partition.values()))}")
        print(f"Louvain modularity: {modularity:.4f}")
    else:
        partition, quality = run_leiden(G, resolution=args.resolution, seed=args.seed)
        print(f"Leiden partition size: {len(set(partition.values()))}")
        print(f"Leiden quality: {quality:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(partition, output_path)
    print(f"Wrote predictions to {output_path}")


if __name__ == "__main__":
    main()
