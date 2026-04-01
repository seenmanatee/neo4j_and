from __future__ import annotations

import argparse
import json
from pathlib import Path

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
import networkx as nx

try:
    import community as community_louvain  # pip install python-louvain
except Exception:
    community_louvain = None

try:
    import igraph as ig
    import leidenalg as la
except Exception:
    ig = None
    la = None


def load_pub_graph_from_neo4j(uri, user, password, db) -> nx.Graph:
    """
    Build a weighted, undirected NetworkX graph from `SIMILAR.weight`.
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    query = """
    MATCH (p1:PUBLICATION)-[r:SIMILAR]-(p2:PUBLICATION)
    RETURN p1.id AS a, p2.id AS b, coalesce(toFloat(r.weight), 0.0) AS w
    """

    graph = nx.Graph()
    with driver.session(database=db) as session:
        for record in session.run(query):
            a, b, weight = record["a"], record["b"], float(record["w"])
            if not a or not b or a == b or weight <= 0.0:
                continue
            if graph.has_edge(a, b):
                graph[a][b]["weight"] += weight
            else:
                graph.add_edge(a, b, weight=weight)
    driver.close()
    return graph


def run_louvain(graph: nx.Graph, resolution: float = 1.0, seed: int = 42):
    if community_louvain is None:
        raise RuntimeError("python-louvain not installed. `pip install python-louvain`")
    partition = community_louvain.best_partition(
        graph,
        weight="weight",
        resolution=resolution,
        random_state=seed,
    )
    modularity = community_louvain.modularity(partition, graph, weight="weight")
    return partition, modularity


def run_leiden(graph: nx.Graph, resolution: float = 1.0, seed: int = 42):
    if ig is None or la is None:
        raise RuntimeError("Leiden not installed. `pip install python-igraph leidenalg`")

    nodes = list(graph.nodes())
    index_of = {node_id: index for index, node_id in enumerate(nodes)}
    edges = [(index_of[u], index_of[v]) for u, v in graph.edges()]
    weights = [float(graph[u][v].get("weight", 1.0)) for u, v in graph.edges()]

    ig_graph = ig.Graph(n=len(nodes), edges=edges, directed=False)
    ig_graph.es["weight"] = weights

    partition = la.find_partition(
        ig_graph,
        la.RBConfigurationVertexPartition,
        weights=ig_graph.es["weight"],
        resolution_parameter=resolution,
        seed=seed,
    )
    membership = partition.membership
    node_partition = {nodes[i]: int(membership[i]) for i in range(len(nodes))}
    return node_partition, partition.quality()


def _drop_graph_if_exists(driver, db: str, graph_name: str) -> None:
    with driver.session(database=db) as session:
        try:
            session.run(
                "CALL gds.graph.drop($graph_name, false) YIELD graphName RETURN graphName",
                graph_name=graph_name,
            ).consume()
        except Neo4jError as error:
            message = str(error)
            if (
                "procedure.procedurenotfound" in message.lower()
                or "no such procedure" in message.lower()
                or "not found" in message.lower()
                or "does not exist" in message.lower()
            ):
                return
            raise


def _gds_available(driver, db: str) -> bool:
    with driver.session(database=db) as session:
        try:
            record = session.run(
                """
                SHOW PROCEDURES
                YIELD name
                WHERE name IN ['gds.graph.project', 'gds.leiden.write']
                RETURN count(*) AS procedure_count
                """
            ).single()
            return bool(record and int(record["procedure_count"]) >= 2)
        except Neo4jError:
            return False


def run_gds_leiden(
    uri: str,
    user: str,
    password: str,
    db: str,
    graph_name: str,
    write_property: str,
    seed: int,
):
    """
    Project `SIMILAR` as an undirected weighted graph and run GDS Leiden.
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        if not _gds_available(driver, db):
            raise RuntimeError(
                "Neo4j GDS procedures are not available on this database instance. "
                "Falling back requires local Leiden (`python-igraph` + `leidenalg`) instead."
            )
        _drop_graph_if_exists(driver, db, graph_name)

        project_query = """
        CALL gds.graph.project(
            $graph_name,
            'PUBLICATION',
            {
                SIMILAR: {
                    orientation: 'UNDIRECTED',
                    properties: 'weight'
                }
            }
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """
        leiden_query = """
        CALL gds.leiden.write(
            $graph_name,
            {
                writeProperty: $write_property,
                relationshipWeightProperty: 'weight',
                randomSeed: $seed
            }
        )
        YIELD communityCount, modularity, modularities, ranLevels, nodePropertiesWritten
        RETURN communityCount, modularity, modularities, ranLevels, nodePropertiesWritten
        """
        predictions_query = """
        MATCH (p:PUBLICATION)
        WHERE p[$write_property] IS NOT NULL
        RETURN p.id AS node_id, toString(p[$write_property]) AS cluster_id
        ORDER BY node_id
        """

        with driver.session(database=db) as session:
            projection_stats = session.run(project_query, graph_name=graph_name).single()
            leiden_stats = session.run(
                leiden_query,
                graph_name=graph_name,
                write_property=write_property,
                seed=seed,
            ).single()
            predictions = {
                record["node_id"]: record["cluster_id"]
                for record in session.run(predictions_query, write_property=write_property)
            }
        
        return predictions, {
            "graphName": projection_stats["graphName"],
            "nodeCount": projection_stats["nodeCount"],
            "relationshipCount": projection_stats["relationshipCount"],
            "communityCount": leiden_stats["communityCount"],
            "modularity": leiden_stats["modularity"],
            "modularities": leiden_stats["modularities"],
            "ranLevels": leiden_stats["ranLevels"],
            "nodePropertiesWritten": leiden_stats["nodePropertiesWritten"],
        }
    except Neo4jError as error:
        raise RuntimeError(
            "GDS Leiden failed. Confirm the Neo4j Graph Data Science plugin is installed "
            "and that the `SIMILAR.weight` relationships exist."
        ) from error
    finally:
        try:
            _drop_graph_if_exists(driver, db, graph_name)
        finally:
            driver.close()


def write_predictions(partition: dict, output_path: Path) -> None:
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
    parser.add_argument(
        "--method",
        default="leiden",
        choices=["gds-leiden", "leiden", "louvain"],
        help="Clustering method",
    )
    parser.add_argument("--resolution", type=float, default=1.0, help="Resolution parameter for local algorithms")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="pred_clusters.jsonl", help="Output JSONL path")
    parser.add_argument("--graph-name", default="pubGraph", help="Temporary GDS in-memory graph name")
    parser.add_argument("--write-property", default="communityId", help="Node property written by GDS Leiden")
    args = parser.parse_args()

    if args.method == "gds-leiden":
        try:
            partition, stats = run_gds_leiden(
                args.uri,
                args.user,
                args.password,
                args.db,
                args.graph_name,
                args.write_property,
                args.seed,
            )
            print(f"GDS Leiden communities: {stats['communityCount']}")
            print(f"GDS Leiden modularity: {stats['modularity']:.4f}")
            print(
                f"Projected graph: {stats['nodeCount']} nodes, "
                f"{stats['relationshipCount']} undirected relationships"
            )
        except RuntimeError as error:
            print(str(error))
            print("Falling back to local Leiden over `SIMILAR.weight`.")
            graph = load_pub_graph_from_neo4j(args.uri, args.user, args.password, args.db)
            partition, quality = run_leiden(graph, resolution=args.resolution, seed=args.seed)
            print(f"Leiden partition size: {len(set(partition.values()))}")
            print(f"Leiden quality: {quality:.4f}")
    else:
        graph = load_pub_graph_from_neo4j(args.uri, args.user, args.password, args.db)
        if args.method == "louvain":
            partition, modularity = run_louvain(graph, resolution=args.resolution, seed=args.seed)
            print(f"Louvain partition size: {len(set(partition.values()))}")
            print(f"Louvain modularity: {modularity:.4f}")
        else:
            partition, quality = run_leiden(graph, resolution=args.resolution, seed=args.seed)
            print(f"Leiden partition size: {len(set(partition.values()))}")
            print(f"Leiden quality: {quality:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(partition, output_path)
    print(f"Wrote predictions to {output_path}")


if __name__ == "__main__":
    main()
