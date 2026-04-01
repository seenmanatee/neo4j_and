"""
Neo4j import script (step 2 of 2)

Purpose
- Read the cached JSON produced by `neo4j_data.py`
- Create `PUBLICATION` nodes and one weighted `SIMILAR` relationship per publication pair

Current graph model
- Nodes: `(:PUBLICATION {id, title, year, authors, venue})`
- Edges: `[:SIMILAR {weight, coauthor_score, covenue_score, field_score, title_score, shared_refs_score, ...}]`

The `weight` property is an aggregate similarity score intended for weighted community detection,
including Neo4j GDS Leiden with `relationshipWeightProperty: 'weight'`.
"""

from __future__ import annotations

import json
import statistics
from typing import Dict, Set, Tuple

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Optional local graph clustering helpers ---
import networkx as nx

try:
    import community as community_louvain  # pip install python-louvain
except Exception:
    community_louvain = None

try:
    import igraph as ig
    import leidenalg as la  # pip install python-igraph leidenalg
except Exception:
    ig = None
    la = None

#EDIT
DEFAULT_FEATURE_WEIGHTS = {
    "coauthor": 0.30,
    "covenue": 0.15,
    "field": 0.20,
    "title": 0.15,
    "shared_refs": 0.20,
}


def load_pub_graph_from_neo4j(uri, user, password, db) -> nx.Graph:
    """
    Build a weighted, undirected NetworkX graph from Neo4j `SIMILAR` relationships.
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


class Neo4jImportData:
    def __init__(self, uri, user, password, db, data_path):
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            print("Connection to Neo4j database successful!")
        except ServiceUnavailable as error:
            print(f"Connection failed: {error}")
            raise

        self.db = db

        with open(data_path, "r", encoding="utf-8") as file_handle:
            self.data = json.load(file_handle)

    def close(self):
        self.driver.close()

    @staticmethod
    def _normalize_text(value: str | None) -> str:
        if not value or not isinstance(value, str):
            return ""
        return " ".join(value.strip().lower().split())

    @staticmethod
    def _normalize_field_id(raw_id):
        if not raw_id or not isinstance(raw_id, str):
            return None
        if raw_id.startswith("https://openalex.org/"):
            return raw_id.replace("https://openalex.org/", "")
        return raw_id.strip()

    @classmethod
    def _extract_research_fields(cls, work: dict) -> Set[str]:
        fields: Set[str] = set()

        topics = work.get("topics", [])
        if isinstance(topics, list):
            for topic in topics:
                if not isinstance(topic, dict):
                    continue
                field = topic.get("field") or {}
                field_id = cls._normalize_field_id(field.get("id") or field.get("display_name"))
                if field_id:
                    fields.add(field_id)

        primary_topic = work.get("primary_topic")
        if isinstance(primary_topic, dict):
            field = primary_topic.get("field") or {}
            field_id = cls._normalize_field_id(field.get("id") or field.get("display_name"))
            if field_id:
                fields.add(field_id)

        concepts = work.get("concepts", [])
        if isinstance(concepts, list):
            for concept in concepts:
                if not isinstance(concept, dict):
                    continue
                concept_id = cls._normalize_field_id(concept.get("id") or concept.get("display_name"))
                if concept_id:
                    fields.add(concept_id)

        raw_fields = work.get("fields")
        if isinstance(raw_fields, list):
            for item in raw_fields:
                if isinstance(item, dict):
                    field_id = cls._normalize_field_id(item.get("id") or item.get("display_name"))
                else:
                    field_id = cls._normalize_field_id(item)
                if field_id:
                    fields.add(field_id)

        return fields

    @classmethod
    def _publication_features(cls, work: dict) -> dict:
        authors = {
            (author.get("id"), cls._normalize_text(author.get("name")))
            for author in work.get("authors", [])
            if isinstance(author, dict) and (author.get("id") or author.get("name"))
        }
        references = {
            reference
            for reference in work.get("referenced_works", [])
            if isinstance(reference, str) and reference
        }
        return {
            "authors": authors,
            "venue": cls._normalize_text(work.get("venue")),
            "fields": cls._extract_research_fields(work),
            "references": references,
        }

    @staticmethod
    def _overlap_score(left: Set[str], right: Set[str]) -> Tuple[float, int]:
        if not left or not right:
            return 0.0, 0
        shared = left & right
        score = len(shared) / max(len(left), len(right))
        return score, len(shared)

    @staticmethod
    def _coauthor_score(left: Set[Tuple[str, str]], right: Set[Tuple[str, str]], cap: float) -> Tuple[float, int]:
        if not left or not right:
            return 0.0, 0
        shared = left & right
        shared_count = len(shared)
        if shared_count == 0:
            return 0.0, 0
        if cap <= 0:
            return 1.0, shared_count
        return min(shared_count / cap, 1.0), shared_count

    @staticmethod
    def _combine_similarity_weight(scores: Dict[str, float], feature_weights: Dict[str, float]) -> float:
        return sum(feature_weights[name] * scores.get(name, 0.0) for name in feature_weights)

    @staticmethod
    def _normalize_similarity_weight(
        scores: Dict[str, float],
        feature_weights: Dict[str, float],
        available_features: Dict[str, bool],
    ) -> Tuple[float, float]:
        """
        Normalize the aggregate score by the total weight of features that are actually available.

        This prevents sparse caches from being unfairly penalized when a whole feature family
        (for example references or research fields) is missing from the source data.
        """
        active_weight = sum(
            feature_weights[name]
            for name, is_available in available_features.items()
            if is_available and name in feature_weights
        )
        if active_weight <= 0.0:
            return 0.0, 0.0
        raw_weight = sum(
            feature_weights[name] * scores.get(name, 0.0)
            for name, is_available in available_features.items()
            if is_available and name in feature_weights
        )
        return raw_weight / active_weight, active_weight

    @staticmethod
    def _print_weight_statistics(
        candidate_weights: list[float],
        kept_weights: list[float],
        similarity_threshold: float,
    ) -> None:
        """
        Print summary statistics to help tune the `SIMILAR` threshold.
        """
        if not candidate_weights:
            print("No candidate publication pairs were scored.")
            return

        sorted_candidates = sorted(candidate_weights)

        def percentile(values: list[float], pct: float) -> float:
            if not values:
                return 0.0
            if len(values) == 1:
                return values[0]
            position = (len(values) - 1) * pct
            lower_index = int(position)
            upper_index = min(lower_index + 1, len(values) - 1)
            fraction = position - lower_index
            lower_value = values[lower_index]
            upper_value = values[upper_index]
            return lower_value + (upper_value - lower_value) * fraction

        print("SIMILAR edge weight statistics")
        print(f" Candidate pairs scored: {len(candidate_weights)}")
        print(f" Current threshold: {similarity_threshold:.3f}")
        print(f" Min: {min(sorted_candidates):.4f}")
        print(f" P25: {percentile(sorted_candidates, 0.25):.4f}")
        print(f" Median: {statistics.median(sorted_candidates):.4f}")
        print(f" Mean: {statistics.mean(sorted_candidates):.4f}")
        print(f" P75: {percentile(sorted_candidates, 0.75):.4f}")
        print(f" P90: {percentile(sorted_candidates, 0.90):.4f}")
        print(f" P95: {percentile(sorted_candidates, 0.95):.4f}")
        print(f" Max: {max(sorted_candidates):.4f}")

        for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
            count = sum(1 for weight in candidate_weights if weight >= threshold)
            print(f" Pairs with weight >= {threshold:.2f}: {count}")

        if kept_weights:
            print(f" Relationships created at current threshold: {len(kept_weights)}")
            print(f" Kept weight range: {min(kept_weights):.4f} - {max(kept_weights):.4f}")
        else:
            print(" Relationships created at current threshold: 0")

    def publication_as_nodes(self):
        """
        Create PUBLICATION nodes with properties id, title, year, authors (JSON string), venue.
        """
        for work in self.data["works_data"].values():
            try:
                pub_id = work["id"]
                title = work["title"]
                year = work["year"]
                authors = work["authors"]
                venue = work["venue"]
                author_string = json.dumps(authors)

                self.driver.execute_query(
                    """
                    CREATE (:PUBLICATION {
                        id: $pub_id,
                        title: $pub_title,
                        year: $pub_year,
                        authors: $pub_authors,
                        venue: $pub_venue
                    })
                    """,
                    pub_id=pub_id,
                    pub_title=title,
                    pub_year=year,
                    pub_authors=author_string,
                    pub_venue=venue,
                    database=self.db,
                )
            except KeyError as error:
                print(f"Missing key in work data: {error}")
            except Neo4jError as error:
                print(f"Neo4j error while inserting '{work.get('title', '<unknown>')}': {error}")
            except Exception as error:
                print(f"Unexpected error creating PUBLICATION node: {error}")

        print("All nodes were successfully added.")

    def node_count(self):
        result = self.driver.execute_query(
            """
            MATCH (n)
            RETURN count(n) AS node_count
            """,
            database=self.db,
        )
        count = result.records[0]["node_count"]
        print(f"Number of nodes: {count}")

    def edge_count(self):
        result = self.driver.execute_query(
            """
            MATCH ()-[r]->()
            RETURN count(r) AS total_relationships
            """,
            database=self.db,
        )
        count = result.records[0]["total_relationships"]
        print(f"Total relationships: {count}")

    def delete_all_nodes(self):
        self.driver.execute_query(
            """
            MATCH (n)
            DETACH DELETE n
            """,
            database=self.db,
        )
        print("All nodes were successfully deleted.")
        self.node_count()

    def cotitle_pairs_tfidf(self, min_similarity: float = 0.60, max_features: int = 10000) -> Dict[Tuple[str, str], float]:
        """
        Compute sparse TF-IDF cosine similarities for publication titles.

        Returns:
            Mapping of `(smaller_pub_id, larger_pub_id)` to cosine similarity.
        """
        works_data = self.data["works_data"]
        pub_ids = list(works_data.keys())
        titles = [works_data[pid].get("title") or "" for pid in pub_ids]

        if not pub_ids:
            return {}

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=max_features,
        )
        try:
            matrix = vectorizer.fit_transform(titles)
        except ValueError:
            return {}
        similarity_matrix = cosine_similarity(matrix, dense_output=False).tocoo()

        pairs: Dict[Tuple[str, str], float] = {}
        for row_index, col_index, similarity in zip(
            similarity_matrix.row,
            similarity_matrix.col,
            similarity_matrix.data,
        ):
            if row_index >= col_index or similarity < min_similarity:
                continue
            left_id = pub_ids[row_index]
            right_id = pub_ids[col_index]
            pairs[(left_id, right_id)] = float(similarity)
        return pairs

    def add_similarity_edges(
        self,
        similarity_threshold: float = 0.30,
        min_title_similarity: float = 0.20,
        max_features: int = 10000,
        feature_weights: Dict[str, float] | None = None,
        coauthor_cap: float = 3.0,
    ):
        """
        Build one weighted `SIMILAR` edge per publication pair.

        Each component score is normalized into `[0, 1]` before aggregation:
        - `coauthor_score`: min(shared_authors / coauthor_cap, 1.0)
        - `covenue_score`: 1.0 if venue matches else 0.0
        - `field_score`: shared field overlap
        - `title_score`: TF-IDF cosine similarity
        - `shared_refs_score`: shared references overlap

        The aggregate weight is the weighted sum of those normalized features.
        """
        works_data = self.data["works_data"]
        pub_ids = list(works_data.keys())
        feature_weights = feature_weights or DEFAULT_FEATURE_WEIGHTS

        title_pairs = self.cotitle_pairs_tfidf(
            min_similarity=min_title_similarity,
            max_features=max_features,
        )
        features_by_pub = {
            pub_id: self._publication_features(work_data)
            for pub_id, work_data in works_data.items()
        }

        created_edges = 0
        candidate_weights: list[float] = []
        kept_weights: list[float] = []
        for index, pub1 in enumerate(pub_ids):
            features1 = features_by_pub[pub1]
            for pub2 in pub_ids[index + 1:]:
                features2 = features_by_pub[pub2]
                ordered_pair = (pub1, pub2) if pub1 < pub2 else (pub2, pub1)

                coauthor_score, shared_author_count = self._coauthor_score(
                    features1["authors"],
                    features2["authors"],
                    coauthor_cap,
                )
                shared_refs_score, shared_reference_count = self._overlap_score(
                    features1["references"],
                    features2["references"],
                )
                field_score, shared_field_count = self._overlap_score(
                    features1["fields"],
                    features2["fields"],
                )
                covenue_score = 1.0 if (
                    features1["venue"] and features1["venue"] == features2["venue"]
                ) else 0.0
                title_score = title_pairs.get(ordered_pair, 0.0)

                component_scores = {
                    "coauthor": coauthor_score,
                    "covenue": covenue_score,
                    "field": field_score,
                    "title": title_score,
                    "shared_refs": shared_refs_score,
                }
                available_features = {
                    "coauthor": bool(features1["authors"]) and bool(features2["authors"]),
                    "covenue": bool(features1["venue"]) and bool(features2["venue"]),
                    "field": bool(features1["fields"]) and bool(features2["fields"]),
                    "title": bool(works_data[pub1].get("title")) and bool(works_data[pub2].get("title")),
                    "shared_refs": bool(features1["references"]) and bool(features2["references"]),
                }
                raw_weight = self._combine_similarity_weight(component_scores, feature_weights)
                weight, active_feature_weight = self._normalize_similarity_weight(
                    component_scores,
                    feature_weights,
                    available_features,
                )
                candidate_weights.append(weight)
                if weight < similarity_threshold:
                    continue

                self.driver.execute_query(
                    """
                    MATCH (p1:PUBLICATION {id: $pub1}), (p2:PUBLICATION {id: $pub2})
                    MERGE (p1)-[r:SIMILAR]->(p2)
                    SET r.weight = $weight,
                        r.raw_weight = $raw_weight,
                        r.active_feature_weight = $active_feature_weight,
                        r.coauthor_score = $coauthor_score,
                        r.covenue_score = $covenue_score,
                        r.field_score = $field_score,
                        r.title_score = $title_score,
                        r.shared_refs_score = $shared_refs_score,
                        r.shared_author_count = $shared_author_count,
                        r.shared_reference_count = $shared_reference_count,
                        r.shared_field_count = $shared_field_count,
                        r.same_venue = $same_venue
                    """,
                    pub1=ordered_pair[0],
                    pub2=ordered_pair[1],
                    weight=round(float(weight), 6),
                    raw_weight=round(float(raw_weight), 6),
                    active_feature_weight=round(float(active_feature_weight), 6),
                    coauthor_score=round(float(coauthor_score), 6),
                    covenue_score=round(float(covenue_score), 6),
                    field_score=round(float(field_score), 6),
                    title_score=round(float(title_score), 6),
                    shared_refs_score=round(float(shared_refs_score), 6),
                    shared_author_count=shared_author_count,
                    shared_reference_count=shared_reference_count,
                    shared_field_count=shared_field_count,
                    same_venue=bool(covenue_score),
                    database=self.db,
                )
                created_edges += 1
                kept_weights.append(weight)

        print(
            "Created "
            f"{created_edges} SIMILAR relationships "
            f"(threshold={similarity_threshold}, title_min={min_title_similarity})."
        )
        self._print_weight_statistics(candidate_weights, kept_weights, similarity_threshold)


def main(
    URI,
    USER,
    PASSWORD,
    DB,
    PATH,
    similarity_threshold: float = 0.30,
    min_title_similarity: float = 0.20,
):
    """
    Create a Neo4j graph with publication nodes and aggregated weighted similarity edges.
    """
    importer = Neo4jImportData(URI, USER, PASSWORD, DB, PATH)
    try:
        importer.delete_all_nodes()
        importer.publication_as_nodes()
        importer.add_similarity_edges(
            similarity_threshold=similarity_threshold,
            min_title_similarity=min_title_similarity,
        )
        importer.node_count()
        importer.edge_count()
    finally:
        importer.close()


if __name__ == "__main__":
    URI = "neo4j://127.0.0.1:7687"
    USER = "neo4j"
    PASSWORD = "JohnGoat1000"
    DB = "neo4j"
    PATH = "/Users/seanmaniti/name_disam_exp/neo4j_and/cache/rajakumar_balla_data.json"  # FILE EDIT HERE
    SIMILARITY_THRESHOLD = 0.1  #  EDIT HERE
    MIN_TITLE_SIMILARITY = 0.1  #  EDIT HERE

    main(
        URI,
        USER,
        PASSWORD,
        DB,
        PATH,
        similarity_threshold=SIMILARITY_THRESHOLD,
        min_title_similarity=MIN_TITLE_SIMILARITY,
    )
