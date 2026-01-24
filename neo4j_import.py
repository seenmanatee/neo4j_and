"""
Neo4j import script (step 2 of 2)

Purpose
- Read the cached JSON produced by `neo4j_data.py`
- Create publication nodes and similarity edges in Neo4j for disambiguation experiments

Current capabilities
- PUBLICATION nodes with properties: id, title, year, authors (JSON string), venue
- COAUTHOR edges: connect publications that share at least one author
- COVENUE edges: connect publications that share the same venue
- COTITLE edges: connect publications with similar titles
- SHARED_REFERENCES edges: connect publications with overlapping references
- COAUTHOR_OVERLAP edges: connect publications with high coauthor overlap
- RESEARCH_FIELD edges: connect publications sharing research fields

Usage
1) Ensure a Neo4j instance is running and accessible
2) Set connection variables in the main block (URI, USER, PASSWORD, DB, PATH)
3) Run from repo root, e.g. `PYTHONPATH=. python neo4j_import.py`

Outputs (example counts from "David Nathan")
    Connection to Neo4j database successful!
    All nodes were successfully deleted.
    Number of nodes: 0
    All nodes were successfully added.
    Created 5122 CoVenue relationships
    Created 17853 CoAuthor relationships
    Created 85 cotitle relationships (cosine ≥ 0.6)
    Number of nodes: 428
    Total relationships: 23060

Notes
- Relationships are currently modeled as directional in code, but clustering can treat the graph as undirected
- Consider converting to undirected by creating a single relationship with `MERGE` or by normalizing during analysis
"""

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, Neo4jError
import json
from typing import List, Dict, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Graph clustering helpers (Louvain / Leiden) ---
import networkx as nx

# Louvain (python-louvain)
try:
    import community as community_louvain  # pip install python-louvain
except Exception:
    community_louvain = None

# Leiden (igraph + leidenalg)
try:
    import igraph as ig
    import leidenalg as la  # pip install python-igraph leidenalg
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
      - COAUTHOR: coauthor_scale * (log(1 + weight) if use_log_coauthor else weight or 1.0 if missing)
      - COVENUE : covenue_scale * 1.0 (if r.weight is null or 0) else r.weight
      - COTITLE : cotitle_scale * coalesce(r.similarity, 0.0)

    Returns:
      networkx.Graph with 'weight' on each edge.
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
    with driver.session(database=db) as sess:
        for rec in sess.run(q,
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
      - partition_dict: {node_id: community_id}
      - modularity: Louvain modularity score
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
      - partition_dict: {node_id: community_id}
      - quality: CPM objective value (not modularity)
    """
    if ig is None or la is None:
        raise RuntimeError("Leiden not installed. `pip install python-igraph leidenalg`")

    # Map NetworkX nodes -> indices
    nodes = list(G.nodes())
    index_of = {n: i for i, n in enumerate(nodes)}

    # Build igraph from NetworkX
    edges = [(index_of[u], index_of[v]) for u, v in G.edges()]
    weights = [float(G[u][v].get("weight", 1.0)) for u, v in G.edges()]
    g = ig.Graph(n=len(nodes), edges=edges, directed=False)
    g.es["weight"] = weights

    # Leiden with Constant Potts Model (CPM), resolution_parameter controls granularity
    rng = la.RNG(seed)
    part = la.find_partition(
        g,
        la.CPMVertexPartition,
        weights=g.es["weight"],
        resolution_parameter=resolution,
        seed=rng
    )
    membership = part.membership
    partition = {nodes[i]: int(membership[i]) for i in range(len(nodes))}
    quality = part.quality()
    return partition, quality


class Neo4jImportData:
    def __init__(self, uri, user, password, db, data_path):
        """
        Initialize a Neo4j driver and load the cached data

        Args
            uri (str): Neo4j URI (e.g. bolt://localhost:7687)
            user (str): instance username
            password (str): instance password
            db (str): database name
            data_path (str): Path to JSON created by neo4j_data.py (cache/<Author>_data.json)
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user,password))
            self.driver.verify_connectivity()
            print("Connection to Neo4j database successful!")
        except ServiceUnavailable as e:
            print(f"Connection failed: {e}")

        self.db = db

        with open(data_path,'r', encoding='utf-8') as file:
            self.data = json.load(file)

    def close(self):
        self.driver.close()

    
    def publication_as_nodes(self):
        """
        Create PUBLICATION nodes with properties id, title, year, authors (JSON string), venue
        """
        for work in self.data['works_data']:
            pub_id, title, year, authors, venue = (
                self.data['works_data'][work][k] for k in ['id', 'title', 'year', 'authors', 'venue']
            )

            # Must convert authors data to json string
            author_string = json.dumps(authors)

            try:
                summary = self.driver.execute_query("""
                    CREATE (n:PUBLICATION {id: $pub_id, title: $pub_title, year: $pub_year, authors: $pub_authors, venue: $pub_venue})
                    """,
                    pub_id = pub_id, 
                    pub_title = title, 
                    pub_year = year, 
                    pub_authors = author_string, 
                    pub_venue = venue,
                    database = self.db,
                )

            except KeyError as ke:
                print(f"Missing key in work data: {ke}")
            except Neo4jError as ne:
                print(f"Neo4j error while inserting '{title}': {ne}")
            except Exception as e:  # Keep broad catch to continue bulk ingestion
                print(f"Unexpected error creating PUBLICATION node: {e}")

        print("All nodes were successfully added.")


    def node_count(self):
        """
        Print total node count
        """
        result = self.driver.execute_query("""
            MATCH (n) RETURN count(n) AS node_count
        """,
        database = self.db)
        count = result.records[0]["node_count"]
        print(f"Number of nodes: {count}")

    def edge_count(self):
        """
        Print total relationship count
        """
        result = self.driver.execute_query(
            """
            MATCH ()-[r]->() RETURN COUNT(r) AS totalRelationships
            """,
            database=self.db
        )
        count = result.records[0]["totalRelationships"]
        print(f"Total relationships: {count}")


    def delete_all_nodes(self):
        """
        Delete all nodes and relationships from the selected database
        """
        self.driver.execute_query("""
            MATCH (n) DETACH DELETE n
        """,
        database = self.db)
        print("All nodes were successfully deleted.")
        self.node_count()


    def add_covenue_edge(self):
        """
        Create COVENUE edges between publications that share the same venue
        Currently directional; community detection can treat as undirected
        """
        id_venue = {}
        for work_data in self.data['works_data'].values():
            pub_id = work_data['id']
            venue = work_data['venue']
            id_venue[pub_id] = venue

        created_edges = 0

        pub_keys = list(id_venue.keys())
        for i, pub1 in enumerate(pub_keys):
            for pub2 in pub_keys[i+1:]:
                if id_venue[pub1] == id_venue[pub2]:
                    #print(f"Trying edge between {pub1} and {pub2}, venue: {id_venue[pub1]}")
                    self.driver.execute_query("""
                        MATCH (p1:PUBLICATION {id: $pub_name1}), (p2:PUBLICATION {id: $pub_name2})
                        CREATE (p1) - [:COVENUE {venue: $pub_venue, weight: 1.0}] -> (p2)
                    """,
                    pub_name1 = pub1, 
                    pub_name2 = pub2, 
                    pub_venue = id_venue[pub1],
                    database = self.db
                    )
                    created_edges += 1
        
        print(f" Created {created_edges} CoVenue relationships")


    def add_coauthor_edge(self):
        """
        Create COAUTHOR edges between publications that share at least one author
        Adds a `weight` equal to the number of shared authors
        Currently directional; community detection can treat as undirected
        """

        works_data = self.data['works_data']

        #print(works_data)
        created_edges = 0

        pub_keys = list(works_data.keys())
        # Creates a set of pub1 authors
        for i, pub1 in enumerate(pub_keys):
            authors1 = works_data[pub1]['authors']
            set1 = {(a['id'], a['name']) for a in authors1}

            # Set of pub2 authors
            for pub2 in pub_keys[i+1:]:
                authors2 = works_data[pub2]['authors']
                set2 = {(a['id'], a['name']) for a in authors2}

                # Shared authors between two publications, including ambiguous name
                shared_authors = set1 & set2

                #### FUTURE: may need to create metric for number of shared authors for weighted edge
                if shared_authors:
                    shared_authors_json = [{"id": aid, "name": name} for (aid, name) in shared_authors]
                    json_string = json.dumps(shared_authors_json)

                    # Add weight for COAUTHOR (# of shared authors)
                    weight = len(shared_authors)

                    self.driver.execute_query(
                        """
                        MATCH (p1:PUBLICATION {id: $pub_name1}), (p2:PUBLICATION {id: $pub_name2})
                        CREATE (p1)-[:COAUTHOR {coauthor: $pub_coauthor, weight: $weight}]->(p2)
                        """,
                        pub_name1=pub1,
                        pub_name2=pub2,
                        pub_coauthor=json_string,
                        weight = weight,
                        database=self.db
                    )
                    created_edges += 1
                    #print(f" Created {created_edges} relationships.")
            
        print(f" Created {created_edges} CoAuthor relationships")

    def add_shared_references_edge(self, min_overlap):
        """
        Create SHARED_REFERENCES edges between publications that cite common works.
        Weight equals the percentage of shared references.

        Args:
            min_overlap: minimum percentage of shared references (0.0-1.0) to create edge
        """
        works_data = self.data["works_data"]
        pub_keys = list(works_data.keys())
        created_edges = 0

        for i, pub1 in enumerate(pub_keys):
            refs1 = set(works_data[pub1].get("referenced_works", []))

            for pub2 in pub_keys[i + 1:]:
                refs2 = set(works_data[pub2].get("referenced_works", []))

                if not refs1 or not refs2:
                    continue

                shared_refs = refs1 & refs2
                overlap_pct = len(shared_refs) / max(len(refs1), len(refs2))

                if overlap_pct >= min_overlap:
                    self.driver.execute_query(
                        """
                        MATCH (p1:PUBLICATION {id: $pub1}), (p2:PUBLICATION {id: $pub2})
                        CREATE (p1)-[:SHARED_REFERENCES {overlap_percent: $overlap, shared_count: $count}]->(p2)
                        """,
                        pub1=pub1,
                        pub2=pub2,
                        overlap=round(overlap_pct, 3),
                        count=len(shared_refs),
                        database=self.db,
                    )
                    created_edges += 1

        print(f" Created {created_edges} Shared References relationships (min_overlap={min_overlap}).")

    def add_coauthor_overlap_edge(self, min_overlap):
        """
        Create COAUTHOR_OVERLAP edges weighted by percentage of shared co-authors.

        Args:
            min_overlap: minimum percentage of shared authors (0.0-1.0) to create edge
        """
        works_data = self.data["works_data"]
        pub_keys = list(works_data.keys())
        created_edges = 0

        for i, pub1 in enumerate(pub_keys):
            authors1 = {(a["id"], a["name"]) for a in works_data[pub1]["authors"]}

            for pub2 in pub_keys[i + 1:]:
                authors2 = {(a["id"], a["name"]) for a in works_data[pub2]["authors"]}

                if not authors1 or not authors2:
                    continue

                shared = authors1 & authors2
                overlap_pct = len(shared) / max(len(authors1), len(authors2))

                if overlap_pct >= min_overlap:
                    self.driver.execute_query(
                        """
                        MATCH (p1:PUBLICATION {id: $pub1}), (p2:PUBLICATION {id: $pub2})
                        CREATE (p1)-[:COAUTHOR_OVERLAP {overlap_percent: $overlap, shared_count: $count}]->(p2)
                        """,
                        pub1=pub1,
                        pub2=pub2,
                        overlap=round(overlap_pct, 3),
                        count=len(shared),
                        database=self.db,
                    )
                    created_edges += 1

        print(f" Created {created_edges} Coauthor Overlap relationships (min_overlap={min_overlap}).")

    def add_research_field_edge(self, min_overlap):
        """
        Create RESEARCH_FIELD edges between publications sharing fields using OpenAlex taxonomy.
        Weight equals the percentage of shared research fields.

        Args:
            min_overlap: minimum percentage of shared fields (0.0-1.0) to create edge
        """
        works_data = self.data["works_data"]
        pub_keys = list(works_data.keys())
        created_edges = 0

        research_fields_by_pub = {}
        pubs_with_topics = 0
        pubs_with_concepts = 0
        pubs_with_fields = 0

        def normalize_field_id(raw_id):
            if not raw_id or not isinstance(raw_id, str):
                return None
            if raw_id.startswith("https://openalex.org/"):
                return raw_id.replace("https://openalex.org/", "")
            return raw_id

        for pub_id in pub_keys:
            work = works_data[pub_id]
            fields = set()

            topics = work.get("topics", [])
            if isinstance(topics, list) and topics:
                pubs_with_topics += 1
                for topic in topics:
                    if not isinstance(topic, dict):
                        continue
                    field = topic.get("field") or {}
                    field_id = normalize_field_id(field.get("id") or field.get("display_name"))
                    if field_id:
                        fields.add(field_id)

            primary_topic = work.get("primary_topic")
            if isinstance(primary_topic, dict):
                field = primary_topic.get("field") or {}
                field_id = normalize_field_id(field.get("id") or field.get("display_name"))
                if field_id:
                    fields.add(field_id)

            concepts = work.get("concepts", [])
            if isinstance(concepts, list) and concepts:
                pubs_with_concepts += 1
                for concept in concepts:
                    if not isinstance(concept, dict):
                        continue
                    concept_id = normalize_field_id(concept.get("id") or concept.get("display_name"))
                    if concept_id:
                        fields.add(concept_id)

            raw_fields = work.get("fields")
            if isinstance(raw_fields, list) and raw_fields:
                for item in raw_fields:
                    if isinstance(item, dict):
                        field_id = normalize_field_id(item.get("id") or item.get("display_name"))
                    else:
                        field_id = normalize_field_id(item)
                    if field_id:
                        fields.add(field_id)

            if fields:
                pubs_with_fields += 1
            research_fields_by_pub[pub_id] = fields

        print(f" Research field extraction: {pubs_with_fields}/{len(pub_keys)} publications have fields.")
        print(f" Field sources: topics={pubs_with_topics}, concepts={pubs_with_concepts}.")

        for i, pub1 in enumerate(pub_keys):
            fields1 = research_fields_by_pub[pub1]

            for pub2 in pub_keys[i + 1:]:
                fields2 = research_fields_by_pub[pub2]

                if not fields1 or not fields2:
                    continue

                shared_fields = fields1 & fields2
                overlap_pct = len(shared_fields) / max(len(fields1), len(fields2))

                if overlap_pct >= min_overlap:
                    self.driver.execute_query(
                        """
                        MATCH (p1:PUBLICATION {id: $pub1}), (p2:PUBLICATION {id: $pub2})
                        CREATE (p1)-[:RESEARCH_FIELD {overlap_percent: $overlap, shared_count: $count}]->(p2)
                        """,
                        pub1=pub1,
                        pub2=pub2,
                        overlap=round(overlap_pct, 3),
                        count=len(shared_fields),
                        database=self.db,
                    )
                    created_edges += 1

        print(f" Created {created_edges} Research Field relationships (min_overlap={min_overlap}).")

    def cotitle_pairs_tfidf(self, min_similarity=0.60, max_features=10000):
        """
        Create COTITLE pairs using TF-IDF cosine similarity between publication titles.

        Args
            min_similarity: minimum cosine similarity to create an edge
            top_k: for each publication, create edges only to its top_k most similar publications

        Requirements
            scikit-learn must be installed (see requirements.txt)
        """
        works_data = self.data["works_data"]
        pub_ids = list(works_data.keys())
        titles = [works_data[pid].get("title") or "" for pid in pub_ids]

        vec = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=max_features
        )
        X = vec.fit_transform(titles)

        # sparse cosine sim
        S = cosine_similarity(X, dense_output=False).tocsr()

        pairs = []
        # iterate upper triangle only (undirected)
        S_coo = S.tocoo()
        for i, j, sim in zip(S_coo.row, S_coo.col, S_coo.data):
            if i < j and sim >= min_similarity:
                pairs.append((pub_ids[i], pub_ids[j], float(sim)))
        return pairs

    def add_cotitle_edge_from_pairs(self, pairs, threshold=0.60):
        created = 0
        for id1, id2, sim in pairs:
            if id1 == id2 or sim < threshold:
                continue
            a, b = (id1, id2) if id1 < id2 else (id2, id1)
            self.driver.execute_query(
                """
                MATCH (p1:PUBLICATION {id: $a}), (p2:PUBLICATION {id: $b})
                MERGE (p1)-[r:COTITLE]->(p2)
                SET r.similarity = $sim
                """,
                a=a, b=b, sim=float(sim), database=self.db
            )
            created += 1
        print(f"Created {created} cotitle relationships (cosine ≥ {threshold})")


def main(URI, USER, PASSWORD, DB, PATH):
    """
    Creates neo4j graph in database with publication node and coauthor, cotitle, covenue relationships

    Args
        uri (str): Neo4j URI (e.g. bolt://localhost:7687)
        user (str): instance username
        password (str): instance password
        db (str): database name
        data_path (str): Path to JSON created by neo4j_data.py (cache/<Author>_data.json)
    """

    imp = Neo4jImportData(URI, USER, PASSWORD, DB, PATH)

    imp.delete_all_nodes()

    # Add all publications
    imp.publication_as_nodes()

    # Add covenue, coauthor, cotitle, and overlap-based relationships
    imp.add_covenue_edge()
    imp.add_coauthor_edge()

    pairs = imp.cotitle_pairs_tfidf()
    imp.add_cotitle_edge_from_pairs(pairs)
    min_overlap = 0.5
    imp.add_shared_references_edge(min_overlap)
    imp.add_coauthor_overlap_edge(min_overlap)
    imp.add_research_field_edge(min_overlap)

    # Metrics
    imp.node_count()
    imp.edge_count()


if __name__ == "__main__":

    URI = "neo4j://127.0.0.1:7687"  # Example local instance
    USER = "neo4j"
    PASSWORD = "JohnGoat1000"
    DB = "neo4j"
    PATH = "/Users/seanmaniti/name_disam_exp/neo4j_and/cache/combined_data.json"

    main(URI, USER, PASSWORD, DB, PATH)
