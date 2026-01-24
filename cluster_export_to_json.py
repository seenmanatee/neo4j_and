from neo4j import GraphDatabase
import pandas as pd
import json

# Connect to your Neo4j instance
URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", "JohnGoat1000")   # replace with your credentials
driver = GraphDatabase.driver(URI, auth=AUTH)

def export_clusters():
    query = """
    MATCH (p:PUBLICATION)
    RETURN p.id AS pubId, p.title AS title, p.community AS community, p.authors AS coauthors
    ORDER BY community, pubId
    """

    with driver.session() as session:
        result = session.run(query)
        records = [record.data() for record in result]

    # Convert to DataFrame for easy handling
    df = pd.DataFrame(records)
    print(df.head())

    # Group by community and dump as JSON
    grouped = df.groupby("community").apply(
        lambda x: x[["pubId", "title", "coauthors"]].to_dict("records")
    ).to_dict()

    with open("clusters.json", "w") as f:
        json.dump(grouped, f, indent=2)

    print("Exported to clusters.json")

if __name__ == "__main__":
    export_clusters()