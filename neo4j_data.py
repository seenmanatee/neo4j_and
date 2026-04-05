"""
Neo4j data preparation script (step 1 of 2)

Purpose
- Fetch publications for an ambiguous author name from OpenAlex
- Materialize a compact JSON cache that the Neo4j importer can consume

What it creates
- cache/<Author Name>_data.json
  {
    "author_name": str,
    "author_data": {<openalex_author_id>: {...}},
    "works_data": {<openalex_work_id>: {id,title,year,authors,venue}},
    "author_id_to_label": {<openalex_author_id>: "0|1|2|..."}
  }

How to run
- Adjust `author_name` below (default: "David Nathan")
- Ensure Python can import `name_disambiguation/openAlex_to_HGCN.py` as `openAlex_to_HGCN`
  If needed, run from repo root and set PYTHONPATH: `PYTHONPATH=. python neo4j_data.py`
- This script does not require a running Neo4j instance; it only creates the cache JSON

Notes
- Labels are simple integer strings mapping each OpenAlex author-id candidate to a group label
- The importer script `neo4j_import.py` uses the generated JSON to create nodes/edges
"""

import openAlex_to_HGCN as oth
import argparse

def fetch_data(name):
    #1. Fetch author data from OpenAlex
    author_data = oth.fetch_author_data(name)

    #2. Mapping author ID to lable (0,1,2,...)
    author_id_to_label = {}
    for i, author_id in enumerate(author_data.keys()):
        author_id_to_label[author_id] = str(i)

    #3. Fetch works for each author
    works_data = {}
    for author_id, author in author_data.items():
        author_works = oth.fetch_works_for_author(author_id, name)
        author["works"] = [w["id"] for w in author_works]

        for work in author_works:
            works_data[work["id"]] = work

    #4. Save to JSON (consumed later by neo4j_import.py)
    oth.save_data_to_json(name, author_data, works_data, author_id_to_label)


def main():
    parser = argparse.ArgumentParser(description="Fetch publications for an ambiguous author name from OpenAlex")
    parser.add_argument("author_name", help="Author name to fetch data for (e.g., 'David Nathan')")
    args = parser.parse_args()
    author_name = args.author_name

    print(f"Retrieving publication data from {author_name}\n")

    fetch_data(author_name)

    print(f"\nData is imported to cache/{author_name}_data.json")


if __name__ == "__main__":
    main()
