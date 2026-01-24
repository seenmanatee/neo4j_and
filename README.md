## Neo4j Graph for Name Disambiguation
This repo now includes a two-step workflow to build a publication graph in Neo4j for author name disambiguation experiments.

Shared IR
- Canonical mentions format is defined in `configs/mentions_ir.md`
- JSONL path: `data/processed/<dataset>/mentions.jsonl`

Virtual environment setup
- Create venv: `python3 -m venv .venv_name_disam`
- Activate: `source .venv_name_disam/bin/activate`
- Install deps: `pip install -r neo4j_and/requirements.txt`
- Run scripts with venv active (or use `.venv_name_disam/bin/python` directly)

1) Prepare OpenAlex data and cache
   - Script: `neo4j_data.py`
   - Action: fetch authors and publications for an ambiguous name and write JSON cache to `cache/<Author>_data.json`
   - Example run: python3 neo4j_data.py "David Nathan"

2) Import into Neo4j and build edges
   - Script: `neo4j_import.py`
   - Nodes: `PUBLICATION {id, title, year, authors(JSON string), venue}`
   - Edges:
     - `COAUTHOR {coauthor: JSON, weight: int}` between publications sharing any author
     - `COVENUE {venue}` between publications sharing the same venue
   - Usage: set `URI`, `USER`, `PASSWORD`, `DB`, `PATH` in the main block, then run `PYTHONPATH=. python neo4j_import.py`

3) Community detection + predictions
   - Script: `community_detection.py`
   - Action: run Leiden/Louvain on the Neo4j graph and emit `pred_clusters.jsonl`
   - Example run: `python3 neo4j_and/community_detection.py --output pred_clusters.jsonl`
   - Then evaluate with `python3 scripts/evaluate_b3.py --predictions pred_clusters.jsonl --mentions ...`

Optional: combine multiple cache files
- Script: `combine_cache_jsons.py`
- Action: merge all `*_data.json` files in a cache directory into one JSON that `neo4j_import.py` can read.
- Example run: `python3 neo4j_and/combine_cache_jsons.py --cache-dir neo4j_and/cache --output neo4j_and/cache/combined_data.json`
- Then set `PATH` in `neo4j_import.py` to `neo4j_and/cache/combined_data.json`.

Notes and observations
- Publications for "David Nathan" cluster clearly; promising for community detection
- Many more `COAUTHOR` than `COVENUE` edges; `COAUTHOR` likely contributes more to accuracy
- Edges are currently added directionally; clustering can treat graph as undirected

Planned next steps
- Add `COTITLE` edges using NLP-based title similarity — Implemented via TF-IDF cosine similarity in `neo4j_import.py` (use `add_cotitle_edge`) with thresholds.
- Explore Louvain community detection — Added `name_disambiguation/louvain_from_neo4j.py` to run Louvain on the imported Neo4j graph using igraph.

### Run the Neo4j pipeline
1) Prepare cache JSON
   - `PYTHONPATH=. python neo4j_data.py` (edits `author_name` as needed)
2) Import into Neo4j and build edges
   - Set credentials/DB in `neo4j_import.py`
   - `PYTHONPATH=. python neo4j_import.py`
   - This will add `PUBLICATION` nodes and `COAUTHOR`, `COVENUE`, and `COTITLE` edges
3) Run Louvain community detection
   - Set credentials/DB in `name_disambiguation/louvain_from_neo4j.py`
   - `PYTHONPATH=. python -m name_disambiguation.louvain_from_neo4j`
