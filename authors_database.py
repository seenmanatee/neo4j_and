import json
import subprocess
import sys

# ── Config ────────────────────────────────────────────────────────────────────
NAMES_FILE = "name_medium.json"          # change to your actual filename
SCRIPT     = "neo4j_data.py"
# ──────────────────────────────────────────────────────────────────────────────

def load_names(path):
    with open(path, "r") as f:
        return json.load(f)

def run_author(name):
    print(f"\n{'─'*50}")
    print(f"▶ Running: {name}")
    print(f"{'─'*50}")
    result = subprocess.run(
        ["python3", SCRIPT, name],
        capture_output=False   # streams output live to terminal
    )
    if result.returncode != 0:
        print(f"⚠️  Warning: script exited with code {result.returncode} for '{name}'")

def main():
    entries = load_names(NAMES_FILE)

    # Optional: filter by category
    # entries = [e for e in entries if e["category"] == "easy"]

    total = len(entries)
    print(f"Found {total} authors to process.")

    for i, entry in enumerate(entries, 1):
        name = entry["name"]
        print(f"\n[{i}/{total}] {name}  (complexity: {entry.get('complexity', '?')}, category: {entry.get('category', '?')})")
        run_author(name)

    print(f"\n Done — processed {total} authors.")

if __name__ == "__main__":
    main()