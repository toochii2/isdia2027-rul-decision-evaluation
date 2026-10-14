"""Merge per-subset JSON results into one file (keys are subsets).

Usage:
    python scripts/merge_json.py OUT_JSON IN_JSON [IN_JSON ...]
"""

import json, sys

out = {}
for path in sys.argv[2:]:
    out.update(json.load(open(path)))
json.dump(out, open(sys.argv[1], "w"))
print("merged", sorted(out), "->", sys.argv[1])
