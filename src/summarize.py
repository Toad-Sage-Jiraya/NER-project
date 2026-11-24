import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
runs = Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
keys = ["token_accuracy", "entity_precision_strict", "entity_recall_strict",
        "entity_f1_strict", "entity_f1_relaxed", "params"]
by = defaultdict(list)
for m in sorted(runs.glob("*_seed*/metrics_test.json")):
    by[m.parent.name.rsplit("_seed", 1)[0]].append(json.load(open(m)))
print(f"{'model':18s}" + "".join(f"{k[:22]:>24s}" for k in keys) + "   n")
for name, lst in by.items():
    cells = []
    for k in keys:
        v = np.array([x[k] for x in lst], float)
        cells.append(f"{v.mean():.0f}" if k == "params" else f"{v.mean():.4f}+-{v.std():.4f}")
    print(f"{name:18s}" + "".join(f"{c:>24s}" for c in cells) + f"  {len(lst)}")
