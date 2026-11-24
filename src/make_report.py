"""Builds results/results_summary.md from the REAL metrics files written by evaluate.py.
Nothing is typed by hand: every number comes from runs_<type>/<model>_seed<N>/metrics_test.json."""
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "..")
KEYS = [("token_accuracy", "Token acc"), ("entity_precision_strict", "Entity P"),
        ("entity_recall_strict", "Entity R"), ("entity_f1_strict", "Entity F1 (strict)"),
        ("entity_f1_relaxed", "Entity F1 (relaxed)")]
ORDER = ["bilstm", "bilstm_attn", "cnn_bilstm", "cnn_bilstm_attn", "cnn_bigru_attn"]
out = ["# MultiClinAI results summary (auto-generated)\n",
       "All numbers are test-set scores computed by evaluate.py; values are mean +- std over seeds.",
       "Headline metric: strict entity-level F1. Token accuracy is inflated by the many O tokens.\n"]
for rdir in sorted(root.glob("runs_*")):
    etype = rdir.name.replace("runs_", "")
    by = defaultdict(list)
    for m in rdir.glob("*_seed*/metrics_test.json"):
        by[m.parent.name.rsplit("_seed", 1)[0]].append(json.load(open(m)))
    if not by:
        continue
    stats = root / f"processed_{etype}" / "stats.json"
    out.append(f"## Entity type: {etype.upper()}\n")
    if stats.exists():
        s = json.load(open(stats))
        out.append(f"Data: {s['documents']} documents, {s['tokens']} tokens, {s['entities']} entities "
                   f"({s['dropped_overlaps']} overlapping spans dropped).\n")
    out.append("| Model | " + " | ".join(n for _, n in KEYS) + " | Params | Seeds |")
    out.append("|---|" + "---|" * (len(KEYS) + 2))
    for name in [n for n in ORDER if n in by] + [n for n in by if n not in ORDER]:
        lst = by[name]; cells = []
        for k, _ in KEYS:
            v = np.array([x[k] for x in lst], float)
            cells.append(f"{v.mean():.4f} +- {v.std():.4f}")
        out.append(f"| {name} | " + " | ".join(cells) + f" | {int(lst[0]['params'])} | {len(lst)} |")
    out.append("")
(root / "results").mkdir(exist_ok=True)
(root / "results" / "results_summary.md").write_text("\n".join(out), encoding="utf-8")
print("\n".join(out))
