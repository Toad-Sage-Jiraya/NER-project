import argparse, json, random, re
from collections import Counter
from pathlib import Path
from common import save_json

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
TYPES = {"DISEASE": "diseases", "SYMPTOM": "symptoms", "PROCEDURE": "procedures"}
CORPORA = {
    "spaccc": {"DISEASE": "SPACCC_DisTEMIST_diseases",
               "SYMPTOM": "SPACCC_SympTEMIST_symptoms",
               "PROCEDURE": "SPACCC_MedProcNER_procedures", "sub": "train"},
    "cardio": {"DISEASE": "CardioCCC_diseases",
               "SYMPTOM": "CardioCCC_symptoms",
               "PROCEDURE": "CardioCCC_procedures", "sub": ""},
}

def read_ann(path, keep=TYPES):
    """Return list of (start, end, TYPE) from a BRAT .ann file."""
    spans = []
    if not path.exists():
        return spans
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split("\t")
        if len(parts) < 3 or not parts[0].startswith("T"):
            continue
        info = parts[1].split()
        etype = info[0].upper()
        if etype not in keep or ";" in parts[1]:   # skip notes + discontinuous
            continue
        spans.append((int(info[1]), int(info[2]), etype))
    return spans

def tokenize(text):
    return [(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]

def to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    taken = [False] * len(tokens)
    dropped = 0
    for s, e, t in sorted(spans, key=lambda x: x[1] - x[0], reverse=True):  # longest first
        idx = [i for i, (_, ts, te) in enumerate(tokens) if ts < e and te > s]
        if not idx or any(taken[i] for i in idx):
            dropped += 1
            continue
        for k, i in enumerate(idx):
            labels[i] = ("B-" if k == 0 else "I-") + t
            taken[i] = True
    return labels, dropped

def chunk(tokens, labels, max_len):
    """Cut into pieces <= max_len, preferring to cut after a full stop, never inside an entity."""
    out, start, n = [], 0, len(tokens)
    while start < n:
        end = min(start + max_len, n)
        if end < n:
            cut = None
            for i in range(end, start + max_len // 2, -1):       # look for a '.' boundary
                if tokens[i - 1][0] == "." and not labels[i].startswith("I-"):
                    cut = i
                    break
            if cut is None:
                for i in range(end, start, -1):                    # else any non-entity boundary
                    if not labels[i].startswith("I-"):
                        cut = i
                        break
            end = cut or end
        out.append((start, end))
        start = end
    return out

def resolve_paths(data_root, corpus, language, keep):
    """Resolves file locations supporting both v1.1 and legacy v1.0 directory layouts."""
    root = Path(data_root)

    # Check v1.1 structure (MultiClinNER/MultiClinNER-<lang>/...)
    v1_1_candidates = [
        root / "MultiClinNER" / f"MultiClinNER-{language}",
        root / f"MultiClinNER-{language}",
        root,
    ]
    for cand in v1_1_candidates:
        train_dir = cand / f"MultiClinNER-{language}-train"
        if train_dir.exists():
            dirs = {}
            for t in keep:
                sub = train_dir / f"MultiClinNER-{language}-train-{t.lower()}"
                dirs[t] = {
                    "txt": sub / "txt" if (sub / "txt").exists() else sub,
                    "ann": sub / "ann" if (sub / "ann").exists() else sub,
                }
            primary_txt = dirs[keep[0]]["txt"]
            all_txt = sorted(primary_txt.glob("*.txt"))
            if corpus == "spaccc":
                files = [f for f in all_txt if int(f.stem.split("-")[-1]) <= 750]
            elif corpus == "cardio":
                files = [f for f in all_txt if int(f.stem.split("-")[-1]) > 750]
            else:
                files = all_txt
            return dirs, files, True

    # Fallback: legacy v1.0 structure
    cfg = CORPORA.get(corpus, CORPORA["spaccc"])
    dirs = {t: {
        "txt": root / cfg[t] / language / cfg["sub"],
        "ann": root / cfg[t] / language / cfg["sub"],
    } for t in keep}
    txt_files = sorted(dirs[keep[0]]["txt"].glob("*.txt"))
    return dirs, txt_files, False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="folder MultiClinNER or MultiClinAI-training_data")
    ap.add_argument("--corpus", default="spaccc", choices=["spaccc", "cardio", "all"])
    ap.add_argument("--language", default="en")
    ap.add_argument("--types", default="DISEASE,SYMPTOM,PROCEDURE",
                    help="comma list, e.g. DISEASE   (one type = no overlaps lost)")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="processed")
    a = ap.parse_args()

    keep = [t.strip().upper() for t in a.types.split(",")]
    dirs, txt_files, is_v1_1 = resolve_paths(a.data_root, a.corpus, a.language, keep)
    print(f"Dataset layout: {'v1.1' if is_v1_1 else 'v1.0'}, documents found: {len(txt_files)}")

    docs, stats = {}, Counter()
    for f in txt_files:
        text = f.read_text(encoding="utf-8-sig")
        spans = []
        for t in keep:
            if is_v1_1:
                ann_name = f.name.replace(keep[0].lower(), t.lower()).replace(".txt", ".ann")
            else:
                ann_name = f.stem + ".ann"
            ann_file = dirs[t]["ann"] / ann_name
            spans += read_ann(ann_file, keep)
        bad = [x for x in spans if text[x[0]:x[1]].strip() == ""]
        spans = [x for x in spans if x not in bad]
        toks = tokenize(text)
        labels, dropped = to_bio(toks, spans)
        stats["documents"] += 1; stats["tokens"] += len(toks)
        stats["entities"] += len(spans); stats["dropped_overlaps"] += dropped
        rows = []
        for s, e in chunk(toks, labels, a.max_len):
            rows.append({"id": f.stem, "tokens": [t[0] for t in toks[s:e]], "labels": labels[s:e]})
        docs[f.stem] = rows

    ids = sorted(docs); random.Random(a.seed).shuffle(ids)
    n = len(ids); ntr, nva = int(.7 * n), int(.15 * n)
    splits = {"train": ids[:ntr], "val": ids[ntr:ntr + nva], "test": ids[ntr + nva:]}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    save_json(splits, out / "splits.json")
    for name, idl in splits.items():
        cnt = Counter()
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for d in idl:
                for r in docs[d]:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
                    cnt["chunks"] += 1
                    cnt["entities"] += sum(l.startswith("B-") for l in r["labels"])
        print(name, "docs:", len(idl), dict(cnt))
    save_json(dict(stats), out / "stats.json")
    print("overall:", dict(stats))

if __name__ == "__main__":
    main()
