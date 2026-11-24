import argparse, json, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
from seqeval.scheme import IOB2
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from common import ID2LABEL, load_json, save_json
from train import read_jsonl, encode
import models

def spans_of(tags):
    out, start, typ = set(), None, None
    for i, t in enumerate(list(tags) + ["O"]):
        if t == "O" or t.startswith("B-") or (t.startswith("I-") and t[2:] != typ):
            if start is not None:
                out.add((start, i, typ)); start = None; typ = None
            if t != "O":
                start, typ = i, t[2:]
    return out

def relaxed_prf(y_true, y_pred):
    tp_p = n_p = tp_r = n_r = 0
    for t, p in zip(y_true, y_pred):
        ts, ps = spans_of(t), spans_of(p)
        n_p += len(ps); n_r += len(ts)
        tp_p += sum(any(q[2] == s[2] and q[0] < s[1] and s[0] < q[1] for q in ts) for s in ps)
        tp_r += sum(any(q[2] == s[2] and q[0] < s[1] and s[0] < q[1] for q in ps) for s in ts)
    P = tp_p / n_p if n_p else 0.0
    R = tp_r / n_r if n_r else 0.0
    return P, R, (2 * P * R / (P + R) if P + R else 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="processed")
    ap.add_argument("--run", required=True, help="e.g. runs/bilstm_seed42")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    run = Path(a.run)
    cfg = load_json(run / "config.json"); max_len = cfg["max_len"]
    words, chars = load_json(run / "words.json"), load_json(run / "chars.json")
    rows = read_jsonl(Path(a.data) / f"{a.split}.jsonl")
    W, C, Y, M = encode(rows, words, chars, max_len)
    model = tf.keras.models.load_model(run / "model.keras", safe_mode=False)
    X = [W, C] if len(model.inputs) == 2 else W
    t0 = time.time()
    probs = model.predict(X, batch_size=64, verbose=0)
    ms = (time.time() - t0) * 1000 / max(1, int(M.sum())) * 1000
    pred = np.argmax(probs[:, :, 1:], axis=-1) + 1          # never predict the PAD class

    y_true, y_pred = [], []
    for i, r in enumerate(rows):
        n = min(len(r["labels"]), max_len)
        y_true.append(r["labels"][:n])
        y_pred.append([ID2LABEL[int(pred[i, j])] for j in range(n)])

    flat_t = [x for s in y_true for x in s]; flat_p = [x for s in y_pred for x in s]
    labs = [l for l in sorted(set(flat_t)) if l != "O"]
    tok_p, tok_r, tok_f, _ = precision_recall_fscore_support(flat_t, flat_p, labels=labs, average="micro", zero_division=0)
    rel = relaxed_prf(y_true, y_pred)
    res = {
        "token_accuracy": accuracy_score(flat_t, flat_p),
        "token_micro_precision_no_O": tok_p, "token_micro_recall_no_O": tok_r, "token_micro_f1_no_O": tok_f,
        "entity_precision_strict": precision_score(y_true, y_pred, mode="strict", scheme=IOB2),
        "entity_recall_strict": recall_score(y_true, y_pred, mode="strict", scheme=IOB2),
        "entity_f1_strict": f1_score(y_true, y_pred, mode="strict", scheme=IOB2),
        "entity_precision_relaxed": rel[0], "entity_recall_relaxed": rel[1], "entity_f1_relaxed": rel[2],
        "ms_per_1000_tokens": ms,
        "params": load_json(run / "params.json")["params"],
    }
    print(json.dumps(res, indent=2))
    print(classification_report(y_true, y_pred, mode="strict", scheme=IOB2, digits=4))
    save_json(res, run / f"metrics_{a.split}.json")
    with open(run / f"predictions_{a.split}.jsonl", "w", encoding="utf-8") as f:
        for r, t, p in zip(rows, y_true, y_pred):
            f.write(json.dumps({"id": r["id"], "tokens": r["tokens"][:len(t)], "true": t, "pred": p}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
