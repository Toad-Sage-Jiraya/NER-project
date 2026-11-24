import argparse, json, random
from collections import Counter
from pathlib import Path
import numpy as np
import tensorflow as tf
from common import LABEL2ID, save_json
from models import build_model, MODEL_NAMES

MAX_CHARS = 20

def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

def norm(tok):                       # words are lower-cased; characters keep their case
    return tok.lower()

def build_vocabs(rows, min_freq=2):
    wc = Counter(norm(t) for r in rows for t in r["tokens"])
    words = {"<PAD>": 0, "<UNK>": 1}
    for w, f in wc.most_common():
        if f >= min_freq:
            words[w] = len(words)
    chars = {"<PAD>": 0, "<UNK>": 1}
    for r in rows:
        for t in r["tokens"]:
            for ch in t:
                chars.setdefault(ch, len(chars))
    return words, chars

def encode(rows, words, chars, max_len):
    n = len(rows)
    W = np.zeros((n, max_len), np.int32)
    C = np.zeros((n, max_len, MAX_CHARS), np.int32)
    Y = np.zeros((n, max_len), np.int32)
    M = np.zeros((n, max_len), np.float32)
    for i, r in enumerate(rows):
        for j, t in enumerate(r["tokens"][:max_len]):
            W[i, j] = words.get(norm(t), 1)
            for k, ch in enumerate(t[:MAX_CHARS]):
                C[i, j, k] = chars.get(ch, 1)
            Y[i, j] = LABEL2ID[r["labels"][j]]
            M[i, j] = 1.0
    return W, C, Y, M

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="processed")
    ap.add_argument("--model", default="bilstm", choices=MODEL_NAMES)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); tf.random.set_seed(a.seed)
    train = read_jsonl(Path(a.data) / "train.jsonl")
    val = read_jsonl(Path(a.data) / "val.jsonl")
    words, chars = build_vocabs(train)
    Wt, Ct, Yt, Mt = encode(train, words, chars, a.max_len)
    Wv, Cv, Yv, Mv = encode(val, words, chars, a.max_len)

    run = Path(a.out) / f"{a.model}_seed{a.seed}"
    run.mkdir(parents=True, exist_ok=True)          # <- the folder the old code forgot to create
    save_json(words, run / "words.json"); save_json(chars, run / "chars.json")
    save_json({"max_len": a.max_len, "model": a.model, "seed": a.seed}, run / "config.json")

    model = build_model(a.model, len(words), len(chars), len(LABEL2ID) + 1, a.max_len)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",
                  weighted_metrics=["sparse_categorical_accuracy"])
    model.summary()
    save_json({"params": int(model.count_params())}, run / "params.json")

    use_c = len(model.inputs) == 2
    Xt = [Wt, Ct] if use_c else Wt
    Xv = [Wv, Cv] if use_c else Wv
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2)]
    hist = model.fit(Xt, Yt, sample_weight=Mt, validation_data=(Xv, Yv, Mv),
                     epochs=a.epochs, batch_size=a.batch_size, callbacks=cbs)
    model.save(run / "model.keras")
    save_json({k: [float(v) for v in vs] for k, vs in hist.history.items()}, run / "history.json")
    print("saved to", run)

if __name__ == "__main__":
    main()
