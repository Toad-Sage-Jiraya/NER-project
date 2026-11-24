from pathlib import Path
import json

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_LABEL_ID = 0

LABELS = [
    "O",
    "B-DISEASE", "I-DISEASE",
    "B-SYMPTOM", "I-SYMPTOM",
    "B-PROCEDURE", "I-PROCEDURE",
]
LABEL2ID = {label: i + 1 for i, label in enumerate(LABELS)}
ID2LABEL = {i + 1: label for i, label in enumerate(LABELS)}
ID2LABEL[PAD_LABEL_ID] = "<PAD>"


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
