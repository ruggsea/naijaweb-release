"""The classifier on the human test set, scored the way score.py scores the corpus.

  python classifier/test.py [classifier]   (default: NAIJAWEB_CLASSIFIER)

Test set: $NAIJAWEB_LABELS/dataset/test.jsonl from build_label_dataset.py, gold = the raters' majority keep/drop.
Same scoring as score.py: first 512 tokens, keep if p(keep) > 0.5. Prints accuracy, and precision and recall of keep,
per language and overall.
"""
import json, sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASSIFIER, LABELS, need

model_id = sys.argv[1] if len(sys.argv) > 1 else need(CLASSIFIER, "NAIJAWEB_CLASSIFIER")
rows = [json.loads(l) for l in open(Path(need(LABELS, "NAIJAWEB_LABELS")) / "dataset/test.jsonl")]
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
with torch.no_grad():
    pred = [int(model(**tok(r["text"], truncation=True, max_length=512, return_tensors="pt")).logits.softmax(-1)[0, 1] > 0.5)
            for r in rows]

for lang in ("hau", "ibo", "yor", "all"):
    idx = [i for i, r in enumerate(rows) if lang in ("all", r["lang"])]
    gold, p = [rows[i]["keep"] for i in idx], [pred[i] for i in idx]
    tp = sum(g and q for g, q in zip(gold, p))
    print(f"{lang}  docs {len(idx)}  accuracy {sum(g == q for g, q in zip(gold, p)) / len(idx):.3f}  "
          f"precision(keep) {tp / max(1, sum(p)):.3f}  recall(keep) {tp / max(1, sum(gold)):.3f}  keep-all baseline {sum(gold) / len(idx):.3f}")
