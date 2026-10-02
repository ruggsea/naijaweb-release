"""The held-out text the three corpus versions are scored on. Four sets, kept apart
and reported separately, because a single news-like set would just reward whichever
filter likes news.

  flores     FLORES paragraphs, taken from Belebele (facebook/belebele), because the
             FLORES+ repo itself is gated and this account has no access
  news       MasakhaNEWS test split, news articles
  wiki       Wikipedia articles, encyclopedic
  exam       AfriMMLU test questions, school exam text

Writes data/heldout/<set>_<lang>.jsonl with {"text": ...} per line, for hau, ibo, yor.
"""
import json, random
from pathlib import Path
from datasets import load_dataset

OUT = Path(__file__).resolve().parent.parent / "data/heldout"
OUT.mkdir(parents=True, exist_ok=True)
LANGS = {"hau": ("hau_Latn", "hau", "ha"), "ibo": ("ibo_Latn", "ibo", "ig"), "yor": ("yor_Latn", "yor", "yo")}
MAX_DOCS = 500

def write(name, lang, texts):
    texts = [t.strip() for t in texts if t and len(t.strip()) > 40]
    (OUT / f"{name}_{lang}.jsonl").write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    print(name, lang, len(texts), flush=True)

for lang, (flores, iso3, iso2) in LANGS.items():
    d = load_dataset("facebook/belebele", flores, split="test")
    write("flores", lang, sorted(set(d["flores_passage"])))

    d = load_dataset("masakhane/masakhanews", iso3, split="test")
    write("news", lang, [f"{h}\n{t}" for h, t in zip(d["headline"], d["text"])][:MAX_DOCS])

    d = load_dataset("wikimedia/wikipedia", f"20231101.{iso2}", split="train")
    idx = random.Random(0).sample(range(len(d)), min(MAX_DOCS, len(d)))
    write("wiki", lang, [d[i]["text"] for i in idx])

    d = load_dataset("masakhane/afrimmlu", iso3, split="test")
    write("exam", lang, [f"{q}\n" + " ".join(eval(c) if isinstance(c, str) else c) for q, c in zip(d["question"], d["choices"])])
