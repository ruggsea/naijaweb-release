"""Build the binary keep/drop train and test sets from the raw annotations.

The filter makes one decision: does this document belong in the pretraining corpus. So the four codebook keys
(classifier/CODEBOOK.md) collapse to one bit -- keep = key 1 (right language, good quality), drop = keys 2/3/4. The 3-vs-4
distinction is where raters disagree most and it buys the filter nothing.

test  = the overlap documents (labelled by several trained raters), labelled by what most of them said. Each row carries
        `agreement`, 1.0 where every rater said the same and lower where only a majority did, so a document can be weighted
        or dropped at training time. A document whose raters split evenly is dropped from the test set.
train = the documents with a single rater, with every host that appears in test removed, so the split is by host and not
        at random.

Input: $NAIJAWEB_LABELS/labels.jsonl, one label per line (released with the corpus): doc_id, annotator, status
("void" labels are ignored), label (1-4), lang, url, text, and shared (1 for the overlap documents).
Output: $NAIJAWEB_LABELS/dataset/{train,test}.jsonl.
"""
import json, sys, collections
from pathlib import Path
from urllib.parse import urlparse
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config.py
from config import LABELS, need

ANNOT = Path(need(LABELS, "NAIJAWEB_LABELS"))
OUT = ANNOT / "dataset"

def keep(label):
    return int(label == 1)

def host(url):
    return urlparse(url).netloc

def doc(r, y, votes):
    return {"doc_id": r["doc_id"], "lang": r["lang"], "url": r["url"], "text": r["text"],
            "host": host(r["url"]), "keep": y, "n_labels": len(votes),
            "agreement": round(sum(1 for v in votes if v == y) / len(votes), 3)}

def by_doc():
    d = collections.defaultdict(list)
    for line in (ANNOT / "labels.jsonl").open():
        r = json.loads(line)
        if r["status"] != "void":
            d[r["doc_id"]].append(r)
    return d

def test_set(by_doc):
    test, tied = [], 0
    for rs in by_doc.values():
        if str(rs[0]["shared"]) != "1" or len(rs) < 2:
            continue
        votes = [keep(r["label"]) for r in rs]
        if sum(votes) * 2 == len(votes):
            tied += 1
            continue
        test.append(doc(rs[0], int(sum(votes) * 2 > len(votes)), votes))
    unan = sum(1 for d in test if d["agreement"] == 1.0)
    print(f"test: {len(test)} documents ({unan} unanimous), {tied} tied and dropped, {len({d['host'] for d in test})} hosts")
    return test

def train_set(by_doc, test_hosts):
    train, leaked = [], 0
    for rs in by_doc.values():
        if str(rs[0]["shared"]) == "1":
            continue
        if host(rs[0]["url"]) in test_hosts:
            leaked += 1
            continue
        votes = [keep(r["label"]) for r in rs]
        train.append(doc(rs[0], int(sum(votes) * 2 >= len(votes)), votes))
    print(f"train: {len(train)} documents, {leaked} held out for sharing a host with test")
    return train

def write(split, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{split}.jsonl"
    tmp = p.with_suffix(".tmp")
    with tmp.open("w") as f:
        for r in sorted(rows, key=lambda r: r["doc_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.rename(p)
    n = collections.Counter((r["lang"], r["keep"]) for r in rows)
    print(f"wrote {p.name}: " + "  ".join(f"{l} {n[(l,1)]} keep / {n[(l,0)]} drop" for l in ("hau", "ibo", "yor")))

if __name__ == "__main__":
    docs = by_doc()
    test = test_set(docs)
    write("test", test)
    write("train", train_set(docs, {d["host"] for d in test}))
