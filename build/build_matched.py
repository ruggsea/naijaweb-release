"""Training sets for one comparison: documents kept by the quality classifier vs a random sample of the same pool.

  python build/build_matched.py <re-crawl build dir> [--write]

Pool = the 30-snapshot pool (P_ALL30) minus duplicates across snapshots, plus the re-crawl survivors of
build/recrawl_build.py (already deduplicated against P_ALL30 and inside itself). From both, drop the documents that
share a 13-word sequence with a held-out set and the Wikipedia pages that ARE held-out articles
(data/final/wiki_heldout_ids.txt). Then two versions:
  random  every document of the pool
  cls     documents with p > 0.5 from classifier/score.py (name `cls`; Wikipedia pages scored on infobox-stripped text)
and, in each version and language:
  1. site cap: no site (URL host, www. dropped) holds more than CAP of the language's tokens. The largest site over the
     cap loses documents at random (seeded) down to the cap, repeated until no site is over it.
  2. language quota: both versions get the same tokens per language, the smaller of the two after the cap; the version
     with more is cut at random (seeded, greedy fill).
Prints what each step removed and, per version, unique tokens, language shares and the top sites. --write writes
train/bins/matched_{random,cls}.bin (documents shuffled, each ended by <|endoftext|>) and data/matched/<version>.ids.
"""
import glob, io, json, sys, collections
from multiprocessing import Pool
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config.py
from config import DATA
from urllib.parse import urlparse
import numpy as np, zstandard
from tokenizers import ByteLevelBPETokenizer
from grams import grams

CAP, SEED = 0.20, 0
ROOT = Path(__file__).resolve().parent.parent
V = DATA / "wet/versions"
R = Path(sys.argv[1])
WRITE = "--write" in sys.argv
LANGS = ["hau_Latn", "ibo_Latn", "yor_Latn"]
tok = ByteLevelBPETokenizer(str(ROOT / "train/tok32k/vocab.json"), str(ROOT / "train/tok32k/merges.txt"))
held = set()
for f in sorted((ROOT / "data/heldout").glob("*.jsonl")):
    for line in open(f):
        held |= grams(json.loads(line)["text"])
across = {f"{l.split(chr(9))[0]}:{l.split(chr(9))[1]}" for l in open(ROOT / "data/dedup_filtered/ALL30/drop_global.tsv")}
wiki_held = set(open(ROOT / "data/final/wiki_heldout_ids.txt").read().split())

def site(url):
    h = urlparse(url).netloc.lower().split(":")[0]
    return h[4:] if h.startswith("www.") else h

def work(path):
    out = []
    with open(path, "rb") as fh:
        for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh, read_across_frames=True), encoding="utf-8"):
            r = json.loads(line)
            if r["id"] in across: continue
            ids = np.array(tok.encode(r["text"]).ids + [0], dtype=np.uint16)   # 0 = <|endoftext|>
            out.append(dict(id=r["id"], lang=r["lang"], site=site(r["url"]), recrawl="source" in r,
                            held=bool(grams(r["text"]) & held), wiki_held=r["id"] in wiki_held, ids=ids, n=len(ids)))
    return out

def tokens(ds): return sum(d["n"] for d in ds)
def line(label, ds): print(f"{label:58} {len(ds):>9,} docs {tokens(ds):>13,} tokens", flush=True)

def cap(ds, rng):
    """Cap every site at CAP of these documents' tokens; returns (kept docs, {site: (tokens before, after)})."""
    removed = {}
    while True:
        by = collections.defaultdict(list)
        for d in ds: by[d["site"]].append(d)
        total = tokens(ds)
        big, docs = max(by.items(), key=lambda kv: tokens(kv[1]))
        n = tokens(docs)
        if n <= CAP * total: return ds, removed
        allow = CAP / (1 - CAP) * (total - n)   # x = CAP * (rest + x)
        keep, got = [], 0
        for i in rng.permutation(len(docs)):
            if got + docs[i]["n"] <= allow: keep.append(docs[i]); got += docs[i]["n"]
        removed[big] = (removed.get(big, (n,))[0], got)
        drop = {id(d) for d in docs} - {id(d) for d in keep}
        ds = [d for d in ds if id(d) not in drop]

def fill(ds, target, rng):
    """Random documents, greedily, up to target tokens."""
    out, got = [], 0
    for i in rng.permutation(len(ds)):
        if got + ds[i]["n"] <= target: out.append(ds[i]); got += ds[i]["n"]
    return out

if __name__ == "__main__":
    files = sorted(glob.glob(str(V / "P_ALL30/part_*.jsonl.zst"))) + sorted(glob.glob(str(R / "*.jsonl.zst")))
    with Pool() as pool:
        docs = [d for o in pool.imap_unordered(work, files) for d in o]
    docs.sort(key=lambda d: d["id"])   # pool order must not depend on worker timing: the random draws index into it
    line("pool: P_ALL30 after duplicates across snapshots", [d for d in docs if not d["recrawl"]])
    line("pool: re-crawl after recrawl_build.py", [d for d in docs if d["recrawl"]])
    line("minus 13-word held-out overlap", [d for d in docs if d["held"]])
    line("minus Wikipedia identity hold-out", [d for d in docs if d["wiki_held"] and not d["held"]])
    line("minus other languages", [d for d in docs if d["lang"] not in LANGS and not d["held"] and not d["wiki_held"]])
    pool_docs = [d for d in docs if not d["held"] and not d["wiki_held"] and d["lang"] in LANGS]
    line("clean pool", pool_docs)

    p = {}
    for f in glob.glob(str(V / "scores/cls/*part_*.jsonl")) + glob.glob(str(V / "scores/cls/rescrape_*.jsonl")):
        p.update(json.load(open(f)))
    for f in V.glob("scores_wikistrip*/cls/rescrape_wikistrip.jsonl"):
        p.update(json.load(open(f)))
    unscored = [d for d in pool_docs if d["id"] not in p]
    assert all(d["recrawl"] for d in unscored), "a Common Crawl document has no score"
    line("minus re-crawl documents with no score (dropped)", unscored)
    pool_docs = [d for d in pool_docs if d["id"] in p]

    versions = {"random": pool_docs, "cls": [d for d in pool_docs if p[d["id"]] > 0.5]}
    capped = {}
    print(f"\nSite cap {CAP:.0%} of a language's tokens, per version:")
    for v, ds in versions.items():
        line(f"{v}: before the cap", ds)
        capped[v] = {}
        for l in LANGS:
            kept, removed = cap([d for d in ds if d["lang"] == l], np.random.RandomState(SEED))
            capped[v][l] = kept
            for s, (before, after) in sorted(removed.items(), key=lambda kv: kv[1][1] - kv[1][0]):
                print(f"   {v:6} {l}: {s:34} {before:>12,} -> {after:>12,} tokens ({after - before:+,})")
        line(f"{v}: after the cap", [d for l in LANGS for d in capped[v][l]])

    target = {l: min(tokens(capped[v][l]) for v in versions) for l in LANGS}
    budget = sum(target.values())
    print(f"\nLanguage quota: tokens per language = the smaller of the two versions after the cap; budget {budget:,}")
    for l in LANGS:
        print(f"   {l}: {target[l]:>12,} tokens ({100 * target[l] / budget:.1f}%); available " +
              ", ".join(f"{v} {tokens(capped[v][l]):,}" for v in versions))
    final = {v: [d for l in LANGS for d in fill(capped[v][l], target[l], np.random.RandomState(SEED + 1))] for v in versions}

    print("\nFinal versions:")
    for v, ds in final.items():
        n = tokens(ds)
        print(f"{v:6} {len(ds):>9,} docs {n:>13,} unique tokens (budget {budget:,}, {n - budget:+,}); "
              f"re-crawl {100 * tokens([d for d in ds if d['recrawl']]) / n:.1f}% of tokens")
        for l in LANGS:
            dl = [d for d in ds if d["lang"] == l]
            top = collections.Counter()
            for d in dl: top[d["site"]] += d["n"]
            print(f"   {l}: {100 * tokens(dl) / n:5.1f}% of tokens; biggest sites " +
                  ", ".join(f"{s} {100 * t / tokens(dl):.1f}%" for s, t in top.most_common(3)))
    a, b = set(d["id"] for d in final["random"]), set(d["id"] for d in final["cls"])
    print(f"\nrandom and cls share {len(a & b):,} documents ({100 * len(a & b) / len(a):.1f}% of random)")

    if WRITE:
        (ROOT / "data/matched").mkdir(exist_ok=True)
        for v, ds in final.items():
            out = ROOT / f"train/bins/matched_{v}.bin"
            ds = [ds[i] for i in np.random.RandomState(SEED + 2).permutation(len(ds))]
            np.concatenate([d["ids"] for d in ds]).tofile(str(out) + ".tmp")
            Path(str(out) + ".tmp").rename(out)
            (ROOT / f"data/matched/matched_{v}.ids").write_text("".join(d["id"] + "\n" for d in ds))
            print(f"wrote {out} ({tokens(ds):,} tokens)")
