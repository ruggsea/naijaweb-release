"""Token-matched training sets: our corpus (the release recipe, selected by the quality classifier) vs WURA vs FineWeb-2 as published.

  python build/build_baselines.py <re-crawl build dir> [--write]

Matched on the total only (no per-language or per-site matching):
the same number of unique tokens per version, set by the smallest; each keeps its own language mix and sites.
Three versions, each put through the same steps with counts printed after each:
  ours   the release: the base pool (P_ALL30 minus duplicates across snapshots), the scored re-crawl build, and the
         FineWeb-2 fold-in (F_fineweb2: FineWeb-2 documents not in P_ALL30, through our filters, build/fineweb2_pool.py);
         fold-in documents sharing a MinHash band key with a pool or re-crawl document dropped; our classifier's p > 0.5
  wura   WURA documents-v1.0, train split (the eval split is WURA's own held-out), text = content
  fw2    FineWeb-2 train split, hau/ibo/yor, as published: no classifier, no filters of ours
  1. near-duplicates inside WURA and FineWeb-2 with our MinHash (filter/dedup.py; documents sharing any band key are
     one cluster, the first kept), so tokens are unique tokens everywhere (ours is already deduplicated this way)
  2. decontamination as for every version: drop a document sharing a 13-word sequence with a held-out set (counted per
     held-out set), and a Wikipedia page that is a held-out article (P_ALL30 and re-crawl: data/final/wiki_heldout_ids.txt;
     the others: the page title from the URL against data/heldout_ids/wiki_*.jsonl)
  3. every version cut at random (seeded) to the smallest version's total
Then per version: language shares and biggest sites; and how much of ours is also in FineWeb-2 (same FineWeb-2 id, or a
shared MinHash band key with a FineWeb-2 document).
--write writes train/bins/corpus_{ours,wura,fw2}.bin (documents shuffled, each ended by <|endoftext|>) and data/matched/<name>.ids.
"""
import glob, io, json, sys, collections, importlib.util
from multiprocessing import Pool
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config.py
from config import DATA
from urllib.parse import urlparse, parse_qs, unquote
import numpy as np, pyarrow.parquet as pq, zstandard
from tokenizers import ByteLevelBPETokenizer
from grams import grams

SEED = 0
ROOT = Path(__file__).resolve().parent.parent
V = DATA / "wet/versions"
B = DATA / "baselines"
R = Path(sys.argv[1])
WRITE = "--write" in sys.argv
NAMES = {"ours": "corpus_ours", "wura": "corpus_wura", "fw2": "corpus_fw2"}
LANGS = ["hau_Latn", "ibo_Latn", "yor_Latn"]
tok = ByteLevelBPETokenizer(str(ROOT / "train/tok32k/vocab.json"), str(ROOT / "train/tok32k/merges.txt"))
spec = importlib.util.spec_from_file_location("dd", ROOT / "filter/dedup.py"); dd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dd)
held = {f.stem: set().union(*(grams(json.loads(l)["text"]) for l in open(f))) for f in sorted((ROOT / "data/heldout").glob("*.jsonl"))}
across = {f"{l.split(chr(9))[0]}:{l.split(chr(9))[1]}" for l in open(ROOT / "data/dedup_filtered/ALL30/drop_global.tsv")}
wiki_held_ids = set(open(ROOT / "data/final/wiki_heldout_ids.txt").read().split())

def title_of(url):   # as build/wiki_identity.py (not imported: that module runs its whole job at import)
    u = urlparse(url)
    t = u.path[len("/wiki/"):] if u.path.startswith("/wiki/") else parse_qs(u.query).get("title", [None])[0]
    return unquote(t).replace("_", " ").strip().casefold() if t else None

held_titles = {title_of(json.loads(l)["url"]) for f in (ROOT / "data/heldout_ids").glob("wiki_*.jsonl") for l in open(f)}

def site(url):
    h = urlparse(url or "").netloc.lower().split(":")[0]
    return h[4:] if h.startswith("www.") else h

def doc(id, lang, url, text, keys=False, **extra):
    g = grams(text)
    ids = np.array(tok.encode(text).ids + [0], dtype=np.uint16)   # 0 = <|endoftext|>
    d = dict(id=id, lang=lang, site=site(url), ids=ids, n=len(ids), held=[s for s, h in held.items() if g & h],
             wiki_held=site(url).endswith("wikipedia.org") and title_of(url) in held_titles, **extra)   # ours overrides it
    if keys: d["keys"] = [k * 16 + b for b, k in enumerate(dd.band_keys(dd.signature(text)))]
    return d

def ours_file(path):
    out = []
    with open(path, "rb") as fh:
        for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh, read_across_frames=True), encoding="utf-8"):
            r = json.loads(line)
            if r["id"] in across or r["lang"] not in LANGS: continue
            part = {"fineweb2": "fw2", "rescrape": "recrawl"}.get(r.get("source"), "cc")
            d = doc(r["id"], r["lang"], r["url"], r["text"], keys=True, recrawl=part == "recrawl", part=part)
            if part != "fw2": d["wiki_held"] = r["id"] in wiki_held_ids
            out.append(d)
    return out

def base(args):
    return doc(*args, keys=True, recrawl=False)

def base_args(name):
    if name == "wura":
        for lang in ("hau", "ibo", "yor"):
            for i, l in enumerate(open(B / f"wura/train/{lang}.jsonl")):
                r = json.loads(l); yield (f"wura:{lang}:{i}", lang + "_Latn", r.get("url"), r["content"])
    else:
        for lang in LANGS:
            for f in sorted(glob.glob(str(B / f"fineweb2/data/{lang}/train/*.parquet"))):
                t = pq.read_table(f, columns=["id", "url", "text"]).to_pydict()
                yield from ((f"fineweb2:{i}", lang, u, x) for i, u, x in zip(t["id"], t["url"], t["text"]))

def dedup(ds):
    parent = list(range(len(ds)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    owner = {}
    for i, d in enumerate(ds):
        for k in d["keys"]:
            j = owner.setdefault(k, i)
            if j != i: parent[find(i)] = find(j)
    first = {}
    for i in range(len(ds)): first.setdefault(find(i), i)
    return [ds[i] for i in sorted(first.values())]

def tokens(ds): return sum(d["n"] for d in ds)
def line(label, ds): print(f"{label:58} {len(ds):>9,} docs {tokens(ds):>13,} tokens", flush=True)

def fill(ds, target, rng):
    out, got = [], 0
    for i in rng.permutation(len(ds)):
        if got + ds[i]["n"] <= target: out.append(ds[i]); got += ds[i]["n"]
    return out

def decontaminate(v, ds):
    per_set = collections.Counter(s for d in ds for s in d["held"])
    line(f"{v}: minus 13-word held-out overlap", [d for d in ds if d["held"]])
    print(f"   documents overlapping each held-out set: " + ", ".join(f"{s} {per_set[s]:,}" for s in held), flush=True)
    line(f"{v}: minus Wikipedia identity hold-out", [d for d in ds if d["wiki_held"] and not d["held"]])
    out = [d for d in ds if not d["held"] and not d["wiki_held"]]
    line(f"{v}: clean", out)
    return out

if __name__ == "__main__":
    versions = {}
    files = (sorted(glob.glob(str(V / "P_ALL30/part_*.jsonl.zst"))) + sorted(glob.glob(str(R / "*.jsonl.zst"))) +
             sorted(glob.glob(str(V / "F_fineweb2/part_*.jsonl.zst"))))
    with Pool(48) as pool:
        ds = [d for o in pool.imap_unordered(ours_file, files) for d in o]
    ds.sort(key=lambda d: d["id"])   # order must not depend on worker timing: the random draws index into it
    line("ours: P_ALL30 after duplicates across snapshots, hau/ibo/yor", [d for d in ds if d["part"] == "cc"])
    line("ours: re-crawl after recrawl_build.py, hau/ibo/yor", [d for d in ds if d["part"] == "recrawl"])
    line("ours: FineWeb-2 fold-in (F_fineweb2)", [d for d in ds if d["part"] == "fw2"])
    seen = {k for d in ds if d["part"] != "fw2" for k in d["keys"]}
    dup = [d for d in ds if d["part"] == "fw2" and seen.intersection(d["keys"])]
    line("ours: minus fold-in sharing a band key with pool or re-crawl", dup)
    dup = {id(d) for d in dup}; ds = [d for d in ds if id(d) not in dup]; del seen
    ds = decontaminate("ours", ds)
    cls = {}
    for f in glob.glob(str(V / "scores/cls/*part_*.jsonl")) + glob.glob(str(V / "scores/cls/rescrape_*.jsonl")):
        cls.update(json.load(open(f)))
    cls.update(json.load(open(V / "scores_wikistrip/cls/rescrape_wikistrip.jsonl")))
    cls.update(json.load(open(V / "scores_wikistrip30/cls/rescrape_wikistrip.jsonl")))
    unscored = [d for d in ds if d["id"] not in cls]
    assert all(d["recrawl"] for d in unscored), "a Common Crawl document has no score"
    line("ours: minus re-crawl documents with no score (dropped)", unscored)
    ds = [d for d in ds if d["id"] in cls and cls[d["id"]] > 0.5]
    line("ours: our classifier's p > 0.5", ds)
    for p in ("cc", "recrawl", "fw2"): line(f"   of which {p}", [d for d in ds if d["part"] == p])
    versions["ours"] = ds
    del ds

    for v in ("wura", "fw2"):
        with Pool(64) as pool:
            ds = [dict(d, part=v) for d in pool.map(base, list(base_args(v)), chunksize=500)]
        line(f"{v}: as published", ds)
        ds = dedup(ds)
        line(f"{v}: one per MinHash cluster", ds)
        versions[v] = decontaminate(v, ds)

    budget = min(tokens(ds) for ds in versions.values())
    print(f"\nBudget: the smallest version's total, {budget:,} unique tokens (" +
          ", ".join(f"{v} has {tokens(ds):,}" for v, ds in versions.items()) + ")")
    final = {v: fill(ds, budget, np.random.RandomState(SEED + 1)) for v, ds in versions.items()}

    def describe(v, ds):
        n = tokens(ds)
        print(f"{v:5} {len(ds):>9,} docs {n:>13,} unique tokens; " + ", ".join(
            f"{l} {100 * tokens([d for d in ds if d['lang'] == l]) / n:.1f}%" for l in LANGS))
        if v == "ours":
            print("   parts: " + ", ".join(f"{p} {sum(d['part'] == p for d in ds):,} docs {tokens([d for d in ds if d['part'] == p]):,} tokens "
                                         f"({100 * tokens([d for d in ds if d['part'] == p]) / n:.1f}%)" for p in ("cc", "recrawl", "fw2")))
        top = collections.Counter()
        for d in ds: top[d["site"]] += d["n"]
        print(f"   {len(top):,} sites; biggest: " + ", ".join(f"{s} {100 * t / n:.1f}%" for s, t in top.most_common(10)))
        for l in LANGS:
            dl = [d for d in ds if d["lang"] == l]
            tl = collections.Counter()
            for d in dl: tl[d["site"]] += d["n"]
            print(f"   {l}: biggest sites " + ", ".join(f"{s} {100 * t / max(1, tokens(dl)):.1f}%" for s, t in tl.most_common(5)))

    def overlap(label, ours, fw2):
        fw_ids = {d["id"] for d in fw2}; fw_keys = {k for d in fw2 for k in d["keys"]}
        same = [d for d in ours if d["id"] in fw_ids]
        near = [d for d in ours if d["id"] not in fw_ids and fw_keys.intersection(d["keys"])]
        print(f"ours in FineWeb-2, {label}: same FineWeb-2 id {len(same):,} docs {tokens(same):,} tokens; other docs sharing a band key "
              f"{len(near):,} docs {tokens(near):,} tokens; together {100 * (tokens(same) + tokens(near)) / tokens(ours):.1f}% of ours' tokens")

    print("\nBefore the cut:")
    for v, ds in versions.items(): describe(v, ds)
    overlap("before the cut (ours vs all of clean FineWeb-2)", versions["ours"], versions["fw2"])
    print("\nFinal versions:")
    for v, ds in final.items(): describe(v, ds)
    overlap("final versions", final["ours"], final["fw2"])

    if WRITE:
        (ROOT / "data/matched").mkdir(exist_ok=True)
        for v, ds in final.items():
            out = ROOT / f"train/bins/{NAMES[v]}.bin"
            ds = [ds[i] for i in np.random.RandomState(SEED + 2).permutation(len(ds))]
            ids = "".join(d["id"] + "\n" for d in ds)
            if out.exists():   # an existing bin is never rewritten; a rebuild must reproduce it exactly
                assert (ROOT / f"data/matched/{NAMES[v]}.ids").read_text() == ids, f"{out} exists and this build differs"
                print(f"{out} exists, same documents in the same order: kept"); continue
            np.concatenate([d["ids"] for d in ds]).tofile(str(out) + ".tmp")
            Path(str(out) + ".tmp").rename(out)
            (ROOT / f"data/matched/{NAMES[v]}.ids").write_text(ids)
            print(f"wrote {out} ({tokens(ds):,} tokens)")
