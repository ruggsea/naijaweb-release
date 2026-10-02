"""FineWeb-2 fold-in: FineWeb-2's Hausa/Igbo/Yoruba documents that are not already in our pool,
put through our own steps, written as a pool version the classifiers and the release build read like P_ALL30.

  python build/fineweb2_pool.py [--dry]    -> $NAIJAWEB_DATA/wet/versions/F_fineweb2/part_NNNN.jsonl.zst

Records {id: "fineweb2:<FineWeb-2 id>", url, lang, text, source: "fineweb2", dump}. Steps, counted after each:
  0. FineWeb-2 train split, hau/ibo/yor ($NAIJAWEB_DATA/baselines/fineweb2, from build/get_baselines.sh)
  1. our boilerplate removal (filter/apply_filters.content: a line on 3+ pages of the same host in the same snapshot
     goes, then repeats of the title line) and our rule filters (filter/filters.reason)
  2. our language ID: GlotLID on the text, label in hau/ibo/yor and p >= 0.3 (crawl/wet_lid.THETA); lang = our label
  3. near-duplicates inside FineWeb-2 across its snapshots (our MinHash, filter/dedup.py; documents sharing any band
     key are one cluster), one kept per cluster, the smallest normalised URL
  4. drop what is already in our pool: same normalised URL as a P_ALL30 document, or a shared band key with ALL30
The 13-word test-set rule and the Wikipedia identity hold-out are applied at build time from the text, exactly as for
P_ALL30 (build/build_baselines.py); their counts here are for the report only. Builds in OUT.tmp and
renames it at the end; refuses to run if OUT exists.
"""
import glob, io, json, os, re, sys, collections, importlib.util
from multiprocessing import Pool
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config.py
from config import DATA
import pyarrow.parquet as pq, zstandard
from tokenizers import ByteLevelBPETokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "filter")); sys.path.insert(0, str(ROOT / "crawl"))
import filters as F, apply_filters as AF, wet_lid
from grams import grams
spec = importlib.util.spec_from_file_location("dd", ROOT / "filter/dedup.py"); dd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dd)
B = DATA / "baselines/fineweb2/data"
OUT = DATA / "wet/versions/F_fineweb2"
DRY = "--dry" in sys.argv
PART = 10_000
assert DRY or not (OUT.exists() or Path(str(OUT) + ".tmp").exists()), f"{OUT} or {OUT}.tmp exists"
tok = ByteLevelBPETokenizer(str(ROOT / "train/tok32k/vocab.json"), str(ROOT / "train/tok32k/merges.txt"))
def norm(u): return re.sub(r"^https?://(www\.)?", "", (u or "").split("#")[0]).rstrip("/").lower()
def recs(p):
    with open(p, "rb") as fh:
        for l in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh, read_across_frames=True), encoding="utf-8"):
            yield json.loads(l)

def step12(d):
    """Boilerplate + rule filters, then our LID. Returns the record with cleaned text, or (reason,) if dropped."""
    text = AF.content(d["text"], F.host_of(d["url"]), LINE_PAGES)
    why = F.reason({"url": d["url"], "text": text, "lang": d["fw_lang"]})
    if why: return ("filter:" + why.split(":")[0],)
    lab, p = wet_lid.lid(text)
    if lab not in wet_lid.KEEP or p < wet_lid.THETA: return ("lid",)
    keys = [k * 16 + b for b, k in enumerate(dd.band_keys(dd.signature(text)))]
    return {**d, "text": text, "lang": lab, "keys": keys, "ntok": len(tok.encode(text).ids) + 1}

HELD = set()
def overlaps(text): return bool(grams(text) & HELD)

def report(label, ds): print(f"{label:60} {len(ds):>9,} docs {sum(d['ntok'] for d in ds):>13,} tokens", flush=True)

if __name__ == "__main__":
    docs = []
    for lang in ("hau", "ibo", "yor"):
        for f in sorted(glob.glob(str(B / f"{lang}_Latn/train/*.parquet"))):
            t = pq.read_table(f, columns=["id", "url", "text", "dump"]).to_pydict()
            docs += [dict(id="fineweb2:" + i, url=u, text=x, dump=dp, fw_lang=lang) for i, u, x, dp in zip(t["id"], t["url"], t["text"], t["dump"])]
    print(f"0. FineWeb-2 train, hau/ibo/yor: {len(docs):,} docs; by language {collections.Counter(d['fw_lang'] for d in docs)}", flush=True)
    LINE_PAGES = collections.Counter()   # (host, line) -> pages, per snapshot as in apply_filters (one crawl at a time)
    by_dump = collections.defaultdict(list)
    for d in docs: by_dump[d["dump"]].append(d)
    wet_lid.lid("load the model once here, so the forked workers share it")
    kept, dropped = [], collections.Counter()
    for dump, ds in sorted(by_dump.items()):
        LINE_PAGES = collections.Counter()
        for d in ds:
            host = F.host_of(d["url"])
            LINE_PAGES.update((host, l) for l in {l.strip() for l in d["text"].splitlines() if l.strip()})
        with Pool(32) as pool:   # forked after LINE_PAGES is set, so the workers see this snapshot's counts
            for r in pool.imap(step12, ds, chunksize=200):
                if isinstance(r, tuple): dropped[r[0]] += 1
                else: kept.append(r)
    del docs, by_dump, LINE_PAGES
    report("1+2. after boilerplate, rule filters and our language ID", kept)
    print(f"   removed: {dict(dropped.most_common())}")
    print(f"   our language label vs FineWeb-2's: {collections.Counter((d['fw_lang'], d['lang'][:3]) for d in kept if d['lang'][:3] != d['fw_lang'])}")

    parent = list(range(len(kept)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    owner = {}
    for i, d in enumerate(kept):
        for k in d["keys"]:
            j = owner.setdefault(k, i)
            if j != i: parent[find(i)] = find(j)
    del owner
    best = {}
    for i, d in enumerate(kept):
        r = find(i)
        if r not in best or norm(d["url"]) < norm(kept[best[r]]["url"]): best[r] = i
    uniq = [kept[i] for i in sorted(best.values())]
    del kept, parent, best
    report("3. one per MinHash cluster inside FineWeb-2", uniq)

    ours_keys = set()
    for f in sorted(glob.glob(str(ROOT / "data/dedup_filtered/ALL30/sig_*.jsonl"))):
        for l in open(f): ours_keys.update(k * 16 + b for b, k in enumerate(json.loads(l)[5]))
    ours_urls = {norm(r["url"]) for p in glob.glob(str(DATA / "wet/versions/P_ALL30/*.jsonl.zst")) for r in recs(p)}
    new = [d for d in uniq if norm(d["url"]) not in ours_urls and not any(k in ours_keys for k in d["keys"])]
    del ours_keys, ours_urls
    report("4. not in our pool (URL or shared band key with ALL30)", new)
    for l in ("hau_Latn", "ibo_Latn", "yor_Latn"): report(f"   {l}", [d for d in new if d["lang"] == l])

    for f in sorted((ROOT / "data/heldout").glob("*.jsonl")):
        for line in open(f): HELD |= grams(json.loads(line)["text"])
    with Pool(32) as pool: n_held = sum(pool.map(overlaps, [d["text"] for d in new], chunksize=500))
    print(f"   report only, applied at build time: {n_held:,} share a 13-word sequence with a held-out set; "
          f"{sum('wikipedia.org' in norm(d['url']) for d in new):,} are Wikipedia pages (the identity hold-out is checked at build time)")

    if not DRY:
        tmp = Path(str(OUT) + ".tmp"); tmp.mkdir()
        new.sort(key=lambda d: d["id"])
        for n in range(0, len(new), PART):
            with open(tmp / f"part_{n // PART:04d}.jsonl.zst", "wb") as fh, zstandard.ZstdCompressor().stream_writer(fh) as w:
                for d in new[n:n + PART]:
                    w.write((json.dumps(dict(id=d["id"], url=d["url"], lang=d["lang"], text=d["text"], source="fineweb2",
                                             dump=d["dump"]), ensure_ascii=False) + "\n").encode())
        json.dump(dict(docs=len(new), tokens=sum(d["ntok"] for d in new)), open(tmp / "summary.json", "w"))
        os.rename(tmp, OUT)
        print(f"wrote {len(new):,} docs to {OUT}")
