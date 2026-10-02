"""The release (build/release_sizes.py --write) as Hugging Face parquet, one config per language.

  python build/release_parquet.py RELEASE OUT   -> OUT/data/<lang>/train-NNNNN-of-NNNNN.parquet and OUT/manifest.json

Refuses if OUT exists; builds in OUT.tmp and renames it at the end. Fields per document:
  id                         our document id (a Common Crawl id is <filtered shard>:<line>, a re-crawl id <host>:<sha1 of the URL>)
  text, url, domain          the text as released; domain = host without www.
  source                     cc (our Common Crawl pass), recrawl (our crawl of the sites), fineweb2 (the FineWeb-2 fold-in)
  snapshot                   the Common Crawl snapshot: ours for cc, FineWeb-2's dump for fineweb2, none for recrawl
  fetched                    re-crawl fetch time (UTC), else none
  lid_score                  GlotLID probability of the language: our GlotLID pass for cc and recrawl (cc on the WET text),
                             FineWeb-2's language_score for fineweb2
  quality_score              p(keep) of the NaijaWeb quality classifier, which selects the release (kept at > 0.5)
  robots                     re-crawl only: how the host's robots.txt was treated (fetched, absent_4xx, unreachable_assumed_open);
                             where the crawl state did not record it, read from the host's crawl log logs/rescrape/<host>.log,
                             else "not recorded"
The joins are checked: every Common Crawl document's url and text must equal its line in $NAIJAWEB_DATA/wet/filtered_all.
manifest.json: rows, tok32k tokens (end token included, as the release counts them) and sha256 per file and per config.
"""
import glob, hashlib, io, json, os, re, sys, collections
from multiprocessing import Pool
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config.py
from config import DATA
from urllib.parse import urlparse
import pyarrow as pa, pyarrow.parquet as pq, zstandard
from tokenizers import ByteLevelBPETokenizer

ROOT = Path(__file__).resolve().parent.parent
H = DATA / "wet"
RELEASE, OUT = Path(sys.argv[1]), Path(sys.argv[2])
TMP = Path(str(OUT) + ".tmp")
LANGS = ["hau_Latn", "ibo_Latn", "yor_Latn"]
ROWS = 50_000   # per shard
tok = ByteLevelBPETokenizer(str(ROOT / "train/tok32k/vocab.json"), str(ROOT / "train/tok32k/merges.txt"))
SOURCE = {"cc": "cc", "recrawl": "recrawl", "fw2": "fineweb2"}
SCHEMA = pa.schema([("id", pa.string()), ("text", pa.string()), ("url", pa.string()), ("domain", pa.string()), ("source", pa.string()),
                    ("snapshot", pa.string()), ("fetched", pa.string()), ("lid_score", pa.float64()), ("quality_score", pa.float64()),
                    ("robots", pa.string())])   # one schema for every shard, even an all-empty column

def jsonl(path):
    with open(path, "rb") as fh:
        yield from map(json.loads, io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh, read_across_frames=True), encoding="utf-8"))

def domain(url):
    h = urlparse(url).netloc.lower().split(":")[0]
    return h[4:] if h.startswith("www.") else h

def cc_shard(args):
    """(name, {line: (url, text)}) -> {id: (snapshot, lid)}, asserting url and text equal the filtered shard's line."""
    name, want = args
    raw = zstandard.ZstdDecompressor().stream_reader(io.BytesIO((H / "filtered_all" / name).read_bytes()), read_across_frames=True).read()
    lines = [l for l in raw.decode().split("\n") if l]
    out = {}
    for i, (url, text) in want.items():
        r = json.loads(lines[i])
        assert r["url"] == url and r["wet_text"] == text, f"{name}:{i} does not match its filtered shard"
        out[f"{name}:{i}"] = (r["wet_file"].split("/")[1], float(r["lid_prob"]))
    return out

def robots_from_log(host):
    """How robots.txt was treated, from the crawler's per-host counts in logs/rescrape/<host>.log (build/rescrape.py)."""
    f = ROOT / f"logs/rescrape/{host}.log"
    log = f.read_text() if f.exists() else ""
    seen = set(re.findall(r"^\s*(robots_\S+)", log, re.M))
    sitemaps = [int(n) for n in re.findall(r"(\d+) sitemap lines in robots\.txt", log)]   # older runs print only this line
    if "robots_ok" in seen or any(sitemaps): return "fetched"
    if any(x.endswith("_allow_all_per_rfc9309") for x in seen): return "absent_4xx"
    if any(x.endswith("_assumed_open") for x in seen): return "unreachable_assumed_open"
    return "not recorded"

if __name__ == "__main__":
    assert not OUT.exists() and not TMP.exists(), f"{OUT} or {TMP} exists"
    release = json.load(open(RELEASE / "manifest.json"))
    docs = {l: [r for f in sorted(glob.glob(str(RELEASE / l / "*.jsonl.zst"))) for r in jsonl(f)] for l in LANGS}
    assert {l: len(d) for l, d in docs.items()} == release["docs"], "release files do not match its manifest"

    want = collections.defaultdict(dict)
    for d in docs.values():
        for r in d:
            if r["source"] == "cc":
                name, i = r["id"].rsplit(":", 1); want[name][int(i)] = (r["url"], r["text"])
    cc = {}
    with Pool(48) as pool:
        for o in pool.imap_unordered(cc_shard, want.items(), chunksize=64): cc.update(o)
    rec = {r["id"]: (r.get("lid_prob"), r.get("fetched")) for f in glob.glob(str(Path(release["recrawl_build"]) / "*.jsonl.zst")) for r in jsonl(f)}
    fw2_ids = {r["id"][len("fineweb2:"):] for d in docs.values() for r in d if r["source"] == "fw2"}
    fw2 = {}
    for f in glob.glob(str(DATA / "baselines/fineweb2/data/*/train/*.parquet")):
        t = pq.read_table(f, columns=["id", "language_score"]).to_pydict()
        fw2.update((i, s) for i, s in zip(t["id"], t["language_score"]) if i in fw2_ids)

    manifest = {"release": str(RELEASE), "release_code": release["code"], "configs": {}, "files": {}}
    for l in LANGS:
        rows = []
        for r in sorted(docs[l], key=lambda r: r["id"]):
            s = r["source"]
            if s == "cc": snap, lid = cc[r["id"]]; fetched = None
            elif s == "recrawl": snap = None; lid, fetched = rec[r["id"]]
            else: snap, lid, fetched = r["dump"], fw2[r["id"][len("fineweb2:"):]], None
            rows.append(dict(id=r["id"], text=r["text"], url=r["url"], domain=domain(r["url"]), source=SOURCE[s], snapshot=snap,
                             fetched=fetched, lid_score=float(lid), quality_score=r["quality_p"],
                             robots=robots_from_log(r["id"].split(":")[0]) if r.get("robots") == "not recorded" else r.get("robots")))
        (TMP / "data" / l).mkdir(parents=True)
        n = (len(rows) + ROWS - 1) // ROWS
        ntok = 0
        for k in range(n):
            part = rows[k * ROWS:(k + 1) * ROWS]
            ntok += sum(len(ids) + 1 for ids in (e.ids for e in tok.encode_batch([x["text"] for x in part])))
            path = TMP / "data" / l / f"train-{k:05d}-of-{n:05d}.parquet"
            pq.write_table(pa.Table.from_pylist(part, schema=SCHEMA), path, compression="zstd", row_group_size=5_000)
            manifest["files"][f"data/{l}/{path.name}"] = dict(rows=len(part), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        assert ntok == release["tokens"][l], (l, ntok, release["tokens"][l])
        manifest["configs"][l] = dict(rows=len(rows), tokens=ntok, shards=n,
                                      by_source={s: sum(x["source"] == s for x in rows) for s in SOURCE.values()})
        print(f"{l}: {len(rows):,} rows, {ntok:,} tokens, {n} shards", flush=True)
    (TMP / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    os.rename(TMP, OUT)
    print(f"wrote {OUT}")
