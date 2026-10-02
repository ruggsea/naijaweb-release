"""Build the re-crawl add-on: filter every finished $NAIJAWEB_DATA/wet/rescrape shard and write the survivors, one
.jsonl.zst per host, to OUT (default $NAIJAWEB_DATA/wet/versions/R_recrawl). Prints docs and tokens (tok32k)
left after each step. Never writes to $NAIJAWEB_DATA/wet/rescrape; builds in OUT.tmp and renames it at the end, and
refuses to run if OUT exists. --dry prints the counts and writes nothing.
  1. drop docs whose URL is in Common Crawl (P_ALL30) or whose text is a MinHash near-dup of ALL30
  2. exact repeats inside one host: keep one copy, the smallest normalised URL
  3. drop garbled docs (>=3 marks, mojibake_scan.SIG); on lightofislam.com.ng strip the madrasa ad,
     donation-account and copyright lines, and drop the page if under 200 characters remain
  4. MinHash near-dups inside the re-crawl (docs sharing any band key are one cluster): keep one per
     cluster, the smallest normalised URL. The MinHash is taken on the page minus lines that appear on
     20+ pages of its host (menus, sidebars), so a shared sidebar does not merge different articles;
     pages with under 20 words left are template-only and keep the full-text MinHash
  5. drop listing pages: URL contains /tag/, /category/, /author/ or /page/
Then the classifiers' view of the step-5 survivors: how many ours and our classifier's keep at p>0.5, from the
rescrape_<host>.jsonl score files ({id: p}); docs with no score are listed by host."""
import glob, hashlib, os, sys, importlib.util, io, json, re, collections, zstandard
from multiprocessing import Pool
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA
ROOT = str(Path(__file__).resolve().parent.parent)
DRY = "--dry" in sys.argv
OUT = next((a for a in sys.argv[1:] if a != "--dry"), str(DATA / "wet/versions/R_recrawl"))
CAP = 0.20  # share of the kept build above which a host's capped alternative is printed
assert DRY or not os.path.exists(OUT) and not os.path.exists(OUT + ".tmp"), f"{OUT} or {OUT}.tmp exists"
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m
dd = load("dd", f"{ROOT}/filter/dedup.py")
SIG = re.compile("Æ[\u0099™˜\u0098]|É[\u0097—“\u0093]|Æ´|Ã[\u0080-¿]|á[»º¸¹][\u0080-¿]|â€")  # = mojibake_scan.SIG
BANNER = ("AL-HUDA ONLINE MADRASA", "Koyi Alqur’ani, Tajweed da Larabci", "DANNA NAN KA SHIGA", "Bada gudunmawa domin Yada Addinin Musulunci", "© 2026 · The Light Of Islam")
tok = ByteLevelBPETokenizer(f"{ROOT}/train/tok32k/vocab.json", f"{ROOT}/train/tok32k/merges.txt")

def norm(u): return re.sub(r"^https?://(www\.)?", "", u.split("#")[0]).rstrip("/").lower()
def recs(p):
    with open(p, "rb") as f:
        for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(f, read_across_frames=True), encoding="utf-8"):
            yield json.loads(line)

cc = set()
for f in sorted(glob.glob(f"{ROOT}/data/dedup_filtered/ALL30/sig_*.jsonl")):
    for l in open(f):
        cc.update(k * 16 + b for b, k in enumerate(json.loads(l)[5]))
ccurl = {norm(r["url"]) for p in glob.glob(str(DATA / "wet/versions/P_ALL30/*.jsonl.zst")) for r in recs(p)}

def work(f):
    out = []
    for r in recs(f):
        keys = dd.band_keys(dd.signature(r["text"]))
        text = r["text"]
        if r["host"] == "lightofislam.com.ng":
            text = "\n".join(l for l in text.split("\n") if not l.strip().startswith(BANNER))
        out.append(dict(rec={**r, "text": text}, id=r["id"], host=r["host"], url=norm(r["url"]), md5=hashlib.md5(r["text"].encode()).hexdigest(),
                        in_cc=norm(r["url"]) in ccurl or any(k * 16 + b in cc for b, k in enumerate(keys)),
                        keys=[k * 16 + b for b, k in enumerate(keys)], garbled=len(SIG.findall(r["text"])) >= 3,
                        banner_only=r["host"] == "lightofislam.com.ng" and len(text) < 200,
                        ntok=len(tok.encode(text).ids)))
    return out

files = [f for f in glob.glob(str(DATA / "wet/rescrape") + "/**/*.jsonl.zst", recursive=True) if "dropped_by_decision" not in f]
with Pool(24) as pool: docs = [d for o in pool.imap_unordered(work, sorted(files)) for d in o]
def report(label, ds): print(f"{label:52} {len(ds):>8,} docs {sum(d['ntok'] for d in ds):>12,} tokens", flush=True)
report("0. all finished re-crawl docs", docs)
step1 = [d for d in docs if not d["in_cc"]]
report("1. not in Common Crawl (URL or near-dup)", step1)
best = {}
for d in step1:
    k = (d["host"], d["md5"])
    if k not in best or d["url"] < best[k]["url"]: best[k] = d
step2 = list(best.values())
report("2. one copy per host of exact repeats", step2)
step3 = [d for d in step2 if not d["garbled"] and not d["banner_only"]]
report("3. minus garbled and lightofislam banner-only", step3)
print(f"   step 3 removed: garbled {sum(d['garbled'] for d in step2):,}, banner-only {sum(d['banner_only'] and not d['garbled'] for d in step2):,}")
lines = collections.defaultdict(collections.Counter)
for d in step3:
    for l in set(d["rec"]["text"].split("\n")): lines[d["host"]][l.strip()] += 1
def body(d): return "\n".join(l for l in d["rec"]["text"].split("\n") if lines[d["host"]][l.strip()] < 20)
def body_keys(t): return [k * 16 + b for b, k in enumerate(dd.band_keys(dd.signature(t)))]
bodies = [body(d) for d in step3]
with Pool(24) as pool: bkeys = pool.map(body_keys, bodies, chunksize=500)
for d, t, k in zip(step3, bodies, bkeys):
    if len(re.findall(r"\w+", t)) >= 20: d["keys"] = k
parent = list(range(len(step3)))
def find(i):
    while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
    return i
owner = {}
for i, d in enumerate(step3):
    for k in d["keys"]:
        j = owner.setdefault(k, i)
        if j != i: parent[find(i)] = find(j)
keep = {}
for i, d in enumerate(step3):
    r = find(i)
    if r not in keep or d["url"] < step3[keep[r]]["url"]: keep[r] = i
step4 = [step3[i] for i in sorted(keep.values())]
report("4. one per MinHash cluster inside the re-crawl", step4)
gone = collections.Counter(d["host"] for i, d in enumerate(step3) if keep[find(i)] != i)
print("   step 4 removed by host:", gone.most_common(8))
print("   biggest clusters:", sorted(collections.Counter(find(i) for i in range(len(step3))).values())[-5:])
print("   tokens by host after step 4:", collections.Counter(
    {h: sum(d["ntok"] for d in step4 if d["host"] == h) for h in {d["host"] for d in step4}}).most_common(10))

LISTING = re.compile(r"/(tag|category|author|page)/")
step5 = [d for d in step4 if not LISTING.search(d["url"])]
report("5. minus listing-page URLs", step5)
print("   step 5 removed by host:", collections.Counter(d["host"] for d in step4 if LISTING.search(d["url"])).most_common(8))

if not DRY:
    os.makedirs(OUT + ".tmp")
    by_host = collections.defaultdict(list)
    for d in step5: by_host[d["host"]].append(d["rec"])
    for h, rs in sorted(by_host.items()):
        with open(f"{OUT}.tmp/{h}.jsonl.zst", "wb") as f, zstandard.ZstdCompressor().stream_writer(f) as w:
            for r in sorted(rs, key=lambda r: r["id"]): w.write((json.dumps(r, ensure_ascii=False) + "\n").encode())
    json.dump(dict(docs=len(step5), tokens=sum(d["ntok"] for d in step5), hosts={h: len(rs) for h, rs in by_host.items()}),
              open(f"{OUT}.tmp/summary.json", "w"), indent=1)
    os.rename(OUT + ".tmp", OUT)
    print(f"wrote {len(step5):,} docs in {len(by_host)} hosts to {OUT}")

SCORES = str(DATA / "wet/versions/scores")
for who in ("ours", "cls"):
    p = {}
    for f in glob.glob(f"{SCORES}/{who}/rescrape_*.jsonl"): p.update(json.load(open(f)))
    scored = [d for d in step5 if d["id"] in p]
    report(f"6. {who}: step-5 docs with a score", scored)
    report(f"6. {who}: kept at p>0.5", [d for d in scored if p[d["id"]] > 0.5])
    print(f"   {who} kept tokens by host:", collections.Counter(
        {h: sum(d["ntok"] for d in scored if d["host"] == h and p[d["id"]] > 0.5) for h in {d["host"] for d in scored}}).most_common(10))
    unscored = collections.Counter(d["host"] for d in step5 if d["id"] not in p)
    print(f"   {who}: step-5 docs with no score, by host:", unscored.most_common())
    kept = [d for d in scored if p[d["id"]] > 0.5]
    total = sum(d["ntok"] for d in kept)
    by_host = collections.defaultdict(list)
    for d in kept: by_host[d["host"]].append(d)
    print(f"   {who}: kept tokens by host, every host with 1% or more ({len(by_host)} hosts kept in all)")
    for h, ds in sorted(by_host.items(), key=lambda x: -sum(d["ntok"] for d in x[1])):
        n = sum(d["ntok"] for d in ds)
        if n >= total / 100: print(f"     {h:36} {len(ds):>7,} docs {n:>11,} tokens {100 * n / total:5.1f}%")
    # The alternative to shipping a host at its natural share: cap it at CAP of the whole build, keeping
    # its highest-scored documents. A capped host holds x tokens with x = CAP * (rest + x).
    for h, ds in by_host.items():
        n = sum(d["ntok"] for d in ds)
        if n <= CAP * total: continue
        allow, rest = CAP / (1 - CAP) * (total - n), total - n
        ranked, got, i = sorted(ds, key=lambda d: -p[d["id"]]), 0, 0
        while i < len(ranked) and got + ranked[i]["ntok"] <= allow: got += ranked[i]["ntok"]; i += 1
        take, lost = ranked[:i], ranked[i:]
        print(f"   {who}: {h} capped at {CAP:.0%}: keeps {len(take):,} of {len(ds):,} docs, {got:,} of {n:,} tokens; "
              f"build {len(kept) - len(lost):,} docs {rest + got:,} tokens (was {len(kept):,} / {total:,}); "
              f"the {len(lost):,} dropped docs score p {min(p[d['id']] for d in lost):.3f}-{max(p[d['id']] for d in lost):.3f}")
