"""Our corpus as the release recipe builds it (no site cap, no language quota), per language and per source.

  python build/release_sizes.py <re-crawl build dir>                 -> printed (keep it as the build log)
  python build/release_sizes.py <re-crawl build dir> --write OUT     -> also writes the release to OUT

Same steps as the "ours" half of build/build_baselines.py, which it imports (P_ALL30 + re-crawl + FineWeb-2 fold-in,
fold-in documents sharing a MinHash band key with the pool dropped, 13-word and Wikipedia decontamination, our classifier's p > 0.5),
then counts documents and tokens (tok32k) per language and per source. build_baselines.py's corpus_ours is a token-budget cut of this.
Also printed: the same counts after each step (the per-source funnel), and for the released corpus the Bible and Wikipedia
token shares and the top-10 sites per language (Bible sites: jw.org or a host containing bible, biblica or beblia; Wikipedia: a host ending in wikipedia.org).

--write OUT writes every kept document, with its text as found, to OUT/<lang>/<source>_<input file>.jsonl.zst
(source cc, recrawl or fw2), adding quality_p and, for the re-crawl, robots: how the host's robots.txt was treated
("fetched", "absent_4xx", "unreachable_assumed_open", or "not recorded" for hosts crawled before the crawler logged it).
Builds in OUT.tmp, refuses if OUT or OUT.tmp exists, and before renaming re-reads what it wrote: the documents and
tokens per language must equal the counts above and no written text may share a 13-word sequence with a held-out set.
manifest.json in OUT holds those counts and the inputs.
"""
import glob, io, json, os, sys, collections, importlib.util, subprocess
from multiprocessing import Pool
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config.py
from config import DATA

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("bb", ROOT / "build/build_baselines.py")
bb = importlib.util.module_from_spec(spec); sys.modules["bb"] = bb; spec.loader.exec_module(bb)

W = {}   # keep, cls, robots, tmp: set before the write Pool forks, so workers inherit them instead of receiving copies

def write_file(path):
    """Write the kept documents of one input file; return (lang, docs, tokens, held-out hits) per language written."""
    import zstandard
    keep, cls, robots, tmp = W["keep"], W["cls"], W["robots"], W["tmp"]
    outs, stats = {}, collections.defaultdict(lambda: [0, 0, 0])
    with open(path, "rb") as fh:
        for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh, read_across_frames=True), encoding="utf-8"):
            r = json.loads(line)
            part = keep.get(r["id"])
            if part is None: continue
            rec = dict(id=r["id"], url=r["url"], lang=r["lang"], source=part, text=r["text"], quality_p=round(cls[r["id"]], 4))
            if part == "recrawl": rec["robots"] = robots.get(r["host"], "not recorded")
            if part == "fw2": rec["dump"] = r.get("dump")
            l = r["lang"]
            if l not in outs:
                outs[l] = zstandard.ZstdCompressor(level=10).stream_writer(open(tmp / l / f"{part}_{Path(path).name}", "wb"))
            outs[l].write((json.dumps(rec, ensure_ascii=False) + "\n").encode())
            st = stats[l]; st[0] += 1; st[1] += len(bb.tok.encode(r["text"]).ids) + 1
            st[2] += any(bb.grams(r["text"]) & h for h in bb.held.values())
    for w in outs.values(): w.close()
    return [(l, *v) for l, v in stats.items()]

if __name__ == "__main__":
    V = bb.V
    files = (sorted(glob.glob(str(V / "P_ALL30/part_*.jsonl.zst"))) + sorted(glob.glob(str(bb.R / "*.jsonl.zst"))) +
             sorted(glob.glob(str(V / "F_fineweb2/part_*.jsonl.zst"))))
    with Pool(48) as pool:
        ds = [d for o in pool.imap_unordered(bb.ours_file, files) for d in o]
    def funnel(step, ds):
        for l in bb.LANGS:
            for p in ("cc", "recrawl", "fw2"): bb.line(f"funnel | {step} | {l} {p}", [d for d in ds if d["lang"] == l and d["part"] == p])
    funnel("loaded, one copy across snapshots", ds)
    seen = {k for d in ds if d["part"] != "fw2" for k in d["keys"]}
    ds = [d for d in ds if d["part"] != "fw2" or not seen.intersection(d["keys"])]; del seen
    funnel("fold-in sharing a band key with the pool dropped", ds)
    ds = bb.decontaminate("ours", ds)
    funnel("decontaminated", ds)
    cls = {}
    for f in glob.glob(str(V / "scores/cls/*part_*.jsonl")) + glob.glob(str(V / "scores/cls/rescrape_*.jsonl")):
        cls.update(json.load(open(f)))
    cls.update(json.load(open(V / "scores_wikistrip/cls/rescrape_wikistrip.jsonl")))
    cls.update(json.load(open(V / "scores_wikistrip30/cls/rescrape_wikistrip.jsonl")))
    kept = [d for d in ds if d["id"] in cls and cls[d["id"]] > 0.5]
    funnel("our classifier's p > 0.5", kept)
    print(f"\nRelease recipe (no site cap, no language quota), our classifier's p > 0.5:")
    bb.line("all", kept)
    for l in bb.LANGS:
        dl = [d for d in kept if d["lang"] == l]
        bb.line(f"{l}", dl)
        for p in ("cc", "recrawl", "fw2"): bb.line(f"{l} {p}", [d for d in dl if d["part"] == p])
        print(f"{l} sites {len({d['site'] for d in dl}):,}", flush=True)
    print(f"sites {len({d['site'] for d in kept}):,}")
    bible = lambda s: s == "jw.org" or s.endswith(".jw.org") or any(w in s for w in ("bible", "biblica", "beblia"))
    for l in bb.LANGS + ["all"]:
        dl = [d for d in kept if l == "all" or d["lang"] == l]; t = bb.tokens(dl)
        by = collections.Counter()
        for d in dl: by[d["site"]] += d["n"]
        top = by.most_common(10)
        print(f"profile | {l} | Bible sites {100 * sum(k for s, k in by.items() if bible(s)) / t:.1f}% | Wikipedia "
              f"{100 * sum(k for s, k in by.items() if s.endswith('wikipedia.org')) / t:.1f}% | top-10 sites {100 * sum(k for _, k in top) / t:.1f}% | "
              + ", ".join(f"{s} {100 * k / t:.1f}" for s, k in top))

    if "--write" in sys.argv:
        OUT = Path(sys.argv[sys.argv.index("--write") + 1]); TMP = Path(str(OUT) + ".tmp")
        assert not OUT.exists() and not TMP.exists(), f"{OUT} or {TMP} exists"
        keep = {d["id"]: d["part"] for d in kept}
        robots = {Path(f).name[:-len(".state.json")]: json.load(open(f)).get("robots", "not recorded")
                  for f in glob.glob(str(DATA / "wet/rescrape/*.state.json"))}
        for l in bb.LANGS: (TMP / l).mkdir(parents=True)
        W.update(keep=keep, cls=cls, robots=robots, tmp=TMP)
        with Pool(48) as pool:
            got = [c for o in pool.imap_unordered(write_file, files) for c in o]
        written = collections.Counter(); ntok = collections.Counter(); dirty = 0
        for l, n, t, h in got: written[l] += n; ntok[l] += t; dirty += h
        for l in bb.LANGS:
            dl = [d for d in kept if d["lang"] == l]
            assert (written[l], ntok[l]) == (len(dl), bb.tokens(dl)), (l, written[l], ntok[l], len(dl), bb.tokens(dl))
        assert dirty == 0, f"{dirty} written documents share a 13-word sequence with a held-out set"
        (TMP / "manifest.json").write_text(json.dumps(dict(
            docs=dict(written), tokens=dict(ntok), inputs=[str(Path(f)).replace(str(DATA), "$NAIJAWEB_DATA") for f in files],
            recrawl_build=str(bb.R), code=subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()),
            indent=1) + "\n")
        os.rename(TMP, OUT)
        print(f"wrote {sum(written.values()):,} documents, {sum(ntok.values()):,} tokens to {OUT}; re-read: counts match, 0 held-out overlaps")
