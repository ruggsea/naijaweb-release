"""Pool versions of one crawl set: our rule filters + GlotLID, with duplicates removed within each crawl, across all of
them, or not at all.

  python filter/build_a.py none       -> versions/N_<set>, no dedup at all
  python filter/build_a.py percrawl   -> versions/P_<set>, minus drop.tsv        (near-duplicates inside one snapshot)
  python filter/build_a.py global     -> versions/A_<set>, minus drop_global.tsv (one document per MinHash cluster overall)

<set> is NAIJAWEB_SET (default ALL30), the set named when running filter/dedup_across.sh. P_ALL30 is the base pool the
corpus builds start from (build/build_matched.py and build/build_baselines.py remove across-snapshot duplicates themselves
from drop_global.tsv). N is the master store: P and A are subsets of it by document id.

Reads the shards listed in data/dedup_filtered/<set>/shards.txt from the symlink folder $NAIJAWEB_DATA/wet/filtered_all
and drops every (shard, line) in the mode's drop list. The shard list comes from the set, not from a glob of the
folder, which holds every crawl and would read shards the drop list never saw.

Writes $NAIJAWEB_DATA/wet/versions/<letter>_<set>/part_NNNN.jsonl.zst, records {id, url, lang, text},
id = "<shard>:<line_no>". Atomic per part, skipped if present, so a rerun resumes.
"""
import io
import json, os, sys
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config.py
from config import DATA, SET
import zstandard

IN = DATA / "wet/filtered_all"
MODE = sys.argv[1] if len(sys.argv) > 1 else "global"
DD = Path(__file__).resolve().parent.parent / f"data/dedup_filtered/{SET}"
DROP = {"none": None, "percrawl": DD / "drop.tsv", "global": DD / "drop_global.tsv"}[MODE]
LETTER = {'none': 'N', 'percrawl': 'P', 'global': 'A'}[MODE]
OUT = DATA / "wet/versions" / f"{LETTER}_{SET}"
PARTS = 64

drop = set()
for line in (open(DROP) if DROP else []):
    shard, line_no, _, _ = line.rstrip("\n").split("\t")
    drop.add((shard, int(line_no)))
print(f"{len(drop)} documents to drop", flush=True)

shards = sorted((DD / "shards.txt").read_text().split())
missing = [n for n in shards if not (IN / n).exists()]
assert not missing, f"{len(missing)} shards of {SET} are not in {IN}, e.g. {missing[:3]}"
print(f"set {SET}: {len(shards)} shards -> {OUT}", flush=True)
OUT.mkdir(parents=True, exist_ok=True)
dctx, cctx = zstandard.ZstdDecompressor(), zstandard.ZstdCompressor(level=6)
kept = seen = 0
for part in range(PARTS):
    dst = OUT / f"part_{part:04d}.jsonl.zst"
    if dst.exists():
        continue
    buf = []
    for name in shards[part::PARTS]:
        raw = dctx.stream_reader(io.BytesIO((IN / name).read_bytes()), read_across_frames=True).read()
        for line_no, line in enumerate(l for l in raw.decode().split("\n") if l):
            seen += 1
            if (name, line_no) in drop:
                continue
            r = json.loads(line)
            buf.append(json.dumps({"id": f"{name}:{line_no}", "url": r["url"],
                                   "lang": r["lang"], "text": r["wet_text"]}))
            kept += 1
    tmp = dst.with_suffix(".tmp")
    tmp.write_bytes(cctx.compress(("\n".join(buf) + "\n").encode()))
    os.replace(tmp, dst)
    print(f"part {part}: {len(buf)} docs", flush=True)
print(f"read {seen}, kept {kept}", flush=True)
