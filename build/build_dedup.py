"""Training sets for the deduplication comparison: no dedup vs within each snapshot vs across snapshots, same token budget.

  python build/build_dedup.py      (needs train/tokenize_all.py's train/bins/all.bin + all_index.json; set = NAIJAWEB_SET)

All three start from the no-dedup pool N (rule filters + GlotLID), minus every document that shares a 13-word sequence with
a held-out set. Then
  dedup_none    nothing else
  dedup_within  minus the documents data/dedup_filtered/<set>/drop.tsv removes (near-duplicates inside one snapshot)
  dedup_across  minus the documents drop_global.tsv removes (near-duplicates anywhere, one copy kept)
Each version is shuffled (fixed seed) and cut to the token count of the smallest, so no difference is a size effect.
Writes train/bins/dedup_{none,within,across}.bin and prints the language mix of each version (dedup removes text unevenly).
"""
import collections, io, json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config.py
from config import VERSIONS, SET
import numpy as np, zstandard
from grams import grams

ROOT = Path(__file__).resolve().parent.parent
BINS = ROOT / "train/bins"
dd = ROOT / "data/dedup_filtered" / SET
held = set()
for f in sorted((ROOT / "data/heldout").glob("*.jsonl")):
    for line in open(f):
        held |= grams(json.loads(line)["text"])

index = json.load(open(BINS / "all_index.json"))   # document id -> [start, end] token offsets in all.bin
data = np.memmap(BINS / "all.bin", dtype=np.uint16, mode="r")
lang, contaminated = {}, set()
for part in sorted((VERSIONS / f"N_{SET}").glob("part_*.jsonl.zst")):
    text = zstandard.ZstdDecompressor().stream_reader(io.BytesIO(part.read_bytes()), read_across_frames=True).read().decode()
    for line in filter(None, text.split("\n")):
        r = json.loads(line)
        lang[r["id"]] = r.get("lang") or r.get("language")
        if grams(r["text"]) & held: contaminated.add(r["id"])
print(f"{len(contaminated)} documents share a 13-word sequence with a held-out set", flush=True)

def dropped(name): return {":".join(l.split("\t")[:2]) for l in open(dd / name)}
within, across = dropped("drop.tsv"), dropped("drop_global.tsv")
clean = [i for i in index if i not in contaminated]
versions = {"dedup_none": clean,
            "dedup_within": [i for i in clean if i not in within],
            "dedup_across": [i for i in clean if i not in across]}
size = lambda ids: sum(index[i][1] - index[i][0] for i in ids)
budget = min(size(ids) for ids in versions.values())
for name, ids in versions.items():
    random.Random(0).shuffle(ids)
    out, n, mix = BINS / f"{name}.bin", 0, collections.Counter()
    with open(f"{out}.tmp", "wb") as f:
        for i in ids:
            s, e = index[i]
            if n + e - s > budget: break
            data[s:e].tofile(f)
            n += e - s; mix[lang[i]] += e - s
    Path(f"{out}.tmp").rename(out)
    print(f"{name}: {len(ids):,} documents available, {n:,} tokens written (budget {budget:,}); " +
          ", ".join(f"{l} {100 * t / n:.1f}%" for l, t in sorted(mix.items())), flush=True)
