"""Compare corpus versions by held-out bits per byte, from the curves train/pretrain.py wrote.

  python eval/compare.py <passes> <version> <version> [...]
  python eval/compare.py 10 corpus_ours corpus_wura corpus_fw2 corpus_madlad corpus_madlad_clean corpus_hplt

Reads results/curves_pretrain/<version>_e<passes>_s<seed>.jsonl for every seed, at the final step (lower is better).
Prints per held-out set the mean [min-max] over seeds of each version, the mean over the sets, and for every pair of versions
how many sets it WINS, LOSSES or cannot tell: A beats B on a set only when every seed of A is below every seed of B.
"""
import glob, itertools, json, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
passes, versions = sys.argv[1], sys.argv[2:]

def final_rows(v):
    files = sorted(glob.glob(str(ROOT / f"results/curves_pretrain/{v}_e{passes}_s*.jsonl")))
    assert files, f"no curves for {v} at {passes} passes"
    last = [[json.loads(l) for l in open(f)][-1] for f in files]
    assert len({r["step"] for r in last}) == 1, f"{v}: runs end at different steps"
    return [r["bits_per_byte"] for r in last]

def verdict(x, y): return "WIN" if max(x) < min(y) else "LOSS" if min(x) > max(y) else "cannot tell"
def mm(a): return f"{st.mean(a):.4f} [{min(a):.4f}-{max(a):.4f}]"

final = {v: final_rows(v) for v in versions}
sets = sorted(final[versions[0]][0])
print(f"{passes} passes, final step; bits per byte, mean [min-max] over seeds (n = " + ", ".join(f"{v} {len(final[v])}" for v in versions) + ")")
print("| held-out set | " + " | ".join(versions) + " |")
for k in sets:
    print(f"| {k} | " + " | ".join(mm([r[k] for r in final[v]]) for v in versions) + " |")
print(f"| mean of the {len(sets)} sets | " + " | ".join(mm([st.mean(r.values()) for r in final[v]]) for v in versions) + " |")
for a, b in itertools.combinations(versions, 2):
    n = {"WIN": 0, "LOSS": 0, "cannot tell": 0}
    for k in sets: n[verdict([r[k] for r in final[a]], [r[k] for r in final[b]])] += 1
    print(f"{a} vs {b}: {n['WIN']} win, {n['LOSS']} loss, {n['cannot tell']} cannot tell (of {len(sets)} sets)")
