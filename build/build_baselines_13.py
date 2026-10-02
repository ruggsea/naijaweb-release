"""Two more comparison corpora (MADLAD-400, HPLT 2.0), built with build_baselines.py's steps and cut to its budget.

  python build/build_baselines_13.py <re-crawl build dir> [--write] [--only <name>]   (the dir is only passed through to build_baselines.py)

  madlad  MADLAD-400 noisy, ha/ig/yo, as published (text only: no URL, so no site list and no Wikipedia identity hold-out;
          the 13-word rule still applies). MADLAD stores newlines escaped as the two characters \\n; they are unescaped.
  hplt    HPLT 2.0 cleaned, hau_Latn/ibo_Latn/yor_Latn, every row as published; URL = column u.
  madlad_clean  MADLAD-400 clean, the same way as madlad (it is above the budget after deduplication).
Steps, as in build/build_baselines.py: one document per MinHash cluster inside the corpus; drop documents sharing a 13-word
sequence with a held-out set (per-set counts printed) and held-out Wikipedia articles; then cut at random (same seed as build_baselines.py) to
the same budget, the unique tokens of train/bins/corpus_wura.bin. No language quota, no site cap.
--write writes train/bins/corpus_{madlad,hplt,madlad_clean}.bin and data/matched/<name>.ids.
"""
import gzip, json, sys, importlib.util
from multiprocessing import Pool
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config.py
from config import DATA
import numpy as np, pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("bb", ROOT / "build/build_baselines.py")
bb = importlib.util.module_from_spec(spec); sys.modules["bb"] = bb; spec.loader.exec_module(bb)
B = DATA / "baselines"
NAMES = {"madlad": "corpus_madlad", "hplt": "corpus_hplt", "madlad_clean": "corpus_madlad_clean"}
BUDGET = (ROOT / "train/bins/corpus_wura.bin").stat().st_size // 2
assert BUDGET == 290_447_647, BUDGET

def args(name):
    if name.startswith("madlad"):
        kind, prefix = ("clean", "madlad-clean") if name == "madlad_clean" else ("noisy", "madlad")
        for code, lang in (("ha", "hau_Latn"), ("ig", "ibo_Latn"), ("yo", "yor_Latn")):
            for i, l in enumerate(gzip.open(B / f"madlad/data/{code}/{code}_{kind}_0000.jsonl.gz", "rt", encoding="utf-8")):
                yield (f"{prefix}:{code}:{i}", lang, "", json.loads(l)["text"].replace("\\n", "\n"))
    else:
        for lang in bb.LANGS:
            for f in sorted((B / f"hplt/{lang}").glob("*.parquet")):
                for batch in pq.ParquetFile(f).iter_batches(batch_size=10_000, columns=["id", "u", "text"]):
                    t = batch.to_pydict()
                    yield from ((f"hplt:{i}", lang, u, x) for i, u, x in zip(t["id"], t["u"], t["text"]))

if __name__ == "__main__":
    print(f"Budget: build_baselines.py's, {BUDGET:,} unique tokens (train/bins/corpus_wura.bin)")
    for v in [sys.argv[sys.argv.index("--only") + 1]] if "--only" in sys.argv else NAMES:
        with Pool(64) as pool:
            ds = [dict(d, part=v) for d in pool.imap(bb.base, args(v), chunksize=500)]
        bb.line(f"{v}: as published", ds)
        ds = bb.dedup(ds)
        bb.line(f"{v}: one per MinHash cluster", ds)
        ds = bb.decontaminate(v, ds)
        assert bb.tokens(ds) >= BUDGET, f"{v} is below the budget"
        final = bb.fill(ds, BUDGET, np.random.RandomState(bb.SEED + 1))
        print()
        for label, x in (("before the cut", ds), ("final version", final)):
            n = bb.tokens(x)
            print(f"{v} {label}: {len(x):,} docs {n:,} unique tokens; " +
                  ", ".join(f"{l} {100 * bb.tokens([d for d in x if d['lang'] == l]) / n:.1f}%" for l in bb.LANGS))
            top = bb.collections.Counter()
            for d in x: top[d["site"]] += d["n"]
            print(f"   {len(top):,} sites; biggest: " + ", ".join(f"{s or '(no URL)'} {100 * t / n:.1f}%" for s, t in top.most_common(10)))
            for l in bb.LANGS:
                dl = [d for d in x if d["lang"] == l]; tl = bb.collections.Counter()
                for d in dl: tl[d["site"]] += d["n"]
                print(f"   {l}: {bb.tokens(dl):,} tokens; biggest sites " +
                      ", ".join(f"{s or '(no URL)'} {100 * t / max(1, bb.tokens(dl)):.1f}%" for s, t in tl.most_common(5)))
        if "--write" in sys.argv:
            out = ROOT / f"train/bins/{NAMES[v]}.bin"
            x = [final[i] for i in np.random.RandomState(bb.SEED + 2).permutation(len(final))]
            ids = "".join(d["id"] + "\n" for d in x)
            if out.exists():
                assert (ROOT / f"data/matched/{NAMES[v]}.ids").read_text() == ids, f"{out} exists and this build differs"
                print(f"{out} exists, same documents in the same order: kept")
            else:
                np.concatenate([d["ids"] for d in x]).tofile(str(out) + ".tmp")
                Path(str(out) + ".tmp").rename(out)
                (ROOT / f"data/matched/{NAMES[v]}.ids").write_text(ids)
                print(f"wrote {out} ({bb.tokens(x):,} tokens)")
        del ds, final
