"""Load the release the way a user would and check it against its manifest.

  python build/release_loadtest.py PARQUET_DIR     -> datasets.load_dataset(PARQUET_DIR, <config>)

For each config (hau_Latn, ibo_Latn, yor_Latn): rows and tok32k tokens (end token included) must equal manifest.json in
PARQUET_DIR, and every document id must be unique. Prints one line per config and exits non-zero on any mismatch.
"""
import json, sys
from pathlib import Path
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

ROOT = Path(__file__).resolve().parent.parent
HF = Path(sys.argv[1])
tok = ByteLevelBPETokenizer(str(ROOT / "train/tok32k/vocab.json"), str(ROOT / "train/tok32k/merges.txt"))
man = json.load(open(HF / "manifest.json"))
bad = 0
for cfg, want in man["configs"].items():
    ds = load_dataset(str(HF), cfg, split="train")
    ntok = sum(len(e.ids) + 1 for i in range(0, len(ds), 10_000) for e in tok.encode_batch(ds[i:i + 10_000]["text"]))
    ok = len(ds) == want["rows"] and ntok == want["tokens"] and len(set(ds["id"])) == len(ds)
    bad += not ok
    print(f"{cfg}: {len(ds):,} rows (manifest {want['rows']:,}), {ntok:,} tokens (manifest {want['tokens']:,}), "
          f"unique ids {len(set(ds['id'])) == len(ds)}, columns {ds.column_names} -> {'OK' if ok else 'MISMATCH'}")
sys.exit(1 if bad else 0)
