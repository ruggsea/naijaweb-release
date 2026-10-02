"""Tokenize the no-dedup pool N_<set> once, with the shared 32k tokenizer.

Writes train/bins/all.bin (uint16 token ids, documents concatenated, each ended by
<|endoftext|>) and train/bins/all_index.json ({id: [start, end]}). The deduplicated versions are subsets
of N, so their training files are cut from this one store and no text is tokenized twice.
"""
import io
import json, os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config.py
from config import DATA, SET
from multiprocessing import Pool
import numpy as np, zstandard
from tokenizers import ByteLevelBPETokenizer

ROOT = Path(__file__).resolve().parent.parent
A = DATA / f"wet/versions/N_{SET}"   # the no-dedup pool; every version is a subset of it
OUT = ROOT / "train/bins"
OUT.mkdir(parents=True, exist_ok=True)
TOKDIR = ROOT / "train/tok32k"
EOT = 0  # <|endoftext|> is the first special token

def encode_part(src):
    tok = ByteLevelBPETokenizer(str(TOKDIR / "vocab.json"), str(TOKDIR / "merges.txt"))
    dctx = zstandard.ZstdDecompressor()
    ids, arrs = [], []
    for line in dctx.stream_reader(io.BytesIO(Path(src).read_bytes()), read_across_frames=True).read().decode().split("\n"):
        if not line:
            continue
        r = json.loads(line)
        a = np.array(tok.encode(r["text"]).ids + [EOT], dtype=np.uint16)
        ids.append(r["id"]); arrs.append(a)
    return Path(src).name, ids, np.concatenate(arrs)

if __name__ == "__main__":
    parts = sorted(str(p) for p in A.glob("part_*.jsonl.zst"))
    index, pos = {}, 0
    with open(OUT / "all.bin.tmp", "wb") as out, Pool(24) as pool:
        for name, ids, arr in pool.imap(encode_part, parts):
            start = pos
            lens = []
            # per-document offsets: recover them by scanning for the EOT separator
            ends = np.flatnonzero(arr == EOT) + 1
            prev = 0
            for doc_id, e in zip(ids, ends):
                index[doc_id] = [pos + prev, pos + int(e)]
                prev = int(e)
            arr.tofile(out)
            pos += len(arr)
            print(f"{name}: {len(ids)} docs, {pos} tokens", flush=True)
    os.replace(OUT / "all.bin.tmp", OUT / "all.bin")
    (OUT / "all_index.json").write_text(json.dumps(index))
    print(f"total {pos} tokens over {len(index)} documents")
