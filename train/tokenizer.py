"""One tokenizer for every corpus version, so token counts mean the same thing.

Byte-level BPE, 32k, trained on a sample of the no-dedup pool N_<set>.
Writes train/tok32k/.
"""
import io
import json, random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config.py
from config import DATA, SET
import zstandard
from tokenizers import ByteLevelBPETokenizer

A = DATA / f"wet/versions/N_{SET}"
OUT = Path(__file__).resolve().parent / "tok32k"
OUT.mkdir(exist_ok=True)
dctx = zstandard.ZstdDecompressor()
sample = OUT / "sample.txt"
if not sample.exists():
    random.seed(0)
    with open(sample.with_suffix(".tmp"), "w") as f:
        for src in sorted(A.glob("part_*.jsonl.zst"))[:8]:      # 1/8 of the corpus
            for line in dctx.stream_reader(io.BytesIO(src.read_bytes()), read_across_frames=True).read().decode().split("\n"):
                if line:
                    f.write(json.loads(line)["text"].replace("\n", " ") + "\n")
    sample.with_suffix(".tmp").rename(sample)
tok = ByteLevelBPETokenizer()
tok.train([str(sample)], vocab_size=32000, min_frequency=2,
          special_tokens=["<|endoftext|>", "<|pad|>"])
tok.save_model(str(OUT))
print("vocab", tok.get_vocab_size())
