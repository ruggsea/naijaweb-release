# train/

The tokenizer, the training-file ("bin") builder, and the pretraining script.

- `tokenizer.py` — trains the one byte-level BPE tokenizer (32k vocab) used everywhere, on a sample
  of the no-dedup pool. Writes to `tok32k/` (tracked in git — this is a small, load-bearing
  artifact, not regenerable-and-forgotten: every `.bin` file and every classifier score depends on
  it being exactly this tokenizer).
- `tok32k/` — the tokenizer itself (`vocab.json`, `merges.txt`). Checked in.
- `tokenize_all.py` — tokenizes the full no-dedup store once with that tokenizer; every later
  version's `.bin` is cut from this.
- `pretrain.py` — the model: a 12-layer/768-hidden/12-head decoder (85.0M params without
  embeddings, 134.1M with) over the 32k tokenizer. `python train/pretrain.py --version <bin>
  --seed <n> --passes {1,10}`.

Bins themselves (`train/bins/*.bin`) are not tracked in git — they are cut from
`$NAIJAWEB_DATA/wet/versions/N_<set>` by the builder scripts in `build/`, and are regenerable.
