# eval/

- `compare.py` — reads the curves `train/pretrain.py` wrote (`results/curves_pretrain/<version>_e<passes>_s<seed>.jsonl`)
  and prints, at the final step, bits per byte per held-out set (mean and range over seeds) and a win/loss/tie
  count for every pair of versions. A win needs every seed of one version below every seed of the other.
