#!/bin/bash
# End-to-end smoke test: every pipeline stage, on CPU, over ~275 synthetic/sampled documents
# (smoke/fixture/sampled_text.jsonl -- 75 short public Wikipedia excerpts, hau/ibo/yor, sampled
# once from data/heldout; smoke/make_fixture.py expands them with synthetic boilerplate and
# duplicates). Takes a few minutes. Proves the pipeline runs end to end on this repo's layout; it
# does not, and is not meant to, produce a usable corpus or model.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
PY=.venv/bin/python
command -v "$PY" >/dev/null 2>&1 || PY=python3

export NAIJAWEB_DATA="$(mktemp -d)"
export CUDA_VISIBLE_DEVICES=""   # CPU only, deliberately -- see train/pretrain.py's DEVICE fallback
# Keep this a GOOD CITIZEN on a shared machine: torch defaults to one thread per core, which on a
# big shared box turns "a few smoke-test steps" into dozens of cores at 100%. Cap it.
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCH_NUM_THREADS=4
HELDOUT_BACKUP=""
cleanup() {
  rm -rf "$NAIJAWEB_DATA" "data/wet_paths_SMOKE.txt" "train/bins/SMOKE.bin" \
         "results/curves_pretrain/SMOKE_e1_s1.jsonl" "results/filtered_SMOKE.json" \
         "models/pretrain_SMOKE_e1_s1" 2>/dev/null || true
  rm -rf "data/heldout"
  if [ -n "$HELDOUT_BACKUP" ] && [ -d "$HELDOUT_BACKUP" ]; then
    mv "$HELDOUT_BACKUP" "data/heldout"
  fi
  rmdir train/bins models logs 2>/dev/null || true   # only if this run left them empty
}
trap cleanup EXIT
if [ -d "data/heldout" ]; then
  HELDOUT_BACKUP="$(mktemp -d)/heldout"
  mv "data/heldout" "$HELDOUT_BACKUP"
fi

echo "=== 1/6 filters + boilerplate ==="
$PY smoke/make_fixture.py
$PY filter/apply_filters.py SMOKE
echo

echo "=== 2/6 MinHash dedup ==="
DEDUP_IN="$NAIJAWEB_DATA/wet/filtered/SMOKE" DEDUP_OUT="$NAIJAWEB_DATA/dedup_smoke" DEDUP_PROCS=8 \
  $PY filter/dedup.py sign
DEDUP_OUT="$NAIJAWEB_DATA/dedup_smoke" $PY filter/dedup.py cluster
echo

echo "=== 3/6 decontamination (13-word-sequence check, build/grams.py) ==="
$PY - <<'PYEOF'
import sys
sys.path.insert(0, "build")
from grams import grams
a = "gwamnati jihar kano ta sanar da cewa za ta fara aikin gina sababbin makarantu don yara"
b = "wani abu daban gwamnati jihar kano ta sanar da cewa za ta fara aikin gina sababbin makarantu"
c = "wani rubutu daban wanda babu wata alaka da sauran maganganun da muke magana akai a yau"
shared_ab = grams(a) & grams(b)
shared_ac = grams(a) & grams(c)
print(f"a vs b (overlapping 13-word window): {len(shared_ab)} shared sequence(s) -- would be decontaminated")
print(f"a vs c (unrelated text): {len(shared_ac)} shared sequence(s) -- would be kept")
assert shared_ab and not shared_ac, "decontamination check did not behave as expected"
PYEOF
echo

echo "=== 4/6 tokenize to a bin (the real train/tok32k tokenizer) ==="
$PY - <<'PYEOF'
import json, numpy as np
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer

tok = ByteLevelBPETokenizer("train/tok32k/vocab.json", "train/tok32k/merges.txt")
EOT = 0
# Only a handful of documents -- this feeds train/pretrain.py next, and the point of a smoke test
# is a few optimizer steps in well under a minute, not a real (CPU-bound, many-minute) run.
docs = [json.loads(l)["text"] for l in open("smoke/fixture/sampled_text.jsonl")][:12]
arrs = [np.array(tok.encode(t).ids + [EOT], dtype=np.uint16) for t in docs]
arr = np.concatenate(arrs)
Path("train/bins").mkdir(exist_ok=True)
arr.tofile("train/bins/SMOKE.bin")
print(f"train/bins/SMOKE.bin: {len(docs)} documents, {len(arr)} tokens")
PYEOF
echo

echo "=== 5/6 a tiny data/heldout fixture (pretrain.py reads this relative path directly) ==="
$PY - <<'PYEOF'
import json
from pathlib import Path
lines = [json.loads(l) for l in open("smoke/fixture/sampled_text.jsonl")]
by_lang = {"hau_Latn": "hau", "ibo_Latn": "ibo", "yor_Latn": "yor"}
Path("data/heldout").mkdir(parents=True, exist_ok=True)
for setname in ("exam", "flores", "news", "wiki"):
    for lang, code in by_lang.items():
        rows = [r for r in lines if r["lang"] == lang][:3]
        with open(f"data/heldout/{setname}_{code}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps({"text": r["text"][:200]}, ensure_ascii=False) + "\n")
print("wrote 12 tiny data/heldout/*.jsonl files (3 docs each, from the same sampled text)")
PYEOF
echo

echo "=== 6/6 a few train/pretrain.py steps + held-out bits-per-byte ==="
mkdir -p models logs
$PY train/pretrain.py --version SMOKE --seed 1 --passes 1 --ctx 128 --bs 4 --accum 1
echo "curve written:"
tail -n 1 results/curves_pretrain/SMOKE_e1_s1.jsonl
$PY eval/compare.py 1 SMOKE | head -4
echo

echo "=== SMOKE TEST PASSED ==="
