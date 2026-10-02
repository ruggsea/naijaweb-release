#!/bin/bash
# Download the external comparison corpora (our corpus vs these, matched on total
# tokens). Resumable: curl -C - / huggingface-cli download both skip what's already complete.
#
#   FineWeb-2 (HuggingFaceFW/fineweb-2)         -> $NAIJAWEB_DATA/baselines/fineweb2/
#   WURA (castorini/wura, documents-v1.0)       -> $NAIJAWEB_DATA/baselines/wura/
#   MADLAD-400 clean+noisy (allenai/madlad-400) -> $NAIJAWEB_DATA/baselines/madlad/
#   HPLT 2.0 cleaned (HPLT/HPLT2.0_cleaned)     -> $NAIJAWEB_DATA/baselines/hplt/
#
# File layout matches what build/fineweb2_pool.py, build/build_baselines.py,
# and build/build_baselines_13.py read; don't rename anything under here.
set -euo pipefail
: "${NAIJAWEB_DATA:?set NAIJAWEB_DATA}"
cd "$(dirname "$0")/.."   # repo root, for build/fineweb2_files.txt
B="$NAIJAWEB_DATA/baselines"
mkdir -p "$B"
cd "$B"

echo "=== FineWeb-2 (hau_Latn/ibo_Latn/yor_Latn train) ==="
while read -r p; do
  mkdir -p "fineweb2/$(dirname "$p")"
  curl -sfL -C - -o "fineweb2/$p.part" "https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/resolve/main/$p" \
    && mv "fineweb2/$p.part" "fineweb2/$p"
  echo "fineweb2 $p exit $?"
done < "$OLDPWD/build/fineweb2_files.txt"

echo "=== WURA documents-v1.0 (train + eval) ==="
for s in train eval; do
  for l in hau ibo yor; do
    mkdir -p "wura/$s"
    curl -sfL -C - -o "wura/$s/$l.jsonl.part" "https://huggingface.co/datasets/castorini/wura/resolve/main/documents-v1.0/$s/$l.jsonl" \
      && mv "wura/$s/$l.jsonl.part" "wura/$s/$l.jsonl"
    echo "wura $s $l exit $?"
  done
done

echo "=== MADLAD-400 clean + noisy (ha/ig/yo) ==="
for code in ha ig yo; do
  mkdir -p "madlad/data/$code"
  for kind in clean noisy; do
    f="${code}_${kind}_0000.jsonl.gz"
    curl -sfL -C - -o "madlad/data/$code/$f.part" "https://huggingface.co/datasets/allenai/madlad-400/resolve/main/data/$code/$f" \
      && mv "madlad/data/$code/$f.part" "madlad/data/$code/$f"
    echo "madlad $code $kind exit $?"
  done
done

echo "=== HPLT 2.0 cleaned (hau_Latn/ibo_Latn/yor_Latn) ==="
# Shard count differs per language (hau_Latn: 3 files, ibo_Latn/yor_Latn: 1 each as of this build),
# so this uses huggingface-cli's own file listing rather than a hardcoded filename per language.
for lang in hau_Latn ibo_Latn yor_Latn; do
  mkdir -p "hplt/$lang"
  huggingface-cli download HPLT/HPLT2.0_cleaned --repo-type dataset \
    --include "$lang/*.parquet" --local-dir hplt_tmp
  mv hplt_tmp/"$lang"/*.parquet "hplt/$lang/"
done
rm -rf hplt_tmp

echo "=== done ==="
