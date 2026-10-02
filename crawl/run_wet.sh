#!/bin/bash
# Launch N sharded WET workers on this machine. Shards are strided, so any
# subset of machines covers a disjoint slice and nothing is fetched twice.
# CRAWL SHARDS_TOTAL SHARD_OFFSET SHARDS_HERE
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
: "${NAIJAWEB_DATA:?set NAIJAWEB_DATA, see config.py}"
CRAWL=${1:-CC-MAIN-2026-34}; TOTAL=${2:-96}; OFF=${3:-0}; HERE=${4:-96}
OUT=$NAIJAWEB_DATA/wet/collected; mkdir -p $OUT logs   # every crawl lands here; WET file names are unique across crawls
# Default the cap explicitly. An unset WET_RATE_KBPS means UNLIMITED, which
# is how a crawl can run uncapped at hundreds of MB/s for a day without anyone noticing.
export WET_RATE_KBPS=${WET_RATE_KBPS:-0}
export GLOTLID=${GLOTLID:-$PWD/models--cis-lmu--glotlid/snapshots/85cd6716494360367b75f642b5bc78667605d0b4/model_v3.bin}
for i in $(seq $OFF $((OFF+HERE-1))); do
  nohup .venv/bin/python crawl/wet_lid.py data/wet_paths_$CRAWL.txt $OUT $i $TOTAL \
    > logs/wet_$i.log 2>&1 &
done
echo "launched $HERE shards ($OFF..$((OFF+HERE-1))) of $TOTAL on $(hostname -s)"
