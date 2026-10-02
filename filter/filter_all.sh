# Filters + MinHash dedup over every crawl whose WET files are all in collected/.
# Skips a crawl whose data/dedup_filtered/<crawl>/drop.tsv exists, so reruns are cheap.
cd "$(dirname "$0")/.."   # repo root
# data/ is an INTERFACE, not a scratch directory: this glob turns every wet_paths_*.txt
# into a crawl. Anything dropped in here is an argument to this pipeline, so scratch
# queues belong outside data/.
for p in data/wet_paths_*.txt; do
  c=$(basename $p .txt); c=${c#wet_paths_}
  # An empty queue is not a crawl: it would sail through the miss check below (0 missing of 0)
  # and reach apply_filters.py, which asserts.
  [ -s $p ] && grep -q "[^[:space:]]" $p || { echo "$c skip: queue is empty"; continue; }
  [ -f data/dedup_filtered/$c/drop.tsv ] && { echo "$c done already"; continue; }
  miss=$(sed 's#.*/##; s#\.warc\.wet\.gz$#.jsonl.zst#' $p | while read n; do [ -f $NAIJAWEB_DATA/wet/collected/$n ] || echo x; done | wc -l)
  [ "$miss" -gt 0 ] && { echo "$c skip: $miss files not collected"; continue; }
  echo "$c start $(date -u +%H:%MZ)"
  uv run python filter/apply_filters.py $c > /dev/null || { echo "$c filters exit $?"; continue; }
  export DEDUP_IN=$NAIJAWEB_DATA/wet/filtered/$c DEDUP_OUT=$(dirname "$0")/../data/dedup_filtered/$c
  uv run python filter/dedup.py sign | tail -1 && uv run python filter/dedup.py cluster | tail -3
  echo "$c exit $? $(date -u +%H:%MZ)"
done
