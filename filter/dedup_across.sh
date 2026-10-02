# MinHash over all filtered crawls at once, to count pages repeated between crawls.
# Input is a folder of symlinks to every $NAIJAWEB_DATA/wet/filtered/<crawl>/ file (WET names are unique across crawls).
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root
# The output set is named, and an existing one is never overwritten silently: this script begins with
# rm -rf on it, and its drop.tsv + drop_global.tsv define every corpus version built from them.
# Name a set for what it covers; the crawl count lives in $OUT.crawls, as data. A number in the
# name has to equal the number of crawls (checked below), so a result is never quoted with the wrong denominator.
SET=${DEDUP_SET:?set DEDUP_SET, e.g. DEDUP_SET=ALL30}
OUT=$(dirname "$0")/../data/dedup_filtered/$SET
if [ -d "$OUT" ] && [ "${DEDUP_OVERWRITE:-}" != yes ]; then
  echo "$OUT exists; pass DEDUP_OVERWRITE=yes to replace it. Nothing done."; exit 1
fi
bash filter/filter_all.sh   # picks up any crawl that finished collecting since the last run
L=$NAIJAWEB_DATA/wet/filtered_all; rm -rf $L.tmp; mkdir $L.tmp
find $NAIJAWEB_DATA/wet/filtered -mindepth 2 -maxdepth 2 -path '*/filtered/CC-MAIN-*' -name '*.jsonl.zst' -exec ln -s -t $L.tmp {} +   # a glob overflows argv at 90k files
want=$(find $NAIJAWEB_DATA/wet/filtered -mindepth 2 -maxdepth 2 -path '*/filtered/CC-MAIN-*' -name '*.jsonl.zst' | wc -l); got=$(ls $L.tmp | wc -l)
[ "$want" -eq "$got" ] || { echo "linked $got of $want"; exit 1; }
rm -rf $L; mv $L.tmp $L; echo "files $got"
# The crawls actually in the link farm are written next to the output; if the set name contains a number it has to
# be the number of crawls in it. Derived at run time, never typed once and left behind.
ls $L | sed -E "s/^CC-MAIN-([0-9]{8}).*/\1/" | sort -u > /dev/null   # names carry timestamps, not crawl ids
find $NAIJAWEB_DATA/wet/filtered -mindepth 1 -maxdepth 1 -name "CC-MAIN-*" -printf "%f\n" | sort > $OUT.crawls
NCRAWLS=$(wc -l < $OUT.crawls)
NUM=$(echo "$SET" | tr -dc 0-9)
if [ -n "$NUM" ] && [ "$NUM" != "$NCRAWLS" ]; then
  echo "DEDUP_SET=$SET says $NUM but the set covers $NCRAWLS crawls (see $OUT.crawls). Nothing done."
  rm -f $OUT.crawls; exit 1
fi
echo "crawls in this set: $NCRAWLS -> $OUT.crawls"
rm -rf $OUT   # signatures are cached per chunk; a changed input set must start clean
export DEDUP_IN=$L DEDUP_OUT=$OUT
uv run python filter/dedup.py sign | tail -1
uv run python filter/dedup.py cluster
