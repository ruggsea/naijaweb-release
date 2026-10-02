#!/bin/bash
# Runner: deep-recrawl the curated seeds, a few hosts at a time.
#
# The network link may be shared and capped (set LINE_MBPS to its ceiling), and the WET crawl may
# already use most of it. So this measures before it adds load, and refuses rather than
# apologising afterwards: an uncapped run is invisible until someone else's transfer is slow.
#
# One request/second PER HOST is enforced inside rescrape.py. Parallelism here is
# across hosts, which is what keeps a fan-out from being impolite to any one site.
#
#   build/rescrape_run.sh                 # measure, then run the seed list
#   build/rescrape_run.sh --parallel 4    # 4 hosts at once
#   build/rescrape_run.sh --check-only    # just measure and report
#
# Stop: kill the runner's pid, then each rescrape.py whose parent is a runner subshell
# (ps --ppid <subshell pid>). Never pkill -f: it matches every shell whose command line
# mentions the name, including the one running it. A killed host restarts from zero.
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root

SEEDS=${SEEDS:-data/rescrape_seeds.tsv}
PARALLEL=${PARALLEL:-3}
MAX_PAGES=${MAX_PAGES:-10000}
LINE_MBPS=${LINE_MBPS:-150}          # the link's ceiling, MB/s
MARGIN_MBPS=${MARGIN_MBPS:-20}       # leave this much for the WET crawl and everything else
OUT=${RESCRAPE_OUT:-/local/$USER/rescrape}
[ -d /local/$USER ] || OUT=${RESCRAPE_OUT:-/tmp/rescrape}
LOGS=logs/rescrape
CHECKS=(${CHECK_ONLY:-no})

export GLOTLID=${GLOTLID:-$PWD/models--cis-lmu--glotlid/snapshots/85cd6716494360367b75f642b5bc78667605d0b4/model_v3.bin}
mkdir -p "$OUT" "$LOGS"

while [ $# -gt 0 ]; do
  case "$1" in
    --parallel) PARALLEL=$2; shift 2 ;;
    --max-pages) MAX_PAGES=$2; shift 2 ;;
    --seeds) SEEDS=$2; shift 2 ;;
    --out-dir) OUT=$2; shift 2 ;;
    --check-only) CHECKS=(yes); shift ;;
    *) echo "unknown arg $1" >&2; exit 2 ;;
  esac
done

# rx bytes over 10 s on the PUBLIC interface only, MB/s. The cap is on the uplink, and a box may
# also carry a private storage network; summing every interface would count storage traffic against
# the uplink. Two samples -- one cannot tell a quiet link from one that is between bursts.
PUBLIC_IF=$(ip route get 1.1.1.1 | awk '{for(k=1;k<=NF;k++) if($k=="dev") print $(k+1)}')
[ -r "/sys/class/net/$PUBLIC_IF/statistics/rx_bytes" ] || { echo "no public interface found (got '$PUBLIC_IF')"; exit 1; }
rx() {
  local r0 r1
  r0=$(cat "/sys/class/net/$PUBLIC_IF/statistics/rx_bytes")
  sleep 10
  r1=$(cat "/sys/class/net/$PUBLIC_IF/statistics/rx_bytes")
  echo $(( (r1 - r0) / 10 / 1000000 ))
}

echo "=== rescrape $(date -u +%FT%TZ) on $(hostname -s) ==="
A=$(rx); B=$(rx)
echo "uplink rx on $PUBLIC_IF: ${A} then ${B} MB/s (ceiling ${LINE_MBPS}, margin ${MARGIN_MBPS})"
if [ "$A" -gt $((LINE_MBPS - MARGIN_MBPS)) ] || [ "$B" -gt $((LINE_MBPS - MARGIN_MBPS)) ]; then
  echo "REFUSING: the link is already above $((LINE_MBPS - MARGIN_MBPS)) MB/s on a 10 s sample."
  echo "This run is the newest load; it stops itself rather than taking someone else's bandwidth."
  echo "Re-run when the WET crawl is between shards, or raise MARGIN_MBPS deliberately."
  exit 3
fi
[ "${CHECKS[0]}" = "yes" ] && { echo "check-only: not starting"; exit 0; }

[ -s "$SEEDS" ] || { echo "no seed file at $SEEDS"; exit 1; }
[ -s "$GLOTLID" ] || { echo "GLOTLID model missing at $GLOTLID"; exit 1; }

# One host per line, skipping comments and the header.
mapfile -t HOSTS < <(awk -F'\t' '!/^#/ && NF>=1 && $1!="host" {print $1}' "$SEEDS")
echo "seeds: ${#HOSTS[@]} hosts, ${PARALLEL} at a time, max ${MAX_PAGES} pages each, out $OUT"

queue=$(mktemp); printf '%s\n' "${HOSTS[@]}" > "$queue"
running=0
started=0

run_one() {
  local h="$1"
  # Skip a host already finished, so a re-run continues rather than restarts.
  if [ -s "$OUT/$h.jsonl.zst" ]; then
    echo "[skip] $h already written"
    return
  fi
  # Skip a host another rescrape.py is still fetching. Two writers on one host share one
  # .tmp and one state file. Matched on ps FIELDS (script, then the --host value), so a
  # shell whose command line merely mentions the host cannot qualify.
  if ps -eo args --no-headers | awk -v h="$h" '$1 ~ /python/ && $2=="build/rescrape.py" && $3=="--host" && $4==h {f=1} END{exit !f}'; then
    echo "[skip] $h is being fetched by another rescrape.py right now"
    return
  fi
  echo "[start] $h"
  .venv/bin/python build/rescrape.py --host "$h" --max-pages "$MAX_PAGES" \
      --out-dir "$OUT" > "$LOGS/$h.log" 2>&1
  echo "[done ] $h $(tail -1 "$LOGS/$h.log" | head -c 160)"
}

while read -r h; do
  [ -z "$h" ] && continue
  run_one "$h" &
  running=$((running+1)); started=$((started+1))
  if [ "$running" -ge "$PARALLEL" ]; then
    wait -n 2>/dev/null || wait
    running=$((running-1))
    # Re-measure between batches: the WET crawl's own rate moves, and a run that was
    # polite an hour ago is not automatically polite now.
    C=$(rx)
    if [ "$C" -gt $((LINE_MBPS - MARGIN_MBPS)) ]; then
      # Stop ADDING hosts; the ones running finish. (Killing the run_one subshells would orphan the
      # rescrape.py children and lose their [done] lines.)
      echo "STOPPING: link at ${C} MB/s mid-run; not adding the next host, letting $running finish."
      break
    fi
  fi
done < "$queue"
wait
rm -f "$queue"

kept=$(find "$OUT" -name '*.jsonl.zst' -size +0 2>/dev/null | wc -l)
echo "=== done $(date -u +%FT%TZ): $started hosts started, $kept shards written in $OUT ==="
echo "per-host outcomes are in $LOGS/<host>.log (every rejection reason is counted there)"
