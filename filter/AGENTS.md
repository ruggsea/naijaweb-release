# filter/

Rule-based quality filters, CCNet-style boilerplate removal, and MinHash deduplication.

- `filters.py` — the mechanical rules (Gopher quality/repetition, machine-translated-site URL
  rules, Wikipedia-infrastructure-page rule, a stopword-count rule). `path_mt_hosts.tsv` (hand-sorted
  keep/drop list) is one rule's input.
- `apply_filters.py` — runs the rules plus boilerplate removal over one crawl's collected WET
  output; writes the funnel counts and the survivors.
- `filter_all.sh` — drives `apply_filters.py` + `dedup.py` over every crawl whose WET files are
  collected.
- `dedup.py` — MinHash near-duplicate removal (FineWeb's parameters: 112 hashes / 14 bands of 8).
  `sign` then `cluster`, run per snapshot.
- `dedup_across.sh` — the same MinHash, across all 30 filtered snapshots at once; this is what
  defines the corpus's base pool.
- `build_a.py` — builds the deduplicated pool version from `dedup_across.sh`'s output
  (`NAIJAWEB_SET=ALL30 python filter/build_a.py percrawl`).

Gotcha: `dedup.py`'s `DEDUP_IN`/`DEDUP_OUT` env vars point it at a specific crawl; `filter_all.sh`
sets them per snapshot. Don't run `dedup_across.sh` over a different crawl set without renaming the
output — `ALL30` is load-bearing, see the script's own comments.
