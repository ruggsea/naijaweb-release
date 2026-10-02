# build/

FineWeb-2 fold-in, the re-crawl, held-out sets + decontamination, token-matched corpus builds for
evaluation, and the release recipe.

- `get_baselines.sh`, `fineweb2_files.txt` — downloads FineWeb-2, WURA, MADLAD-400 and HPLT 2.0
  cleaned into `$NAIJAWEB_DATA/baselines/`. Run this first; everything else in this folder that
  reads `DATA / "baselines"` depends on it.
- `fineweb2_pool.py` — folds in FineWeb-2's Hausa/Igbo/Yorùbá train split through our own filters
  and dedup, dropping anything already in the base pool.
- `heldout.py`, `wiki_identity.py`, `grams.py` — build the four held-out sets (AfriMMLU, FLORES,
  MasakhaNEWS test, Wikipedia) and the 13-word-sequence hashing used for decontamination everywhere.
- `wiki_strip.py` — strips Wikipedia page furniture before scoring (not before keeping).
- `rescrape.py`, `rescrape_seeds.py`, `rescrape_run.sh` — the curated re-crawl: robots.txt +
  sitemap enumeration at 1 req/s/host.
- `recrawl_build.py`, `mojibake_scan.py` — build the re-crawl add-on from finished re-crawl shards
  (drops overlap with the base pool, exact/near duplicates, mojibake, listing pages).
- `build_dedup.py` — experiment 1: the no-dedup / within-snapshot / across-snapshot training sets at
  one token budget, cut from the tokenized no-dedup pool.
- `build_matched.py` — experiment 2: classifier selection vs random sample, matched per language with a
  20% site cap.
- `build_baselines.py`, `build_baselines_13.py` — experiment 3: training sets for our corpus, WURA,
  FineWeb-2, MADLAD-400 and HPLT 2.0, matched on total tokens only. They write `train/bins/*.bin` and
  `data/matched/<name>.ids`.
- `release_sizes.py`, `release_parquet.py`, `release_loadtest.py` — the release recipe: same steps as
  the baselines builder but no token-budget cut, writing the release text, its parquet files and a
  load test.

Gotcha: `build_baselines.py`'s `ours` means *our corpus* (the release recipe), not a classifier.
