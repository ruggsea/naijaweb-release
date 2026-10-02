"""Shared path configuration for the whole repo.

Every script that reads or writes large, external, or intermediate data (raw crawl output, built
corpus versions, external baseline corpora, classifier scores) goes through this file instead of
hardcoding a home directory. Set one environment variable before running anything:

    export NAIJAWEB_DATA=/path/to/your/data/volume

Layout under $NAIJAWEB_DATA:

    wet/collected/<crawl>/       language-ID output per Common Crawl snapshot (crawl/wet_lid.py)
    wet/filtered/<crawl>/        post rule-filter + boilerplate-removal survivors (filter/apply_filters.py)
    wet/filtered_all/            filtered snapshots merged/linked for cross-snapshot dedup
    wet/rescrape/                raw re-crawl output + per-host .state.json (build/rescrape.py)
    wet/versions/<name>/         built pool versions: N_<set> (no dedup), P_<set> (within snapshot), A_<set> (across), F_fineweb2, R_recrawl*, ...
    wet/versions/scores/<who>/   classifier scores per pool version (classifier/score.py)
    baselines/<corpus>/          external comparison corpora as published (WURA, FineWeb-2, MADLAD-400, HPLT)

See each script's own docstring for which of these it reads or writes.
"""
import os
from pathlib import Path

DATA = Path(os.environ["NAIJAWEB_DATA"])

WET = DATA / "wet"
COLLECTED = WET / "collected"
FILTERED = WET / "filtered"
FILTERED_ALL = WET / "filtered_all"
RESCRAPE = WET / "rescrape"
VERSIONS = WET / "versions"
SCORES = VERSIONS / "scores"
BASELINES = DATA / "baselines"

# The set of crawls the corpus is built from: named by filter/dedup_across.sh (DEDUP_SET) and read by filter/build_a.py,
# build/build_dedup.py and the tokenizer scripts.
SET = os.environ.get("NAIJAWEB_SET", "ALL30")

# The quality classifier (a Hugging Face repo id or a local folder) and the human keep/drop labels it was trained on
# (a local folder with the raw exports, see classifier/build_label_dataset.py). Both are released with the corpus.
CLASSIFIER = os.environ.get("NAIJAWEB_CLASSIFIER", "")
LABELS = os.environ.get("NAIJAWEB_LABELS", "")

def need(value, env):
    """Fail fast, with the variable's name, when a stage needs a setting that was not given."""
    if not value:
        raise SystemExit(f"set {env}: this stage needs it (the released classifier / labels, see README.md)")
    return value
