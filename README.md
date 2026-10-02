# NaijaWeb

Code to build a quality-filtered web corpus for Hausa, Igbo and Yorùbá from Common Crawl plus a
curated re-crawl of Nigerian sites, and to evaluate a corpus by pretraining a small language model
on it and measuring bits per byte on held-out text.

It does not include raw crawl data, model weights, the classifier or the corpus itself (see "Not
included").

## Install

- Python 3.11.14 and [`uv`](https://docs.astral.sh/uv/) (or plain `venv`):

  ```bash
  uv venv --python 3.11
  uv pip install -r requirements.txt
  ```

- Set `NAIJAWEB_DATA` to a directory for crawl output, built corpus versions and downloaded
  comparison corpora (the layout is documented in `config.py`):

  ```bash
  export NAIJAWEB_DATA=/path/to/a/data/volume
  ```

- Two stages need files that are released with the corpus: the quality classifier
  (`NAIJAWEB_CLASSIFIER`, a Hugging Face repo id or a local folder) and the human keep/drop labels
  (`NAIJAWEB_LABELS`, a local folder). A stage that needs one stops with an error if it is not set.

## Pipeline

Each step lists its command and where its output goes. Run them in order.

Before step 2: `data/wet_paths_<snapshot>.txt` is the snapshot's WET file list, i.e. Common
Crawl's `https://data.commoncrawl.org/crawl-data/<snapshot>/wet.paths.gz`, unzipped. The GlotLID v3
model (`cis-lmu/glotlid`, file `model_v3.bin`) must be downloaded
(`huggingface-cli download cis-lmu/glotlid model_v3.bin --revision 85cd6716494360367b75f642b5bc78667605d0b4`)
and `GLOTLID` set to that file (`crawl/run_wet.sh` reads it from the environment).

| # | Stage | Command | Output |
|---|---|---|---|
| 1 | Download FineWeb-2 and the comparison corpora (WURA, MADLAD-400, HPLT 2.0) | `bash build/get_baselines.sh` | `$NAIJAWEB_DATA/baselines/{fineweb2,wura,madlad,hplt}/` |
| 2 | Crawl and language ID (GlotLID, threshold 0.3) | `bash crawl/run_wet.sh <snapshot>` | `$NAIJAWEB_DATA/wet/collected/*.jsonl.zst` |
| 3 | Rule filters and boilerplate removal, then MinHash dedup within each snapshot | `bash filter/filter_all.sh` (per snapshot: `filter/apply_filters.py`, then `dedup.py sign` and `cluster`) | `$NAIJAWEB_DATA/wet/filtered/<snapshot>/`, `results/filtered_<snapshot>.json` (documents removed per rule), `data/dedup_filtered/<snapshot>/drop.tsv` |
| 4 | MinHash dedup across all snapshots | `DEDUP_SET=ALL30 bash filter/dedup_across.sh` | `data/dedup_filtered/ALL30/drop_global.tsv` |
| 5 | Base pool | `NAIJAWEB_SET=ALL30 python filter/build_a.py percrawl` | `$NAIJAWEB_DATA/wet/versions/P_ALL30/` |
| 6 | Held-out sets and the Wikipedia identity hold-out | `python build/heldout.py` then `python build/wiki_identity.py` | `data/heldout/<set>_<lang>.jsonl` (FLORES via Belebele, AfriMMLU, MasakhaNEWS test, a Wikipedia sample), `data/heldout_ids/wiki_<lang>.jsonl`, `data/final/wiki_heldout_ids.txt` |
| 7 | Re-crawl of curated Nigerian sites | `python build/rescrape_seeds.py`, then `bash build/rescrape_run.sh` (or `python build/rescrape.py --seeds data/rescrape_seeds.tsv`), then `python build/recrawl_build.py <out dir>` | `$NAIJAWEB_DATA/wet/versions/R_recrawl*/` |
| 8 | FineWeb-2 fold-in | `python build/fineweb2_pool.py` | `$NAIJAWEB_DATA/wet/versions/F_fineweb2/` |
| 9 | Classifier scores for each pool version | `python classifier/score.py <name> [classifier]` (set `NAIJAWEB_SRC` to pick the version) | `$NAIJAWEB_DATA/wet/versions/scores/<name>/` |
| 10 | Release | `python build/release_sizes.py <re-crawl dir> --write <out>`, then `python build/release_parquet.py <out> <parquet dir>`, then `python build/release_loadtest.py <parquet dir>` | the corpus as jsonl and parquet, with `manifest.json` |

Decontamination: every build drops a document that shares a 13-word sequence with a held-out set,
and any Wikipedia page that is a held-out article (`build/grams.py`). Run the held-out step (6) before the
re-crawl build (7) and the release (10).

The 32k tokenizer is checked in (`train/tok32k/`); `python train/tokenizer.py` retrains it.

## Evaluating a corpus

A corpus is evaluated by pretraining the same small model on it and reading bits per byte on the
held-out sets from step 6 (`data/heldout/`).

- `train/pretrain.py --version <name> --seed <n> --passes <epochs>` trains a 12-layer, 768-hidden
  decoder (134M parameters with embeddings) on `train/bins/<name>.bin`: uint16 token ids from the
  tokenizer in `train/tok32k/`, each document ended by token 0. To evaluate any other corpus, write
  a `.bin` the same way. It writes `results/curves_pretrain/<name>_e<epochs>_s<seed>.jsonl`, one
  row per evaluation step with `bits_per_byte` for every held-out set and language. About 134
  minutes for 10 epochs at 290M tokens on one H100; `jobs/pretrain_example.sbatch` is a generic
  Slurm launcher (fill in your account and partition).
- `python eval/compare.py <epochs> <name> <name> ...` reads those curves at the final step. Per
  held-out set it prints each version's mean and [min-max] over seeds, and for every pair how many
  sets one version wins, loses or cannot be told apart: a win means every seed of one version is
  below every seed of the other.

## Experiments

Three comparisons are built into the repo. Each needs steps 1-9 above done first, and a run at 1 and at 10 epochs (`--passes 1` and
`--passes 10`); we used 3 seeds per version for the first and 5 for the other two.

**1. Deduplication: none vs within each snapshot vs across snapshots, same token budget.**

```bash
NAIJAWEB_SET=ALL30 python filter/build_a.py none      # the no-dedup pool N_ALL30
python train/tokenize_all.py                          # train/bins/all.bin + all_index.json
python build/build_dedup.py                           # train/bins/dedup_{none,within,across}.bin
for v in dedup_none dedup_within dedup_across; do for s in 1 2 3; do
  python train/pretrain.py --version $v --seed $s --passes 10; done; done
python eval/compare.py 10 dedup_none dedup_within dedup_across
```

**2. The quality classifier's selection vs a random sample of the same pool.** Both versions are
cut to the same tokens per language, with no site above 20% of a language.

```bash
python build/build_matched.py <re-crawl dir> --write  # train/bins/matched_{random,cls}.bin
for v in matched_random matched_cls; do for s in 1 2 3 4 5; do
  python train/pretrain.py --version $v --seed $s --passes 10; done; done
python eval/compare.py 10 matched_random matched_cls
```

**3. Six corpora at the same token budget:** ours, WURA, FineWeb-2, MADLAD-400 (noisy and clean),
HPLT 2.0. Each is deduplicated within itself and decontaminated against the held-out sets before
the cut.

```bash
python build/build_baselines.py <re-crawl dir> --write     # corpus_{ours,wura,fw2}
python build/build_baselines_13.py <re-crawl dir> --write  # corpus_{madlad,madlad_clean,hplt}
for v in corpus_ours corpus_wura corpus_fw2 corpus_madlad corpus_madlad_clean corpus_hplt; do for s in 1 2 3 4 5; do
  python train/pretrain.py --version $v --seed $s --passes 10; done; done
python eval/compare.py 10 corpus_ours corpus_wura corpus_fw2 corpus_madlad corpus_madlad_clean corpus_hplt
```

## Layout

- `crawl/` — Common Crawl WET fetch and language ID.
- `filter/` — rule-based quality filters, boilerplate removal, MinHash deduplication.
- `build/` — held-out sets and decontamination, the re-crawl, the FineWeb-2 fold-in, the training-set
  builders for the three experiments, and the release recipe.
- `classifier/` — the script that applies the quality classifier, its evaluation on the human test
  set (`test.py`), the annotation codebook, and the script that builds the keep/drop split from raw
  annotations.
- `train/` — the tokenizer, the training-file builder, `train/tok32k/` and the pretraining script.
- `eval/` — `compare.py`: bits per byte per held-out set and win/loss/tie between versions.
- `jobs/` — a generic Slurm example for pretraining.
- `data/stopwords/` — the Hausa and Yorùbá stop-word lists used by one filter rule (there is no
  Igbo list; see `filter/filters.py`'s docstring).
- `results/` — where the filter and pretraining stages write their outputs (empty in git).
- `smoke/` — a tiny end-to-end run over synthetic data.

Every folder has a short `AGENTS.md` with entry points and gotchas.

## Smoke test

```bash
bash smoke/run.sh
```

runs the filters, dedup, the decontamination check, tokenization and a few pretraining steps with
held-out bits per byte, over a few hundred synthetic documents, on CPU, in a few minutes. It
checks that the stages run end to end on this layout; it does not produce a usable corpus or model.

## Not included

- Raw crawl data, model weights and intermediate corpus versions under `$NAIJAWEB_DATA`; all are
  regenerable from the scripts given the external inputs (Common Crawl, the re-crawl, the
  comparison corpora).
- The quality classifier, its training code and the human labels. They are released with the
  corpus and read through `NAIJAWEB_CLASSIFIER` and `NAIJAWEB_LABELS`. The annotation web tool is
  not included; its codebook is `classifier/CODEBOOK.md`.
- The corpus itself.
- Scripts that orchestrate crawls across particular machines.

## License

Code: MIT, see `LICENSE`. The corpus's data license is stated with the corpus.
