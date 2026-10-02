# classifier/

The script that applies the quality classifier, its annotation codebook, and the script that
builds the keep/drop split from raw annotations.

- `score.py` — scores any pool version with a trained classifier: a local checkpoint or a Hugging
  Face repo id given on the command line, or, with none given, `NAIJAWEB_CLASSIFIER` (see
  `config.py`). Applied to the pool, the re-crawl and the FineWeb-2 fold-in.
- `test.py` — the classifier on the human test set (majority label of three raters), scored as
  `score.py` scores the corpus: accuracy and keep precision/recall per language.
- `CODEBOOK.md` — the four-key annotation codebook (right language/quality, wrong language, not
  language) annotators read.
- `build_label_dataset.py` — builds the host-disjoint keep/drop training split from the raw
  annotation exports in `NAIJAWEB_LABELS`: collapses the four codebook keys to binary keep/drop,
  and separates a majority-vote overlap test set from a host-disjoint training pool.

The classifier's training code and the annotation web tool are not part of this repo. The classifier
and the labels are released with the corpus.
