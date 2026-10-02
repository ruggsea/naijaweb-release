# data/stopwords/

Stop-word lists for `filter/filters.py`'s `wura_stopwords` rule (drops a document with fewer than
5 stopwords from its language's list).

- `ha.txt` — Hausa.
- `yo.txt` — Yorùbá.

There is no `ibo.txt`: WURA (whose convention this rule follows) has no Igbo stop-word list, so
Igbo documents pass this rule unconditionally. This is deliberate, not a missing file — see
`filter/filters.py`'s docstring.
