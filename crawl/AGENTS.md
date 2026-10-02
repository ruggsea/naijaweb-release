# crawl/

Common Crawl acquisition: the language-ID pass over WET records.

- `wet_lid.py` — streams a snapshot's WET files and runs GlotLID v3 over every record, keeping
  `hau_Latn`/`ibo_Latn`/`yor_Latn` at `theta=0.3`. This decides corpus membership.
- `run_wet.sh` — launches N sharded `wet_lid.py` workers on one machine. Shards are strided, so
  several machines can each run a disjoint slice by passing different offsets.

Gotcha: `wet_lid.py` needs a GlotLID v3 model file (`GLOTLID` env var, see its own docstring) and
is bandwidth-heavy — cap it with `WET_RATE_KBPS` if running on a shared network.
