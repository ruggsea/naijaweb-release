"""Wide-net pass: our own GlotLID over every document in Common Crawl.

WET files are pre-extracted plain text for every page, 6 TB per snapshot
against 90 TB of WARC, so this runs OUR language identification over the whole
crawl rather than trusting the index's CLD2 tags. WET text is boilerplate-heavy
and the filters downstream (filter/apply_filters.py) clean it up.

Streamed and discarded: a WET file is never written to disk. One output shard
per WET file, atomic rename, skipped if already present, so this is resumable
at file granularity and safe to run on many machines at once.
"""
import os, sys, json, time, requests, urllib3, zstandard, fasttext
from warcio.archiveiterator import ArchiveIterator

THETA = 0.3
KEEP = {"hau_Latn", "ibo_Latn", "yor_Latn"}  # pcm dropped: 16 of 21 hits were English boilerplate
BASE = "https://data.commoncrawl.org/"
MINCHARS = 200

_sess = None
def _session():
    global _sess
    if _sess is None:
        _sess = requests.Session()
        _sess.headers["user-agent"] = ("naijaweb-research/0.1 "
                                       "(academic corpus construction; contact via the repo README)")
    return _sess

_model = None
def lid(text):
    global _model
    if _model is None:
        _model = fasttext.load_model(os.environ["GLOTLID"])
    labs, probs = _model.predict(text.replace("\n", " ")[:3000], k=1)
    return labs[0].removeprefix("__label__"), float(probs[0])

class Transient(Exception):
    """A fetch that is worth trying again: bad status, or a network error."""

# Optional per-worker bandwidth cap, from WET_RATE_KBPS (default 0 = unlimited).
#
# Why this exists: with ~1,000 workers and no cap they pull hundreds of MB/s
# in aggregate. That saturates the uplink and starves anything else on it.
#
# Why it is implemented here rather than with tc: there is no passwordless sudo
# on these hosts, so no qdisc; and one fetch is a single 63 MB body with no
# per-request knob to turn. Slowing the reader is the only lever available, and
# it caps real throughput rather than merely spacing requests out.
RATE_KBPS = int(os.environ.get("WET_RATE_KBPS", "0") or "0")
RATE_BPS = RATE_KBPS * 1024


class _Throttled:
    """Wrap a stream so its reads average at most `bps` bytes/sec.

    Paces against elapsed time rather than sleeping a fixed amount per chunk, so
    it holds the average even when the server bursts or stalls, and it never
    sleeps more than the deficit it has actually accumulated. With bps=0 every
    read passes straight through and the wrapper costs one comparison.
    """

    def __init__(self, raw, bps):
        self._raw, self._bps = raw, bps
        self._t0, self._n = time.time(), 0

    def read(self, n=-1):
        chunk = self._raw.read(n)
        if chunk and self._bps:
            self._n += len(chunk)
            deficit = self._n / self._bps - (time.time() - self._t0)
            if deficit > 0:
                time.sleep(deficit)
        return chunk

    def __getattr__(self, name):
        # ArchiveIterator needs the rest of the raw response's surface
        # (readinto, readable, closed, ...) and must not tell the difference.
        return getattr(self._raw, name)

# urllib3's errors are NOT requests errors: ArchiveIterator reads r.raw, the raw
# urllib3 stream, so a body-read timeout arrives as urllib3.ReadTimeoutError and
# a handler catching only RequestException misses it.
RETRYABLE = (Transient, requests.exceptions.RequestException, urllib3.exceptions.HTTPError)

def _fetch_once(path, tmp):
    """One whole attempt: headers, 63 MB body, decode, write. stream=True returns as soon as the
    HEADERS arrive, so the retry unit has to be the whole download, not the .get()."""
    r = _session().get(BASE + path, stream=True, timeout=(30, 300))
    try:
        if r.status_code != 200:
            raise Transient(f"HTTP {r.status_code}")
        seen = kept = 0
        with open(tmp, "wb") as fh:
            w = zstandard.ZstdCompressor(level=6).stream_writer(fh)
            for rec in ArchiveIterator(_Throttled(r.raw, RATE_BPS), arc2warc=False):
                if rec.rec_type != "conversion":
                    continue
                text = rec.content_stream().read().decode("utf-8", "replace")
                seen += 1
                if len(text) < MINCHARS:
                    continue
                lang, p = lid(text)
                if lang in KEEP and p >= THETA:
                    kept += 1
                    w.write((json.dumps({
                        "url": rec.rec_headers.get_header("WARC-Target-URI"),
                        "lang": lang, "lid_prob": p, "wet_text": text[:20000],
                        "wet_file": path}, ensure_ascii=False) + "\n").encode())
            w.close()
        return seen, kept
    finally:
        r.close()

def process(path, outdir):
    name = path.rsplit("/", 1)[-1].replace(".warc.wet.gz", "")
    dst = f"{outdir}/{name}.jsonl.zst"
    if os.path.exists(dst):
        return None
    tmp = dst + ".tmp"
    for attempt in range(6):
        try:
            seen, kept = _fetch_once(path, tmp)
        except RETRYABLE as e:
            # Writing to .tmp is what makes a failed attempt discardable: a
            # half-written file is removed and never renamed over dst.
            if os.path.exists(tmp):
                os.remove(tmp)
            if attempt == 5:
                raise RuntimeError(f"{path}: {type(e).__name__}: {e} after 6 tries") from e
            time.sleep(min(300, 15 * 2 ** attempt))   # polite backoff, as CC asks
            continue
        os.replace(tmp, dst)                          # atomic publish
        return seen, kept

if __name__ == "__main__":
    paths_file, outdir, shard, nshards = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    if RATE_BPS:
        print(f"[shard {shard}/{nshards}] bandwidth capped at {RATE_KBPS} KB/s per worker",
              flush=True)
    os.makedirs(outdir, exist_ok=True)
    paths = [l.strip() for l in open(paths_file) if l.strip()]
    mine = [p for i, p in enumerate(paths) if i % nshards == shard]
    t0, S, K, n = time.time(), 0, 0, 0
    failed = []
    for p in mine:
        # One unfetchable file must not cost the shard's remaining files.
        try:
            r = process(p, outdir)
        except RuntimeError as e:
            print(f"SKIP {p}: {e}", flush=True)
            failed.append(p)
            continue
        if r:
            S += r[0]; K += r[1]; n += 1
            if n % 5 == 0:
                dt = time.time() - t0
                print(f"{n}/{len(mine)} files  {S:,} docs  {K:,} HIY  "
                      f"{S/dt:,.0f} docs/s  {dt/n:.0f}s/file", flush=True)
    print(f"DONE shard {shard}: {n} files, {S:,} docs, {K:,} HIY")
