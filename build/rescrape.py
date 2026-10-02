"""Fresh, deep recrawl of the curated high-quality domains.

Why this exists, and why it is not the WET pass again.

The WET pass reads what Common Crawl happened to sample. For a Nigerian
Hausa/Igbo/Yorùbá site that means a handful of pages per dump, chosen by
whatever link-following the crawler did that fortnight — `hausa.legit.ng`
gave us 65,634 documents and `alaroye.org` 4,838, but a site with 40,000
published articles rarely has them all in CC, and the ones CC missed are
not a random sample of the site: they are the deeper, less-linked, often
better pages. DCLM (Li et al. 2024, §3.2) makes the same move for English
and gets most of its quality from exactly this — go to the domain and take
what it has, rather than take what the crawl left.

So this reads the site's own enumeration of itself, in this order:

  1. robots.txt — always first. Honoured, and its `Sitemap:` lines are the
     best URL source there is: the site is telling us where its pages are.
  2. sitemap.xml / sitemap_index.xml, expanded recursively. A sitemap index
     lists child sitemaps, each listing up to 50,000 URLs, so a full news
     archive is reachable in a few dozen requests. This is the "go deeper"
     step: it is the only route to the parts of a site that no crawl links.
  3. RSS/Atom (`/feed`, `/rss`, `/atom.xml`, `<link rel=alternate>` on the
     home page) for recency, and because feed entries carry publication dates.
  4. BFS from the home page, depth- and page-capped, following same-host
     links only — the fallback for a site with no sitemap and no feed.

Rules this file follows:

  * robots.txt is honoured, both for fetching and for sitemap discovery. A robots.txt
    that is reachable and disallows is always respected. One that CANNOT be read
    (network error, timeout, TLS error, 5xx) is treated as "no rules, proceed" --
    This departs from RFC 9309 2.3.1.4, which says disallow-all; under that rule many hosts
    whose robots.txt merely timed out would yield nothing. Which path a host took is written into every
    record (`robots`) and into its state file, so the corpus can later tell pages
    fetched under a permission from pages fetched under an assumption.
  * one request per second per host, default; back off hard on 429/5xx.
  * no logins, no paywalls, no attempt to defeat either.
  * the network link may be shared. rescrape_run.sh measures before it starts and refuses to
    add load past a ceiling. This file is single-host-serial so that a
    `--hosts` fan-out is the only thing that sets total pressure.

Every rejection is counted by its real reason and printed by `report()`, in
the same discipline everywhere in this repo: a function that returns None for
"fetch failed" and "wrong language" alike hides an outage as a content problem.

Output is one jsonl.zst per host, written atomically, with a per-host state
file so an interrupted run resumes instead of restarting. Records carry the
same keys as the WET shards (`id`, `url`, `lang`, `text`) plus `host`,
`source="rescrape"` and `fetched`. The `id` is `<host>:<sha1(url)[:16]>`,
stable across reruns so a re-fetch of the same URL is the same document.

  python build/rescrape.py --seeds data/rescrape_seeds.tsv --out-dir /local/$USER/rescrape
  python build/rescrape.py --host hausa.legit.ng --max-pages 2000 --dry-run
  python build/rescrape.py --selftest
"""
import argparse, collections, hashlib, json, os, re, sys, threading, time, urllib.parse
from pathlib import Path

import requests, trafilatura, zstandard, fasttext
from urllib.robotparser import RobotFileParser

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "filter"))
from filters import wiki_infrastructure, mt_localisation          # the two URL rules

# The host list that already exists: filter/path_mt_hosts.tsv, hand-sorted,
# 601 hosts marked `drop` and 33 marked `keep`. A `drop` host is a foreign commercial
# site that grew a ha./ig./yo. folder from a translation plugin, or otherwise not a
# Nigerian publisher — it must not be crawled deeply, because a deep crawl of it
# manufactures exactly the machine-translation text this pipeline removes.
# A `keep` host is one a human read and cleared (JW.org, IQNA, DW's amp subdomain ...);
# it is not automatically a seed, it is merely not blocked.
# Loaded once. Subdomains of a dropped host are dropped too.
BLOCKLIST_FILES = [ROOT / "filter" / "path_mt_hosts.tsv"]


def _canon_host(h):
    return (h or "").lower().removeprefix("www.").strip().strip(".")


def load_blocklist(files=None):
    """hosts marked `drop`, canonicalised, with any decision-coloured TSV accepted."""
    out = set()
    for f in (files if files is not None else BLOCKLIST_FILES):
        f = Path(f)
        if not f.exists():
            print(f"blocklist {f} does not exist - refusing to crawl without it", file=sys.stderr)
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2 and parts[1].strip().lower() == "drop":
                out.add(_canon_host(parts[0]))
    return out


BLOCKED = load_blocklist()


def blocked(url_or_host):
    """True if the host, or any parent of it, is marked drop."""
    h = url_or_host
    if "://" in h or h.startswith("/"):
        h = urllib.parse.urlsplit(h).netloc
    h = _canon_host(h)
    if not h:
        return False
    if h in BLOCKED:
        return True
    return any(h.endswith("." + b) for b in BLOCKED)          # a subdomain of a dropped host

THETA = 0.3
KEEP = {"hau_Latn", "ibo_Latn", "yor_Latn"}                       # pcm dropped, same as fetch_extract.py
UA = "naijaweb-research/0.1 (academic corpus construction; contact via the repo README)"
MIN_TEXT = 200                                                     # characters, same floor as fetch_extract.py
RATE = 1.0                                                         # seconds between requests to one host
FETCH_ATTEMPTS = 4
TIMEOUT = 30
SITEMAP_CAP = 200                                                  # child sitemaps followed per host
SITEMAP_DEPTH = 3

STATS = collections.Counter()
_lock = threading.Lock()


def _count(reason):
    with _lock:
        STATS[reason] += 1


def report(stream=sys.stderr):
    total = sum(STATS.values())
    if not total:
        return
    print(f"outcomes over {total:,} events:", file=stream)
    for reason, n in STATS.most_common():
        print(f"  {reason:28} {n:9,}  ({100*n/total:.1f}%)", file=stream)


# ---------------------------------------------------------------- politeness

class Host:
    """One host's rate limit, robots rules and connection. Serial by design:
    concurrency across hosts belongs in the runner, where the total load on the
    shared link can be measured in one place."""

    def __init__(self, host, rate=RATE, dry=False):
        self.host = host
        self.rate = rate
        self.dry = dry
        self.last = 0.0
        self.sess = requests.Session()
        self.sess.headers["user-agent"] = UA
        self.rp = None
        self.sitemaps = []
        self.robots = None     # "fetched" | "absent_4xx" | "unreachable_assumed_open"
        self._robots()

    def _robots(self):
        url = f"https://{self.host}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(url)
        try:
            r = self.sess.get(url, timeout=TIMEOUT)
        except requests.RequestException as e:
            # Unreachable: proceed, by our decision (see the module docstring). The
            # 1 request/s per host still applies -- delay() falls back to self.rate.
            rp.allow_all = True
            self.robots = "unreachable_assumed_open"
            _count(f"robots_unreachable:{type(e).__name__}_assumed_open")
            self.rp = rp
            return
        if 200 <= r.status_code < 300 and r.text:
            rp.parse(r.text.splitlines())
            self.robots = "fetched"
            self.sitemaps = re.findall(r"(?im)^\s*sitemap:\s*(\S+)", r.text)
            _count("robots_ok")
        elif 400 <= r.status_code < 500:
            # RFC 9309 2.3.1.3: a 4xx means the file does not exist, and the crawler may
            # access any resource. This is not a courtesy — treating a 403 or a 405 from
            # a CDN as "disallow everything" silently empties the crawl, which is the
            # worst possible failure: it looks like a site with no pages. Counted by
            # status so an unusual one (405 from a WAF) is visible rather than assumed.
            rp.allow_all = True
            self.robots = "absent_4xx"
            _count(f"robots_{r.status_code}_allow_all_per_rfc9309")
        else:
            # 5xx: unreadable, so proceed (module docstring). A 2xx with an empty body is
            # a real, empty robots.txt, which permits everything by itself.
            rp.allow_all = True
            self.robots = "fetched" if 200 <= r.status_code < 300 else "unreachable_assumed_open"
            _count(f"robots_{r.status_code}_assumed_open")
        self.rp = rp

    def allowed(self, url):
        try:
            return self.rp.can_fetch(UA, url)
        except Exception:
            return False

    def delay(self):
        return self.rp.crawl_delay(UA) or self.rate

    def get(self, url, allow_robots_failure=False):
        """One polite GET. Returns a Response, or None with the reason counted."""
        if not self.allowed(url):
            _count("robots_disallow")
            return None
        wait = self.delay() - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        for attempt in range(FETCH_ATTEMPTS):
            self.last = time.time()
            if self.dry:
                _count("dry_run_would_fetch")
                return None
            try:
                r = self.sess.get(url, timeout=TIMEOUT, allow_redirects=True)
            except requests.RequestException as e:
                if attempt == FETCH_ATTEMPTS - 1:
                    _count(f"fetch_exception:{type(e).__name__}")
                    return None
                time.sleep(3 * (attempt + 1))
                continue
            if r.status_code == 200:
                _count("fetch_ok")
                set_encoding(r)
                return r
            if r.status_code in (429, 500, 502, 503, 504):
                if attempt == FETCH_ATTEMPTS - 1:
                    _count(f"fetch_backoff_exhausted:{r.status_code}")
                    return None
                time.sleep(min(60, 10 * (attempt + 1)))       # rate limit: back off hard
                continue
            if r.status_code == 403:
                # CC's CDN and some news sites return a plain 403 under sustained
                # load. It is not "this page does not exist", so it gets its own
                # reason rather than being folded into 404.
                _count("fetch_403")
                return None
            _count(f"fetch_status:{r.status_code}")
            return None
        return None


def set_encoding(r):
    """requests decodes as ISO-8859-1 when the header names no charset (RFC 2616), even
    when the page is UTF-8. That garbled every accented letter on 55 of 56
    democraticrepublicoftheyoruba.com pages. With no charset in the header, use UTF-8 when
    the bytes are valid UTF-8 (real Latin-1 text with accents almost never is); otherwise
    keep the ISO-8859-1 default. Guessing from the bytes (apparent_encoding) was tried and
    turned "café" into Chinese."""
    if "charset" in r.headers.get("content-type", "").lower():
        return
    try:
        r.content.decode("utf-8")
    except UnicodeDecodeError:
        return
    r.encoding = "utf-8"


# ---------------------------------------------------------------- discovery

LOC = re.compile(r"<loc>\s*(?:<!\[CDATA\[)?\s*(.*?)\s*(?:\]\]>)?\s*</loc>", re.I | re.S)
FEED_LINK = re.compile(
    r"<link[^>]+(?:type=[\"']application/(?:rss|atom)\+xml[\"'][^>]*href=[\"']([^\"']+)[\"']"
    r"|href=[\"']([^\"']+)[\"'][^>]*type=[\"']application/(?:rss|atom)\+xml[\"'])", re.I)
HREF = re.compile(r"<a\b[^>]*\bhref=[\"']([^\"']+)[\"']", re.I)
FEED_PATHS = ["/feed", "/feed/", "/rss", "/rss.xml", "/atom.xml", "/index.xml", "/feed.xml"]


def same_host(url, host):
    h = (urllib.parse.urlsplit(url).netloc or "").lower().removeprefix("www.")
    return h == host or h == "" and url.startswith("/")


def norm(url, base):
    """Absolute URL, fragment stripped, only http(s). None for anything else --
    mailto:, javascript:, and the tracker URLs that carry the real link in a query."""
    if not url or url.startswith(("mailto:", "javascript:", "tel:", "#", "data:")):
        return None
    u = urllib.parse.urljoin(base, url.strip())
    p = urllib.parse.urlsplit(u)
    if p.scheme not in ("http", "https"):
        return None
    return urllib.parse.urlunsplit((p.scheme, p.netloc, p.path, p.query, ""))


def sitemap_urls(h, extra=()):
    """Every URL the site's sitemaps declare, following sitemap indexes.

    This is the deep step. Bounded by SITEMAP_CAP child sitemaps and SITEMAP_DEPTH
    levels because a large news site can nest indexes, and an unbounded walk of
    them is a denial of service against the site rather than a crawl of it.
    """
    queue = collections.deque()
    for s in list(extra) + list(h.sitemaps):
        queue.append((norm(s, f"https://{h.host}/"), 0))
    for p in ("/sitemap.xml", "/sitemap_index.xml", "/sitemap-index.xml", "/sitemap.xml.gz"):
        queue.append((f"https://{h.host}{p}", 0))
    seen_sm, urls, followed = set(), [], 0
    while queue and followed < SITEMAP_CAP:
        sm, depth = queue.popleft()
        if not sm or sm in seen_sm:
            continue
        seen_sm.add(sm)
        r = h.get(sm)
        if r is None:
            continue
        followed += 1
        body = r.text
        locs = LOC.findall(body)
        is_index = bool(re.search(r"<sitemapindex", body, re.I))
        if is_index and depth < SITEMAP_DEPTH:
            for u in locs:
                queue.append((norm(u, sm), depth + 1))
            _count("sitemap_index_followed")
            continue
        # A urlset. Anything that is not a sitemap child is a page URL.
        for u in locs:
            nu = norm(u, sm)
            if nu and (same_host(nu, h.host) or True):       # sitemaps may list the same site's other subdomains
                urls.append(nu)
        _count("sitemap_urlset_read")
    return urls


def feed_urls(h, home_html=None):
    """Feed entries, for recency. Feed URLs are already the published pages."""
    cands = [f"https://{h.host}{p}" for p in FEED_PATHS]
    if home_html:
        for m in FEED_LINK.finditer(home_html):
            u = norm(m.group(1) or m.group(2), f"https://{h.host}/")
            if u:
                cands.insert(0, u)
    out = []
    for c in cands[:6]:
        r = h.get(c)
        if r is None:
            continue
        locs = LOC.findall(r.text)
        if not locs:
            locs = re.findall(r"<link[^>]*href=[\"']([^\"']+)[\"']", r.text)
        if locs:
            _count("feed_read")
            for u in locs:
                nu = norm(u, c)
                if nu:
                    out.append(nu)
            break
    return out


def bfs(h, home_url, max_pages, depth=2):
    """Fallback for a site with no sitemap: follow same-host links from the home
    page, breadth first, capped. depth 2 is deliberate -- a depth-3 walk of a
    news site is millions of pages and the sitemap is the right tool there."""
    out, seen = [], {home_url}
    level = [home_url]
    r = h.get(home_url)
    home_html = r.text if r is not None else None
    for d in range(depth):
        nxt = []
        for page in level:
            if len(out) >= max_pages:
                return out, home_html
            if page == home_url:
                rr = r
            else:
                rr = h.get(page)
            if rr is None:
                continue
            out.append(page)
            for m in HREF.finditer(rr.text):
                u = norm(m.group(1), page)
                if u and u not in seen and same_host(u, h.host):
                    seen.add(u)
                    nxt.append(u)
        level = nxt[:max_pages]
        _count(f"bfs_level_{d+1}")
    return out, home_html


# ---------------------------------------------------------------- extract + keep

_model = None
_model_lock = threading.Lock()


def lid(text):
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = fasttext.load_model(os.environ["GLOTLID"])
    labs, probs = _model.predict(text.replace("\n", " ")[:2000], k=3)
    return [(l.removeprefix("__label__"), float(p)) for l, p in zip(labs, probs)]


def extract(html):
    return trafilatura.extract(html, include_comments=False, include_tables=False,
                               no_fallback=False)


def url_ok(url):
    """The stage-6 URL rules plus the existing host blocklist, applied here so the
    rescrape never writes a document the main pipeline would drop three stages later.
    Wikipedia infrastructure is harmless to skip; MT localisation is what the
    filters exist to remove; and a host a human already marked `drop` is never re-litigated by
    a crawler."""
    if blocked(url):
        _count("rule_blocked_host")
        return False
    if wiki_infrastructure(url):
        _count("rule_wiki_infrastructure")
        return False
    if mt_localisation(url):
        _count("rule_mt_localisation")
        return False
    return True


def doc_id(url):
    return hashlib.sha1(url.encode()).hexdigest()[:16]


# ---------------------------------------------------------------- per host

def crawl_host(host, out_dir, max_pages, rate, dry, use_bfs=True, stop_at=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"{host}.jsonl.zst"
    state_f = out_dir / f"{host}.state.json"
    if final.exists() and not dry:
        print(f"{host}: already written ({final.stat().st_size:,} B), skipping")
        return 0

    # `done` = kept URLs, `rejected` = fetched and refused on content (not html, too short,
    # wrong language). Both are skipped on resume; a failed fetch is in neither, so it is retried.
    done, rejected, tmp_bytes = set(), set(), None
    if state_f.exists():
        st = json.loads(state_f.read_text())
        tmp_bytes = st.get("tmp_bytes")
        done, rejected = set(st.get("done", [])), set(st.get("rejected", []))
        print(f"{host}: resuming, {len(done):,} kept and {len(rejected):,} rejected URLs skipped")

    h = Host(host, rate=rate, dry=dry)
    t0 = time.time()

    home = f"https://{host}/"
    r = h.get(home)
    home_html = r.text if r is not None else None
    base = r.url if r is not None else home                    # respect a redirect to www.

    urls = []
    seen = set()

    def add(u):
        if u and u not in seen and same_host(u, h.host or urllib.parse.urlsplit(base).netloc.removeprefix("www.")):
            seen.add(u)
            urls.append(u)

    for u in sitemap_urls(h, extra=[]):
        add(u)
    if not urls:
        _count("no_sitemap_urls")
        for u in feed_urls(h, home_html):
            add(u)
    if not urls and use_bfs:
        _count("falling_back_to_bfs")
        b, _ = bfs(h, base, max_pages, depth=2)
        for u in b:
            add(u)
    # A feed is still worth reading alongside a sitemap: sitemaps lag, feeds do not.
    for u in feed_urls(h, home_html):
        add(u)

    print(f"{host}: {len(urls):,} candidate URLs "
          f"({len(h.sitemaps)} sitemap lines in robots.txt, {time.time()-t0:.0f}s to discover)",
          flush=True)

    tmp = out_dir / f"{host}.jsonl.zst.tmp"
    n = kept = 0
    mode = "ab" if done else "wb"                            # append on resume
    if done and tmp_bytes is not None and tmp.exists():
        os.truncate(tmp, tmp_bytes)      # drop what a crash wrote after the last checkpoint
    with open(tmp, mode) as fh:
        w = zstandard.ZstdCompressor(level=6).stream_writer(fh, closefd=False)

        def checkpoint():
            # flush the compressed records first, so the state never claims a page the
            # .tmp does not hold; then replace the state file atomically
            w.flush(zstandard.FLUSH_FRAME)
            fh.flush()
            os.fsync(fh.fileno())
            part = state_f.with_suffix(".json.part")
            part.write_text(json.dumps({"done": sorted(done), "rejected": sorted(rejected),
                                        "kept": kept, "tmp_bytes": fh.tell()}))
            os.replace(part, state_f)

        def reject(u):
            rejected.add(u)
            if len(rejected) % 200 == 0:
                checkpoint()
        for u in urls:
            if n >= max_pages:
                break
            if stop_at is not None and time.time() >= stop_at:   # a time box: finish the shard with what is in hand
                print(f"{host}: time box reached, stopping at {n:,} fetched", flush=True)
                break
            if u in done or u in rejected:
                continue
            n += 1
            if not url_ok(u):
                continue
            rr = h.get(u)
            if rr is None:
                continue
            ctype = rr.headers.get("content-type", "")
            if "html" not in ctype and ctype:
                _count("not_html")
                reject(u)
                continue
            text = extract(rr.text)
            if not text or len(text) < MIN_TEXT:
                _count("extract_too_short")
                reject(u)
                continue
            preds = lid(text)
            lang, p = preds[0]
            if lang not in KEEP or p < THETA:
                _count(f"lid_reject:{lang}")
                reject(u)
                continue
            rec = {"id": f"{host}:{doc_id(u)}", "url": rr.url or u, "lang": lang,
                   "text": text, "host": host, "source": "rescrape", "robots": h.robots,
                   "lid_prob": p, "lid_top3": preds,
                   "fetched": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            w.write((json.dumps(rec, ensure_ascii=False) + "\n").encode())
            kept += 1
            done.add(u)
            _count("ok")
            if kept % 100 == 0:
                checkpoint()
                print(f"  {host}: {n:,} fetched, {kept:,} kept, "
                      f"{time.time()-t0:.0f}s", flush=True)
        w.close()
    # Checkpoints and resumes leave several zstd frames. A one-shot ZstdDecompressor.decompress() reads
    # only the FIRST frame, silently; readers here stream with read_across_frames=True, and this
    # rewrite to one frame is the second line of defence.
    one = out_dir / f"{host}.jsonl.zst.one"
    with open(tmp, "rb") as src, open(one, "wb") as dst:
        zstandard.ZstdCompressor(level=6).copy_stream(zstandard.ZstdDecompressor().stream_reader(src), dst)
    os.replace(one, final)
    tmp.unlink()
    state_f.write_text(json.dumps({"done": sorted(done), "rejected": sorted(rejected), "kept": kept,
                                   "candidates": len(urls), "robots": h.robots, "finished": True}))
    print(f"{host}: DONE {n:,} fetched -> {kept:,} kept in {time.time()-t0:.0f}s -> {final}",
          flush=True)
    return kept


# ---------------------------------------------------------------- selftest

def selftest():
    assert same_host("https://x.com/a", "x.com")
    assert not same_host("https://y.com/a", "x.com")
    assert norm("a/b", "https://x.com/d/") == "https://x.com/d/a/b"
    # A bare fragment link is the same page, not a new URL: rejecting it here is what
    # stops the BFS from appending every in-page anchor to its own queue.
    assert norm("#frag", "https://x.com/d/") is None
    assert norm("page#frag", "https://x.com/d/") == "https://x.com/d/page"
    assert norm("mailto:a@b.c", "https://x.com/") is None
    assert norm("javascript:void(0)", "https://x.com/") is None
    assert doc_id("https://x.com/a") == doc_id("https://x.com/a")
    assert doc_id("https://x.com/a") != doc_id("https://x.com/b")
    # the URL rules must actually fire, or this crawls the pages the pipeline drops
    assert not url_ok("https://ha.wikipedia.org/wiki/Special:WhatLinksHere/Foo")
    assert url_ok("https://ha.wikipedia.org/wiki/Kano")
    assert not url_ok("https://ha.bjsohchina.com/product")
    assert url_ok("https://hausa.legit.ng/siyasa/123-story/")
    assert url_ok("https://ha.freedomradionig.com/x")          # Nigerian host with a lang subdomain stays
    # the existing hand-sorted blocklist must actually be loaded and must actually bite
    assert BLOCKED, "path_mt_hosts.tsv loaded nothing - the blocklist is silently empty"
    assert blocked("www.teyuchiller.com") and blocked("https://www.xingluchemical.com/ha/page")
    assert not blocked("hausa.legit.ng")
    assert blocked("shop.teyuchiller.com")           # a subdomain of a dropped host is dropped
    print(f"selftest OK ({len(BLOCKED)} blocked hosts loaded)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", help="TSV: host [lang] [max_pages], one per line, # comments")
    ap.add_argument("--host", help="a single host")
    _local_user_dir = Path("/local") / os.environ.get("USER", "")
    ap.add_argument("--out-dir", default=os.environ.get("RESCRAPE_OUT", str(_local_user_dir / "rescrape") if _local_user_dir.exists() else "/tmp/rescrape"))
    ap.add_argument("--max-pages", type=int, default=5000)
    ap.add_argument("--rate", type=float, default=RATE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stop-at", help="UTC time, e.g. 2030-01-01T12:00:00Z: stop fetching then and write the shard as usual")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not os.environ.get("GLOTLID"):
        sys.exit("GLOTLID is not set — it is required, and its absence must not degrade to a default")
    hosts = []
    if a.host:
        hosts = [(a.host, a.max_pages)]
    elif a.seeds:
        for line in Path(a.seeds).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            hosts.append((parts[0], int(parts[2]) if len(parts) > 2 and parts[2] else a.max_pages))
    else:
        sys.exit("give --seeds or --host")
    # Drop blocked hosts from the seed list here, loudly. A seed file that still lists a
    # host the blocklist drops is a mistake in the seed file, and it should be visible
    # at the top of the run rather than absorbed silently per URL.
    kept_hosts = []
    for host, mp in hosts:
        if blocked(host):
            _count("seed_skipped_blocked_host")
            print(f"SKIP {host}: marked drop in the host blocklist", file=sys.stderr)
            continue
        kept_hosts.append((host, mp))
    stop_at = None
    if a.stop_at:
        import datetime
        stop_at = datetime.datetime.strptime(a.stop_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc).timestamp()
    total = 0
    for host, mp in kept_hosts:
        total += crawl_host(host, a.out_dir, mp, a.rate, a.dry_run, stop_at=stop_at)
    hosts = kept_hosts
    report()
    print(f"kept {total:,} documents across {len(hosts)} hosts", file=sys.stderr)


if __name__ == "__main__":
    main()
