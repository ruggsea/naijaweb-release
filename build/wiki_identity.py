"""Identity hold-out for Wikipedia, needed before any training run that puts Wikipedia back.

data/heldout/wiki_<lang>.jsonl holds 500 random articles per language from wikimedia/wikipedia 20231101 (build/heldout.py)
with text only. This script redraws the same sample (same seed, same dump), checks each text matches the file line for
line, and writes data/heldout_ids/wiki_<lang>.jsonl with the article id, title and url beside it. The texts themselves
are not touched, and the ids stay out of data/heldout/, where every *.jsonl is read as held-out text.

Then it counts the held-out articles whose title appears among the Wikipedia pages in our 30-snapshot pool (P_ALL30),
by title taken from the page URL (/wiki/<Title>, index.php?title=<Title>, mobile hosts too). Those articles, and every
page of ours with the same title, are what a Wikipedia-back run must exclude; the count is printed per language, and
the matching pool ids go to data/final/wiki_heldout_ids.txt. Needs HF_HOME pointing at the repo (where the dump is cached).
"""
import glob, io, json, random
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config.py
from config import DATA
from urllib.parse import urlparse, parse_qs, unquote
import zstandard
from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
V = DATA / "wet/versions"
LANGS = {"hau": "ha", "ibo": "ig", "yor": "yo"}
MAX_DOCS = 500

def norm_title(t): return unquote(t).replace("_", " ").strip().casefold()

def title_of(url):
    u = urlparse(url)
    if u.path.startswith("/wiki/"): return norm_title(u.path[len("/wiki/"):])
    if "title" in parse_qs(u.query): return norm_title(parse_qs(u.query)["title"][0])
    return None

ours = {iso2: {} for iso2 in LANGS.values()}   # title -> pool ids
for p in sorted(glob.glob(str(V / "P_ALL30/part_*.jsonl.zst"))):
    with open(p, "rb") as fh:
        for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh, read_across_frames=True), encoding="utf-8"):
            r = json.loads(line)
            host = urlparse(r["url"]).netloc
            if not host.endswith("wikipedia.org"): continue
            iso2, t = host.split(".")[0], title_of(r["url"])
            if iso2 in ours and t: ours[iso2].setdefault(t, []).append(r["id"])

excluded = []
for lang, iso2 in LANGS.items():
    d = load_dataset("wikimedia/wikipedia", f"20231101.{iso2}", split="train")
    idx = random.Random(0).sample(range(len(d)), min(MAX_DOCS, len(d)))
    kept = [d[i] for i in idx if d[i]["text"] and len(d[i]["text"].strip()) > 40]   # heldout.py's write() filter
    held = [json.loads(l)["text"] for l in open(ROOT / f"data/heldout/wiki_{lang}.jsonl")]
    assert [a["text"].strip() for a in kept] == held, f"{lang}: redrawn sample does not match the held-out file"
    out = ROOT / f"data/heldout_ids/wiki_{lang}.jsonl"
    out.parent.mkdir(exist_ok=True)
    out.with_suffix(".tmp").write_text("".join(json.dumps({"id": a["id"], "title": a["title"], "url": a["url"]}) + "\n" for a in kept))
    out.with_suffix(".tmp").rename(out)
    hits = [a for a in kept if norm_title(a["title"]) in ours[iso2]]
    ids = [i for a in hits for i in ours[iso2][norm_title(a["title"])]]
    excluded += ids
    print(f"{lang}: {len(held)} held-out articles, {len(ours[iso2]):,} titles in our pool, "
          f"{len(hits)} held-out articles also in our pool ({len(ids)} pool pages)", flush=True)

(ROOT / "data/final").mkdir(exist_ok=True)
tmp = ROOT / "data/final/wiki_heldout_ids.txt.tmp"
tmp.write_text("".join(i + "\n" for i in sorted(set(excluded))))
tmp.rename(ROOT / "data/final/wiki_heldout_ids.txt")
print(f"pool pages to exclude: {len(set(excluded))}")
