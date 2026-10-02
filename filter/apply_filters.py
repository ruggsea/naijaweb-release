"""Run the filter/filters.py rules over crawled WET output and count what each removes.

Reads the crawl's files from $NAIJAWEB_DATA/wet/collected/ (fields url, lang, wet_text) and writes
results/filtered_<crawl>.json: documents in, removed per rule (first rule that fires,
in FILTERS order), kept, per language. Kept pages, with the cleaned text in wet_text,
go to $NAIJAWEB_DATA/wet/filtered/<crawl>/ under the same file names (one output per input,
written atomically; the input files are not touched). That folder feeds dedup.py.

Before the text rules, boilerplate is removed CCNet-style: a line that appears on
BOILER_PAGES or more pages of the same host in this crawl is a menu, footer or
sidebar, not content. Then later repeats of the page's first line (its title, which
sites print again as headline and breadcrumb) are removed. Without this, Gopher's
repetition rule dropped 35% of Hausa pages, 4,372 of them real articles.
"""
import json, sys, io
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config.py
from config import DATA
import zstandard
sys.path.insert(0, str(Path(__file__).parent))
import filters as F

BOILER_PAGES = 3


def records(files):
    for f in files:
        with open(f, "rb") as fh:
            for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8"):
                yield json.loads(line)


def content(text, host, line_pages):
    lines = [l for l in text.splitlines() if line_pages[(host, l.strip())] < BOILER_PAGES]
    if not lines:
        return ""
    title = lines[0].strip()
    return "\n".join([lines[0]] + [l for l in lines[1:] if l.strip() != title])


if __name__ == "__main__":
    crawl = sys.argv[1]
    # every host's output for all crawls lands in collected/; pick this crawl by its paths file
    names = [Path(l.strip()).name.replace(".warc.wet.gz", ".jsonl.zst") for l in open(f"data/wet_paths_{crawl}.txt")]
    files = [f for f in (DATA / "wet/collected" / n for n in names) if f.exists()]
    assert files, f"no files for {crawl}"
    line_pages = Counter()
    for r in records(files):
        host = F.host_of(r["url"])
        line_pages.update((host, l) for l in {l.strip() for l in r["wet_text"].splitlines() if l.strip()})
    dst = DATA / "wet/filtered" / crawl
    dst.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    for f in files:
        kept = []
        for r in records([f]):
            lang = r["lang"][:3]
            text = content(r["wet_text"], F.host_of(r["url"]), line_pages)
            why = F.reason({"url": r["url"], "text": text, "lang": lang})
            counts[(lang, (why or "kept").split(":")[0])] += 1
            if not why:
                kept.append(json.dumps({**r, "wet_text": text}, ensure_ascii=False))
        tmp = dst / (f.name + ".tmp")
        tmp.write_bytes(zstandard.ZstdCompressor().compress(("\n".join(kept) + "\n" if kept else "").encode()))
        tmp.replace(dst / f.name)
    out = {"crawl": crawl, "files": len(files), "paths": len(names), "boiler_pages": BOILER_PAGES, "by_lang": {}}
    for (lang, rule), n in sorted(counts.items()):
        out["by_lang"].setdefault(lang, {})[rule] = n
    for lang, d in out["by_lang"].items():
        d["total"] = sum(d.values())
    Path("results").mkdir(exist_ok=True)
    json.dump(out, open(f"results/filtered_{crawl}.json", "w"), indent=1)
    print(json.dumps(out, indent=1))
