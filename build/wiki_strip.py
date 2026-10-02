"""Strip Wikipedia page furniture before SCORING. The corpus text is unchanged.

  python build/wiki_strip.py OUT_DIR [SET DEDUP [IDS]]

Writes OUT_DIR/wikistrip.jsonl.zst: every document of SET (default N, the 18-crawl sample) that survives the
across-crawl dedup of DEDUP (default ALL) and comes from a *.wikipedia.org host, same id, text = strip(text);
with IDS (a file of ids), only those. The shipping corpus uses SET=P_ALL30 DEDUP=ALL30 and the ids
of pages not yet stripped (see build/release_sizes.py for the live equivalent). Pages left empty are not written, so
they keep the score of their original text. classifier/score.py's re-crawl mode scores it.
"""
import io, json, re, sys
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config.py
from config import DATA
from urllib.parse import urlparse
import zstandard
LABEL = re.compile(r"\((en|mul|fr|de|es|ar)\)")
COORD = re.compile(r"\d+°\d*|°[NSEW]\b|\d+\.\d+°[NSEW]")
TOC = re.compile(r"^(\d+(\.\d+)*\s|Toggle\s)")
IMDB = re.compile(r"^nm\d{5,}$|^tt\d{5,}$")
FOOT = re.compile(r'^(Daga "https?://|Anyi gyaran ƙarshe|↑ )')
def strip(text):
    lines = [l for l in text.split("\n") if l.strip()]
    if lines and lines[0].endswith(" - Wikipedia"): lines = lines[1:]
    # everything before the first real sentence (>= 100 chars) is title/infobox furniture
    first = next((i for i, l in enumerate(lines) if len(l) >= 100), len(lines))
    body = lines[first:]
    body = [l for l in body if not (LABEL.search(l) or COORD.search(l) or TOC.match(l) or IMDB.match(l) or FOOT.match(l)
                                    or l.startswith(("Lua error", "Samfuri:", "[[Category")))]
    return "\n".join(body)

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    SET, DEDUP = (sys.argv[2], sys.argv[3]) if len(sys.argv) > 3 else ("N", "ALL")
    only = set(open(sys.argv[4]).read().split()) if len(sys.argv) > 4 else None
    across = {":".join(l.split("\t")[:2]) for l in open(ROOT / f"data/dedup_filtered/{DEDUP}/drop_global.tsv")}
    rows, empty, short = [], 0, 0
    for src in sorted((DATA / "wet/versions" / SET).glob("part_*.jsonl.zst")):
        for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(src, "rb"), read_across_frames=True)):
            r = json.loads(line)
            if r["id"] in across or not (urlparse(r["url"]).hostname or "").endswith("wikipedia.org"): continue
            if only is not None and r["id"] not in only: continue
            t = strip(r["text"])
            if not t: empty += 1; continue
            short += len(t) < 200
            rows.append(json.dumps({"id": r["id"], "url": r["url"], "text": t}, ensure_ascii=False))
    tmp = out / "wikistrip.jsonl.zst.tmp"
    tmp.write_bytes(zstandard.ZstdCompressor().compress(("\n".join(rows) + "\n").encode()))
    tmp.rename(out / "wikistrip.jsonl.zst")
    print(f"{len(rows):,} stripped Wikipedia docs written, {short:,} under 200 chars, {empty:,} empty after stripping (kept original score)")
