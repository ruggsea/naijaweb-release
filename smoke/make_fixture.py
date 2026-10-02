"""Builds the smoke test's WET-LID-shaped fixture from smoke/fixture/sampled_text.jsonl (75
short public Wikipedia excerpts, hau/ibo/yor, sampled once from data/heldout -- see that file's
own header). Expands them into ~275 synthetic "collected" records: each base text becomes a few
pages on a synthetic host, with a shared boilerplate menu prepended/appended (to exercise the
CCNet-style boilerplate removal in filter/apply_filters.py) and, for every third text, an exact
duplicate and a one-word-changed near-duplicate (to exercise filter/dedup.py's MinHash).

Writes one zstd-compressed jsonl shard to $NAIJAWEB_DATA/wet/collected/SMOKE_PART0.jsonl.zst and a
matching data/wet_paths_SMOKE.txt, the two things filter/apply_filters.py needs to find it.
"""
import json, os, sys
from pathlib import Path
import zstandard

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "smoke/fixture/sampled_text.jsonl"
DATA = Path(os.environ["NAIJAWEB_DATA"])

MENUS = {
    "site0.example.ng": "Gida | Labarai | Wasanni | Tuntube mu",
    "site1.example.ng": "Home | News | Sports | Contact us",
    "site2.example.ng": "Oyi | Akụkọ | Egwuregwu | Kpọtụrụ anyị",
}
FOOTER = "Copyright 2026. Duk hakkoki an kiyaye su."


def pages():
    recs = [json.loads(l) for l in open(FIXTURE)]
    n = 0
    for i, r in enumerate(recs):
        host = f"site{i % 3}.example.ng"
        menu = MENUS[host]
        body = r["text"]

        def make(text, suffix):
            nonlocal n
            n += 1
            wet_text = f"{menu}\n{text}\n{FOOTER}"
            return {
                "url": f"https://{host}/article-{i}{suffix}",
                "lang": r["lang"],
                "wet_text": wet_text,
                "wet_file": f"crawl-data/CC-MAIN-2024-99/segments/1/wet/{host}-{i}{suffix}.warc.wet.gz",
            }

        # the base page, plus two more pages of the SAME host sharing the same menu/footer lines
        # (BOILER_PAGES=3 in apply_filters.py -- three pages is exactly enough to trigger it)
        yield make(body, "")
        yield make(body[: max(10, len(body) // 2)] + " " + body, "-b")
        yield make(" ".join(reversed(body.split())), "-c")

        if i % 3 == 0:
            yield make(body, "-dup")                                    # exact duplicate
        if i % 3 == 1 and len(body.split()) > 5:
            words = body.split()
            words[len(words) // 2] = "CHANGED"
            yield make(" ".join(words), "-near")                        # near-duplicate, one word


def main():
    out = list(pages())
    dst = DATA / "wet/collected"
    dst.mkdir(parents=True, exist_ok=True)
    shard_name = "SMOKE_PART0.jsonl.zst"
    blob = ("\n".join(json.dumps(r, ensure_ascii=False) for r in out) + "\n").encode()
    (dst / shard_name).write_bytes(zstandard.ZstdCompressor().compress(blob))

    paths_file = ROOT / "data/wet_paths_SMOKE.txt"
    paths_file.parent.mkdir(parents=True, exist_ok=True)
    # apply_filters.py derives the shard filename from this by replacing .warc.wet.gz -> .jsonl.zst,
    # so the basename here must match shard_name with that suffix swapped back.
    paths_file.write_text("crawl-data/CC-MAIN-2024-99/segments/1/wet/" + shard_name.replace(".jsonl.zst", ".warc.wet.gz") + "\n")

    print(f"wrote {len(out)} synthetic documents to {dst / shard_name}")


if __name__ == "__main__":
    main()
