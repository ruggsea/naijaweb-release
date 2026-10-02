"""The re-crawl seed list: curated Nigerian outlets, minus hosts that must not be crawled deeply.

The seeds are the outlets a Hausa/Igbo/Yorùbá reader would name (CURATED below), kept in one place so a missing outlet is a
missing line. Excluded:

  * Wikipedia and Wiktionary: already over-represented in every corpus; they stay where the crawl found them but are
    not sought out.
  * Jehovah's Witnesses and Bible sites: their keep-rates are about the publisher, not the writing.
  * Everything in filter/path_mt_hosts.tsv marked `drop`: a deep crawl of a machine-translated commercial site would
    manufacture exactly the text the filters are there to remove.

  python build/rescrape_seeds.py                     # write data/rescrape_seeds.tsv
  python build/rescrape_seeds.py --dry-run
"""
import argparse, json, re, sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent.parent
from rescrape import load_blocklist, _canon_host, blocked, BLOCKED

OUT = ROOT / "data" / "rescrape_seeds.tsv"

# The outlets a reader of these languages would name.
# VOA Hausa, BBC's three language services, Freedom Radio Kano, Aminiya, Legit.ng's
# language editions, TRT Afrika, Global Voices, DW, and the Yorùbá/Igbo press..
# lang is a hint only — GlotLID decides on the text.
CURATED = [
    ("hausa.legit.ng", "hau"), ("hausa.premiumtimesng.com", "hau"),
    ("hausa.leadership.ng", "hau"), ("voahausa.com", "hau"),
    ("ha.freedomradionig.com", "hau"), ("freedomradionig.com", "hau"),
    ("aminiya.ng", "hau"), ("hausa.cri.cn", "hau"),
    ("hausa.bbc.com", "hau"), ("bbc.com", "hau"),
    ("dw.com", "hau"), ("amp.dw.com", "hau"),
    ("trtafrika.com", "hau"), ("ha.trtafrika.com", "hau"),
    ("globalvoices.org", "hau"), ("ha.globalvoices.org", "hau"),
    ("kadaura24.com", "hau"), ("idongari.com", "hau"), ("dabofm.com", "hau"),
    ("arewaradio.com", "hau"), ("hutudole.com", "hau"), ("arewa.ng", "hau"),
    ("hausamini.com.ng", "hau"), ("solacebasehausa.com", "hau"),
    ("alfijirnews.com", "hau"), ("gtahausa.com", "hau"),
    ("legit.ng", "hau"), ("dailytrust.com", "hau"), ("premiumtimesng.com", "hau"),
    ("punchng.com", "hau"), ("vanguardngr.com", "hau"), ("channelstv.com", "hau"),
    ("ig.legit.ng", "ibo"), ("bbc.com/igbo", "ibo"),
    ("ig.globalvoices.org", "ibo"), ("lawandmore.com.ng", "ibo"),
    ("owellefm.org", "ibo"), ("imewemmewe.com", "ibo"),
    ("yoruba.legit.ng", "yor"), ("bbc.com/yoruba", "yor"),
    ("yo.globalvoices.org", "yor"), ("iweirohinapere.com", "yor"),
    ("asa.ooduarere.com", "yor"), ("itafaaji.com", "yor"),
    ("iroyinowuro.com.ng", "yor"), ("alaroye.org", "yor"),
    ("yorubanation.com", "yor"), ("akanko.com", "yor"),
]

# Hosts that are never seeds, by name. Wider than the blocklist: these are already
# over-represented, or not a publisher, rather than known machine translation.
NEVER = re.compile(
    r"^(?:[\w-]+\.)*(?:wikipedia|wiktionary|wikimedia|wikidata|wikisource|wikiquote|"
    r"wikinews|wikivoyage)\.org$|"
    r"^(?:[\w-]+\.)*jw\.org$|"
    r"^(?:[\w-]+\.)*bible\w*\.\w+$|"
    r"^(?:[\w-]+\.)*(?:wordpress|blogspot|medium|facebook|twitter|x|youtube|instagram|"
    r"tiktok|reddit|pinterest|linkedin|telegram|whatsapp)\.\w+$", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    cand = {}
    for host, lang in CURATED:
        host = _canon_host(host.split("/")[0])       # curated entries may carry a path hint
        cand.setdefault(host, lang)

    dropped = {"blocked": [], "never": []}
    seeds = []
    for host, lang in sorted(cand.items()):
        if blocked(host):
            dropped["blocked"].append(host)
            continue
        if NEVER.match(host):
            dropped["never"].append(host)
            continue
        seeds.append((host, lang))

    per_lang = {}
    for _, lang in seeds:
        per_lang[lang] = per_lang.get(lang, 0) + 1

    print(f"seeds {len(seeds)}  (curated {len(cand)})", file=sys.stderr)
    print(f"  by language hint: {per_lang}", file=sys.stderr)
    print(f"  dropped as blocked (path_mt_hosts.tsv): {len(dropped['blocked'])}", file=sys.stderr)
    print(f"  dropped as never-seed (wiki/JW/Bible/social): {len(dropped['never'])}", file=sys.stderr)
    for h in dropped["blocked"][:5]:
        print(f"    blocked {h}", file=sys.stderr)
    for h in dropped["never"][:5]:
        print(f"    never   {h}", file=sys.stderr)

    if a.dry_run:
        return

    lines = [
        "# Rescrape seeds. One host per line, TSV:",
        "#   host <TAB> lang-hint <TAB> max_pages",
        "# max_pages is a ceiling per host per run; a sitemap can hold 100k URLs and the",
        "# crawler stops at this number rather than at the end of the site.",
        "#",
        "# Source: the curated media-landscape list in build/rescrape_seeds.py.",
        "#",
        "# Excluded: everything marked `drop` in filter/path_mt_hosts.tsv, and Wikipedia /",
        "# Wiktionary / JW / Bible / social platforms by name.",
        "host\tlang\tmax_pages",
    ]
    for host, lang in seeds:
        lines.append(f"{host}\t{lang}\t10000")
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT} ({len(seeds)} seeds)", file=sys.stderr)


if __name__ == "__main__":
    main()
