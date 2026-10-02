"""Mechanical filters: the two URL rules, and how they score on the labels.

The dominant failure mode of this corpus is not the one an English pipeline is
built to catch. Two patterns carry it, both decidable
from the URL alone, and both invisible to a human rater:

  wiki infrastructure  -- WhatLinksHere, Special:, ?action=edit&redlink=1 stubs and
    their per-language equivalents. A URL rule and deliberately not a host rule:
    ha.wikipedia.org keeps at 32%, but that is a page mix, and dropping the host
    would lose the real articles with it.

  MT localisation      -- a ha./ig./yo. subdomain on a domain that is neither
    Nigerian nor a wiki. Foreign commercial sites with a machine-translated
    localisation. The codebook has no machine-translation key, so fluent
    translationese reads as key 1 and annotators keep it ABOVE the base rate.
    That is the whole reason this one has to be mechanical.

Run with no arguments to print the measurement over the labelled sets. Give it
jsonl files (records with a `url`) to filter, and it writes the survivors to
stdout and the reasons to stderr.
"""
import json, re, sys, unicodedata
from pathlib import Path
from urllib.parse import urlparse, unquote

ROOT = Path(__file__).resolve().parent.parent

# MediaWiki namespaces and actions in English and in the three languages' wikis.
# Matched on the decoded path + query, case-insensitively.
# MediaWiki namespaces and actions, in English and in the three wikis' own language
# forms. A namespace whitelist and not a generic "colon in the title": article titles
# carry colons too (ISO_3166-2:NG, dated titles), and a colon rule would take them.
# The language forms were read off the labelled set (e.g. `Tattaunawar user:`, `Oníṣe:`, `Pàtàkì:`, `Samfuri:`).
WIKI_INFRA = re.compile(
    r"/wiki/(?:"
    r"special|musamman|musanm[aà]n|k[eẹ]b[aà]ntacce|p[uụ]r[uụ]iche|p[aà]t[aà]k[iì]|"   # Special:
    r"user|mai[_ ]amfani|oj[iì]ar[uụ]|on[ií][sṣ]e|"                                     # User:
    r"user[_ ]talk|tattaunawar[_ ]\w+|okwu|[oọ][\u0300-\u0304]?r[oọ][\u0300-\u0304]?[_ ]?on[ií][sṣ]e|hira|"  # talk:
    r"template|templeeti|samfuri|[uụ]kp[uụ]r[uụ]|[aà]d[aà]k[oọ]|"                       # Template:
    r"category|rukuni|[uụ]d[iị]|[eẹ][\u0300-\u0304]?ka|"                              # Category:
    r"file|fayil|[eè]b[uú]t[eé]|"                                                        # File:/Portal:
    r"wikipedia|wiktionary|mediawiki|help|taimako|module"
    r"):"
    r"|whatlinkshere|olonaayeboyi|contributions|gudunmuwa"
    r"|\baction=(?:edit|history|raw|info|purge)\b|\bredlink=1\b|\boldid=|\bdiff=|\bprintable=yes\b",
    re.I)

WIKI_HOST = re.compile(r"(?:^|\.)(?:wikipedia|wiktionary|wikimedia|wikidata|wikisource|wikiquote)\.org$", re.I)

# A localisation subdomain is only evidence of machine translation when the domain
# under it is foreign. A Nigerian domain with a language subdomain is a Nigerian
# outlet publishing in that language, which is exactly what we are looking for.
LANG_SUB = re.compile(r"^(?:ha|ig|yo|hau|ibo|yor)\.", re.I)
NIGERIAN_TLD = re.compile(r"\.ng$", re.I)

# Nigerian outlets that sit on a generic TLD, so the .ng test alone would take them.
# E.g. `ha.freedomradionig.com` is Freedom Radio Kano, not machine translation. Seeded from the hosts
# annotators kept; NOT complete: it needs a human pass over the corpus's lang-subdomain hosts.
NIGERIAN_HOSTS = {
    "freedomradionig.com", "legit.ng", "trtafrika.com", "bbc.com", "bbc.co.uk",
    "voahausa.com", "dw.com", "globalvoices.org", "aminiya.com", "dailytrust.com",
    "punchng.com", "vanguardngr.com", "premiumtimesng.com", "channelstv.com",
}


def host_of(url):
    return (urlparse(url).netloc or "").split(":")[0].lower().removeprefix("www.")


def wiki_infrastructure(url):
    p = urlparse(url)
    if not WIKI_HOST.search(host_of(url)):
        return False
    return bool(WIKI_INFRA.search(unquote(p.path + "?" + p.query)))


def mt_localisation(url):
    h = host_of(url)
    parent = h.split(".", 1)[1] if "." in h else h
    if WIKI_HOST.search(h) or NIGERIAN_TLD.search(h) or parent in NIGERIAN_HOSTS:
        return False
    return bool(LANG_SUB.match(h))


# The same localisation, done as a folder: example.com/yo/... Hand-sorted on the 634
# labelled hosts (filter/path_mt_hosts.tsv): 601 are foreign shops and tools, 33 are
# real publishers (jw.org, IQNA, Pars Today, DW...). A host not in the keep list is
# dropped, so new foreign shops in the full corpus are caught without a new pass.
LANG_PATH = re.compile(r"^/(?:ha|ig|yo|hau|ibo|yor)(?:[-_][a-z]{2,4})?(?:/|$)", re.I)
PATH_KEEP_HOSTS = {host_of("http://" + line.split("\t")[0])
                   for line in open(ROOT / "filter/path_mt_hosts.tsv", encoding="utf-8")
                   if "\tkeep\t" in line}


def mt_path(url):
    h = host_of(url)
    if WIKI_HOST.search(h) or NIGERIAN_TLD.search(h) or h in PATH_KEEP_HOSTS:
        return False
    if any(h == n or h.endswith("." + n) for n in NIGERIAN_HOSTS):
        return False
    return bool(LANG_PATH.match(urlparse(url).path))


# --- Gopher (Rae et al. 2021, Appendix A.1.1, pp. 40-41), 
# The quality rules and the Table A1 repetition limits, exactly as printed, minus
# one: the stop-word rule needs two of eight ENGLISH words and would drop every
# Hausa, Igbo and Yoruba document. Its African equivalent is WURA's (Oladipo et
# al. 2023, s2.1.1): at least 5 stopwords from the Kaggle "Stopword Lists for
# African Languages" set -- wura_stopwords below.
# Words are whitespace tokens; each check returns the name of the first limit a
# document breaks, so the report can say WHICH Gopher rule fails to transfer.
BULLETS = ("•", "-", "*", "·", "‣", "◦", "–")
ELLIPSES = ("...", "…")


def gopher_quality(text):
    words = text.split()
    n = len(words)
    if not 50 <= n <= 100_000:
        return "word_count"
    if not 3 <= sum(map(len, words)) / n <= 10:
        return "mean_word_length"
    if text.count("#") / n > 0.1 or sum(text.count(e) for e in ELLIPSES) / n > 0.1:
        return "symbol_ratio"
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if sum(l.startswith(BULLETS) for l in lines) > 0.9 * len(lines):
        return "bullet_lines"
    if sum(l.endswith(ELLIPSES) for l in lines) > 0.3 * len(lines):
        return "ellipsis_lines"
    if sum(any(c.isalpha() for c in w) for w in words) < 0.8 * n:
        return "alphabetic_words"
    return None


def _dup_fraction(units):
    """Fraction of units that repeat an earlier one, and fraction of characters in them."""
    seen, dup, dup_chars = set(), 0, 0
    for u in units:
        if u in seen:
            dup += 1
            dup_chars += len(u)
        seen.add(u)
    total = sum(map(len, units)) or 1
    return dup / max(len(units), 1), dup_chars / total


def _ngram_char_fractions(words, n):
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return grams, sum(map(len, words)) or 1


def gopher_repetition(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    lf, lc = _dup_fraction(lines)
    pf, pc = _dup_fraction(paras)
    if lf > 0.30: return "dup_line_fraction"
    if pf > 0.30: return "dup_paragraph_fraction"
    if lc > 0.20: return "dup_line_chars"
    if pc > 0.20: return "dup_paragraph_chars"
    words = text.split()
    for n, limit in ((2, 0.20), (3, 0.18), (4, 0.16)):
        grams, total = _ngram_char_fractions(words, n)
        if not grams:
            break
        counts = {}
        for g in grams:
            counts[g] = counts.get(g, 0) + 1
        top, c = max(counts.items(), key=lambda kv: kv[1])
        if c > 1 and c * sum(map(len, top)) / total > limit:
            return f"top_{n}gram_chars"
    for n, limit in zip(range(5, 11), (0.15, 0.14, 0.13, 0.12, 0.11, 0.10)):
        grams, total = _ngram_char_fractions(words, n)
        first, covered = {}, [False] * len(words)
        for i, g in enumerate(grams):
            if g in first:
                for j in range(i, i + n):
                    covered[j] = True
            else:
                first[g] = i
        if sum(len(w) for w, c in zip(words, covered) if c) / total > limit:
            return f"dup_{n}gram_chars"
    return None


# --- WURA (Oladipo et al. 2023, s2.1.1): drop documents with fewer than 5 stopwords.
# Lists: Kaggle rtatman/stopword-lists-for-african-languages, data/stopwords/{ha,yo}.txt.
# It has no Igbo list, so Igbo documents pass untouched. Matching ignores tone marks:
# undiacritised Yoruba is common on the web, and exact matching drops 113 labelled
# Yoruba docs where tone-blind matching drops 38 (humans kept 27% and 26% of them).
def _bare(w):
    return unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", w)
                                                if not unicodedata.combining(c)))


STOPWORDS = {lang: {_bare(w.strip().lower()) for w in open(ROOT / f"data/stopwords/{f}.txt", encoding="utf-8") if w.strip()}
             for lang, f in [("hau", "ha"), ("yor", "yo")]}


def wura_stopwords(text, lang):
    if lang not in STOPWORDS:
        return False
    words = (_bare(w.strip(".,;:!?\"'()[]“”‘’").lower()) for w in text.split())
    return sum(w in STOPWORDS[lang] for w in words) < 5


FILTERS = [
    ("wiki_infrastructure", lambda r: wiki_infrastructure(r["url"])),
    ("mt_localisation", lambda r: mt_localisation(r["url"])),
    ("mt_path", lambda r: mt_path(r["url"])),
    ("gopher_quality", lambda r: gopher_quality(r["text"])),
    ("gopher_repetition", lambda r: gopher_repetition(r["text"])),
    ("wura_stopwords", lambda r: wura_stopwords(r["text"], r["lang"])),
]


def reason(rec):
    """The first filter that rejects this record, or None to keep it. Order is the
    order of FILTERS: each rule changes the statistics the next is computed on."""
    for name, f in FILTERS:
        why = f(rec)
        if why:
            return name if why is True else f"{name}:{why}"
    return None


def _labels_dir():
    """The keep/drop label sets (classifier/build_label_dataset.py's output), released with the corpus."""
    sys.path.insert(0, str(ROOT))
    from config import LABELS, need
    return Path(need(LABELS, "NAIJAWEB_LABELS")) / "dataset"


def report():
    """What each rule removes, and what the humans said about what it removes.

    The number that matters is the keep-rate INSIDE a removed group against the
    base rate. A rule removing documents the raters mostly dropped is only doing
    the raters' work faster; a rule removing documents the raters KEPT is finding
    something the codebook cannot see."""
    for f in sorted(_labels_dir().glob("*.jsonl")):
        rows = [json.loads(l) for l in f.open()]
        base = sum(r["keep"] for r in rows) / len(rows)
        print(f"\n{f.name}  {len(rows)} documents, human keep-rate {base:.1%}")
        # Sequential, in FILTERS order: each rule sees only what the earlier ones left.
        left = rows
        for name, rule in FILTERS:
            why = {id(r): rule(r) for r in left}
            hit = [r for r in left if why[id(r)]]
            left = [r for r in left if not why[id(r)]]
            if not hit:
                print(f"  {name:20s} 0")
                continue
            k = sum(r["keep"] for r in hit) / len(hit)
            arrow = "ABOVE" if k > base else "below"
            print(f"  {name:20s} {len(hit):5d} docs {len(hit)/len(rows):5.1%} "
                  f"· {len(set(r['host'] for r in hit)):3d} hosts "
                  f"· humans kept {k:.1%} ({arrow} the {base:.1%} base rate)")
            by_lang, by_why = {}, {}
            for r in hit:
                by_lang.setdefault(r["lang"], []).append(r["keep"])
                if why[id(r)] is not True:
                    by_why.setdefault(why[id(r)], []).append(r["keep"])
            print("      " + "  ".join(f"{l} {len(v)}@{sum(v)/len(v):.0%}"
                                       for l, v in sorted(by_lang.items())))
            if by_why:
                print("      " + "  ".join(f"{w} {len(v)}@{sum(v)/len(v):.0%}"
                                           for w, v in sorted(by_why.items(), key=lambda kv: -len(kv[1]))))
        k = sum(r["keep"] for r in left) / len(left)
        print(f"  {'survivors':20s} {len(left):5d} docs, humans kept {k:.1%}")


# Cases the rules must catch, and cases they must NOT. The negatives are the half
# that matters: a namespace rule written as "a colon in the title" passes every
# positive here and still eats ISO_3166-2:NG and every dated article title.
SELFTEST = [
    (1, "https://ha.wikipedia.org/wiki/Musamman:WhatLinksHere/Bahi_Ladgham"),
    (1, "https://ha.wikipedia.org/wiki/Tattaunawar_user:Rentangyi"),
    (1, "https://yo.wikipedia.org/wiki/Oníṣe:Testuser"),
    (1, "https://yo.wikipedia.org/wiki/Pàtàkì:Contributions/Someone"),
    (1, "https://ha.wikipedia.org/wiki/Samfuri:Infobox"),
    (1, "https://ig.wikipedia.org/w/index.php?title=Foo&action=edit&redlink=1"),
    (1, "https://ha.wiktionary.org/wiki/Rukuni:Kalmomi"),
    (0, "https://ha.wikipedia.org/wiki/Harshen_Efik"),
    (0, "https://yo.wikipedia.org/wiki/ISO_3166-2:NG"),
    (0, "https://ha.wikipedia.org/wiki/Jonas_Savimbi"),
    (0, "https://punchng.com/some-story/"),
]
MT_SELFTEST = [
    (1, "https://ha.xiesurotomolding.com/products/"),
    (1, "http://ha.bjsohchina.com/bone-cancer-product/"),
    (1, "https://yo.jwtrubber.com/news/"),
    (0, "https://ha.wikipedia.org/wiki/Harshen_Efik"),      # a wiki, not a localisation
    (0, "https://hausa.legit.ng/story"),                     # not a ha. subdomain
    (0, "https://ha.freedomradionig.com/labarai"),           # Nigerian domain, real outlet
    (0, "https://www.trtafrika.com/hausa/story"),
]
PATH_SELFTEST = [
    (1, "https://www.paknpack.com/yo/3-x110yard-16mil-economic-bopp-adhesive-carton-sealing-tapes/"),
    (1, "https://www.roypow.com/ig/rv-ess/solar-panel-product/"),
    (1, "https://www.wxhxh.com/ig/rmr-mini-series/"),             # host starts with w
    (1, "https://unseen-factory.cn/ha-ng/products/"),             # not in the list: dropped
    (0, "https://www.jw.org/yo/Ohun-T%C3%A1-A-N%C3%AD/"),         # keep list
    (0, "https://tabriz.iqna.ir/ha/news/3495106/"),
    (0, "https://www.bbc.com/hausa/labarai"),                     # /hausa is not /ha
    (0, "https://www.yoga.com/yoga-poses/"),                      # /yoga-... is not /yo/
    (0, "https://ha.wikipedia.org/ha/x"),
]


# Text cases. Positives are synthetic; negatives are real documents annotators KEPT, in each
# language's own spelling (Yoruba tone marks, Hausa hooked letters): synthetic negatives test the wrong thing.
_KEPT = {("train", 24674): "hau dabofm.com", ("train", 8265): "yor asa.ooduarere.com",
         ("train", 6309): "ibo owellefm.org"}
GOPHER_SELFTEST = [
    (1, gopher_quality, "Sannu da zuwa"),                                   # 3 words
    (1, gopher_quality, "\n".join(f"• abu {i}" for i in range(60))),         # all bullets
    (1, gopher_quality, " ".join(["1234 5678 ####"] * 30)),                 # no letters
    (1, gopher_repetition, "\n".join(["Kara karanta labarin nan"] * 20)),  # one line repeated
]


def _kept_texts():
    want = {}
    for (split, doc_id) in _KEPT:
        want.setdefault(split, set()).add(doc_id)
    for split, ids in want.items():
        for line in open(_labels_dir() / f"{split}.jsonl"):
            r = json.loads(line)
            if r["doc_id"] in ids:
                yield r


def selftest():
    bad = 0
    for cases, rule in ((SELFTEST, wiki_infrastructure), (MT_SELFTEST, mt_localisation), (PATH_SELFTEST, mt_path)):
        for want, url in cases:
            got = int(rule(url))
            if got != want:
                bad += 1
                print(f"FAIL {rule.__name__} want {want} got {got}: {url}")
    kept = list(_kept_texts())
    assert len(kept) == len(_KEPT), "a selftest document is missing from the label set"
    cases = GOPHER_SELFTEST + [(0, rule, r["text"]) for r in kept
                               for rule in (gopher_quality, gopher_repetition)]
    for want, rule, text in cases:
        got = int(bool(rule(text)))
        if got != want:
            bad += 1
            print(f"FAIL {rule.__name__} want {want} got {got} ({rule(text)}): {text[:60]!r}")
    n = len(SELFTEST) + len(MT_SELFTEST) + len(cases)
    print(f"selftest: {n - bad} passed, {bad} failed")
    return bad


if __name__ == "__main__":
    if len(sys.argv) == 1:
        report()
    elif sys.argv[1] == "--selftest":
        sys.exit(1 if selftest() else 0)
    else:
        dropped = {}
        for path in sys.argv[1:]:
            for line in open(path):
                rec = json.loads(line)
                why = reason(rec)
                if why:
                    dropped[why] = dropped.get(why, 0) + 1
                else:
                    print(line, end="")
        for name, n in dropped.items():
            print(f"{name}\t{n}", file=sys.stderr)
