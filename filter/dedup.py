"""Near-duplicate removal: MinHash, per snapshot, FineWeb's parameters.

FineWeb (Penedo et al. 2024, s3.4 and App. E.1): word 5-grams,
112 hash functions in 14 bands of 8, which matches pairs >=75% similar with high
probability (77% at s=0.75, 98.8% at s=0.85). Documents sharing all 8 hashes of
any band are duplicates; clusters are transitive; one document per cluster is kept.
Each snapshot is deduplicated on its own: FineWeb found global dedup left the
older snapshots with their worst 10% (s3.4, Fig. 4).

That finding is about English over 96 snapshots, so it is not taken on trust
here: the job also counts clusters that span snapshots, which is what global
dedup would additionally remove. That number decides per-snapshot vs global.

Tokens are lower-cased runs of letters, digits and combining marks. Python's \\w
alone splits Yoruba at every tone mark (they are not alphanumeric), which would
turn 'ọ̀rọ̀' into three tokens and every Yoruba document into near-noise.

  dedup.py sign    -- pass 1: one signature file per chunk of shards, atomic,
                      skipped if present, so a rerun resumes where it died
  dedup.py cluster -- pass 2: union-find per snapshot, writes data/dedup/drop.tsv
  dedup.py --selftest
"""
import io
import hashlib, json, os, re, sys, unicodedata
from multiprocessing import Pool
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config.py
from config import DATA
import numpy as np, zstandard

ROOT = Path(__file__).resolve().parent.parent
# DEDUP_IN / DEDUP_OUT point the job at the filtered pages ($NAIJAWEB_DATA/wet/filtered/<crawl>, data/dedup_filtered)
COLLECTED = Path(os.environ.get("DEDUP_IN", DATA / "wet/collected"))
OUT = Path(os.environ.get("DEDUP_OUT", ROOT / "data/dedup"))
NGRAM, BANDS, ROWS = 5, 14, 8
NHASH = BANDS * ROWS
CHUNKS = 512
MERSENNE = (1 << 61) - 1
_rng = np.random.RandomState(20240617)
A = _rng.randint(1, 1 << 61, NHASH, dtype=np.uint64)
B = _rng.randint(0, 1 << 61, NHASH, dtype=np.uint64)
TOKEN = re.compile(r"[\ẁ-ͯ]+")
CRAWL = re.compile(r"crawl-data/(CC-MAIN-\d{4}-\d{2})/")


def tokens(text):
    return TOKEN.findall(unicodedata.normalize("NFC", text).lower())


def signature(text):
    """112 minhashes over word 5-grams. Short documents use the whole text as one gram."""
    w = tokens(text)
    grams = {" ".join(w[i:i + NGRAM]) for i in range(max(len(w) - NGRAM + 1, 1))}
    h = np.array([int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little") >> 3
                  for g in grams], dtype=np.uint64)
    # (a*h + b) mod p, in uint64 with wraparound; good enough for minhash and fast.
    return ((np.outer(h, A) + B) % np.uint64(MERSENNE)).min(axis=0)


def band_keys(sig):
    return [int.from_bytes(hashlib.blake2b(sig[b * ROWS:(b + 1) * ROWS].tobytes(), digest_size=8).digest(), "little")
            for b in range(BANDS)]


def sign_chunk(i):
    dst = OUT / f"sig_{i:04d}.jsonl"
    if dst.exists():
        return i
    shards = (OUT / "shards.txt").read_text().split()[i::CHUNKS]
    tmp = dst.with_suffix(".tmp")
    dctx = zstandard.ZstdDecompressor()
    with open(tmp, "w") as out:
        for name in shards:
            raw = dctx.stream_reader(io.BytesIO((COLLECTED / name).read_bytes()), read_across_frames=True).read()
            # "\n" only: str.splitlines() also breaks on U+2028 and friends, which
            # sit unescaped inside the JSON strings and cut a record in half.
            for line_no, line in enumerate(l for l in raw.decode().split("\n") if l):
                r = json.loads(line)
                m = CRAWL.search(r["wet_file"])
                out.write(json.dumps([name, line_no, m.group(1) if m else "?", r["lang"],
                                      len(r["wet_text"]), band_keys(signature(r["wet_text"]))]) + "\n")
    os.replace(tmp, dst)
    return i


def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def cluster():
    docs, buckets = [], {}
    for f in sorted(OUT.glob("sig_*.jsonl")):
        for line in open(f):
            shard, line_no, crawl, lang, n, keys = json.loads(line)
            idx = len(docs)
            docs.append((shard, line_no, crawl, lang))
            for b, k in enumerate(keys):
                buckets.setdefault((b, k), []).append(idx)
    assert len(list(OUT.glob("sig_*.jsonl"))) == CHUNKS, "pass 1 is not finished"
    parent = list(range(len(docs)))
    for members in buckets.values():
        for m in members[1:]:
            a, b = find(parent, members[0]), find(parent, m)
            if a != b:
                parent[b] = a
    clusters = {}
    for i in range(len(docs)):
        clusters.setdefault(find(parent, i), []).append(i)
    # Per snapshot: within each global cluster, keep the first document of each crawl.
    # Global would keep one document per cluster; the difference is reported, not applied.
    drop, cross = [], 0
    for members in clusters.values():
        seen = set()
        for i in members:
            if docs[i][2] in seen:
                drop.append(i)
            seen.add(docs[i][2])
        cross += len(seen) - 1
    tmp = OUT / "drop.tsv.tmp"
    with open(tmp, "w") as out:
        for i in drop:
            out.write("\t".join(map(str, docs[i])) + "\n")
    os.replace(tmp, OUT / "drop.tsv")
    # Global: keep one document per cluster, whatever snapshot it came from. Written as a
    # separate list (drop_global.tsv) so the corpus versions can use it; the per-snapshot
    # list above is unchanged.
    tmp = OUT / "drop_global.tsv.tmp"
    with open(tmp, "w") as out:
        for members in clusters.values():
            for i in sorted(members)[1:]:
                out.write("\t".join(map(str, docs[i])) + "\n")
    os.replace(tmp, OUT / "drop_global.tsv")
    # How big are the duplicate clusters? FineWeb's hypothesis is that the gain comes from
    # killing the huge clusters, and that removing clusters smaller than the number of
    # crawls hurts. Written out so that can be checked on this corpus.
    sizes = {}
    for members in clusters.values():
        sizes[len(members)] = sizes.get(len(members), 0) + 1
    (OUT / "cluster_sizes.json").write_text(json.dumps(sizes))
    small = sum(n * c for n, c in sizes.items() if n < 30)
    print(f"documents in clusters of fewer than 30 copies: {small} ({small/len(docs):.1%}); "
          f"in clusters of 30 or more: {len(docs) - small} ({1 - small/len(docs):.1%})")
    by_lang = {}
    for i in drop:
        by_lang[docs[i][3]] = by_lang.get(docs[i][3], 0) + 1
    print(f"{len(docs)} documents, {len(clusters)} clusters")
    print(f"per-snapshot dedup drops {len(drop)} ({len(drop)/len(docs):.1%}): {by_lang}")
    # cross counts, per cluster, the distinct crawls after the first. Run over ONE
    # snapshot every document carries the same crawl label, so this is 0 every time by
    # construction, not a measurement - all 20 single-crawl runs printed "0 (0.0%)" and
    # it reads exactly like the finding that no duplicates span snapshots. The real
    # number only exists for a run over several: the 18-crawl ALL run gives 351,541
    # (39.9%). So say which of the two this is.
    # n_crawls comes from the documents themselves, and the set's .crawls comes from the
    # directories - so a crawl that contributed no surviving document is invisible here,
    # and the line below would name a smaller number of snapshots than the run was built
    # from without anything complaining. Seen on a staged run: a 30-crawl set printed
    # "across the 9 snapshots in this run" because only 15 documents were present. In a
    # real run the two coincide; nothing checked that they did.
    n_crawls = len({d[2] for d in docs})
    crawls_file = Path(str(OUT) + ".crawls")
    if crawls_file.exists():
        named = set(crawls_file.read_text().split())
        seen = {d[2] for d in docs}
        if named != seen:
            print(f"WARNING: the set names {len(named)} crawls but the documents carry "
                  f"{len(seen)} labels; only in the set: {sorted(named - seen)}; "
                  f"only in the documents: {sorted(seen - named)}", flush=True)
    if n_crawls < 2:
        print(f"cross-snapshot duplicates: not measurable here - this run covers one "
              f"snapshot ({docs[0][2]}), so the count is 0 by construction")
    else:
        print(f"global dedup would ALSO drop {cross} ({cross/len(docs):.1%}) that repeat "
              f"across the {n_crawls} snapshots in this run")


def selftest():
    import random
    random.seed(0)
    words = "gwamnati jihar kano ta sanar da cewa za ta fara aikin gina sababbin makarantu".split()
    base = " ".join(random.choice(words) + str(random.randint(0, 999)) for _ in range(400))
    near = base.replace(base.split()[200], "canji", 1)                    # one word changed
    other = " ".join(random.choice(words) + str(random.randint(0, 999)) for _ in range(400))
    same = lambda x, y: len(set(band_keys(signature(x))) & set(band_keys(signature(y)))) > 0
    half = " ".join(base.split()[:200] + other.split()[200:])            # 5-gram Jaccard ~1/3
    yo = "Ọ̀rọ̀ àgbà"
    cases = [(True, same(base, near), "one word changed in 400"),
             (True, same(base, base.upper()), "case only"),
             (False, same(base, other), "unrelated documents"),
             (False, same(base, half), "half shared, below the 0.75 cut"),
             (True, tokens(yo) == ["ọ̀rọ̀", "àgbà"], "Yoruba tone marks stay inside the token")]
    bad = [name for want, got, name in cases if want != got]
    for name in bad:
        print("FAIL", name)
    print(f"selftest: {len(cases) - len(bad)} passed, {len(bad)} failed")
    return len(bad)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "--selftest":
        sys.exit(1 if selftest() else 0)
    elif cmd == "sign":
        OUT.mkdir(parents=True, exist_ok=True)
        # Freeze the shard list once. The crawl keeps adding shards to collected/,
        # and a list re-read per chunk would shift every stride underneath the
        # workers. Delete data/dedup/ to start over on a newer corpus.
        if not (OUT / "shards.txt").exists():
            names = sorted(p.name for p in COLLECTED.glob("*.jsonl.zst") if p.stat().st_size > 100)
            (OUT / "shards.tmp").write_text("\n".join(names) + "\n")
            os.replace(OUT / "shards.tmp", OUT / "shards.txt")
        with Pool(int(os.environ.get("DEDUP_PROCS", 16))) as pool:
            for n, i in enumerate(pool.imap_unordered(sign_chunk, range(CHUNKS)), 1):
                print(f"{n}/{CHUNKS} chunks signed", flush=True)
    elif cmd == "cluster":
        cluster()
    else:
        sys.exit(__doc__)
