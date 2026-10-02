"""Count re-crawl docs with UTF-8-read-as-Latin-1 garbling (>=3 marks per doc), per host.
Read-only. Usage: python build/mojibake_scan.py [host ...]   (no args = every shard)
Marks include the Hausa hooks: Æ™ = ƙ, É— = ɗ, Æ´ = ƴ; and Yoruba/Igbo dotted letters."""
import glob, io, json, re, sys, collections, zstandard
SIG = re.compile("Æ[\u0099™˜\u0098]|É[\u0097—“\u0093]|Æ´|Ã[\u0080-¿]|á[»º¸¹][\u0080-¿]|â€")
hosts = set(sys.argv[1:])
hit, tot = collections.Counter(), collections.Counter()
for f in glob.glob("/tmp/rescrape/**/*.jsonl.zst", recursive=True):
    if "dropped_by_decision" in f or (hosts and f.split("/")[-1][:-len(".jsonl.zst")] not in hosts):
        continue
    with open(f, "rb") as fh:
        for line in io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh, read_across_frames=True), encoding="utf-8"):
            r = json.loads(line); tot[r["host"]] += 1
            hit[r["host"]] += len(SIG.findall(r["text"])) >= 3
print(f"{sum(hit.values()):,} garbled of {sum(tot.values()):,} docs in {len(tot)} hosts")
for h in sorted(tot, key=lambda h: -hit[h]):
    if hit[h] or hosts: print(f"  {h:36} {hit[h]:>5}/{tot[h]}")
