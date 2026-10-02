"""Score documents with the keep/drop classifier: p(keep) per document.

  python classifier/score.py <out-name> [<classifier>]             # default classifier: NAIJAWEB_CLASSIFIER
  NAIJAWEB_SRC=F_fineweb2 python classifier/score.py ...           # score another pool version (default P_<set>)
  NAIJAWEB_SRC=rescrape python classifier/score.py ...             # score finished re-crawl hosts, new ids only

<classifier> is a Hugging Face repo id or a local folder. Writes
$NAIJAWEB_DATA/wet/versions/scores/<out-name>/<src>_part_NNNN.jsonl, one {"id": p_keep} map per part of the source
version, atomic and skipped if present so a rerun resumes. A document already scored from another version in this
folder is not scored twice (ids are "<shard>:<line>", the same in every version).
The input is the raw document text, first 512 tokens.

Re-crawl mode reads every finished host file under $NAIJAWEB_DATA/wet/rescrape (a host still being fetched is a .tmp and
is not read), scores only ids that no score file in this folder has yet, and merges them into rescrape_<host>.jsonl,
leaving existing answers as they are. Re-crawl ids are "<seed host>:<sha1 of the requested URL>", so a host fetched
again later matches on the same URL even if it now redirects elsewhere. Run it after any host finishes; a run with
nothing new costs one pass over the ids.
"""
import io
import json, os, sys
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for config.py
from config import DATA, SET, CLASSIFIER, need
import torch, zstandard
from transformers import AutoTokenizer, AutoModelForSequenceClassification

name = sys.argv[1]
classifier = sys.argv[2] if len(sys.argv) > 2 else need(CLASSIFIER, "NAIJAWEB_CLASSIFIER")
SRC = os.environ.get("NAIJAWEB_SRC", f"P_{SET}")
OUT = Path(os.environ.get("NAIJAWEB_SCORES", DATA / "wet/versions/scores")) / name
OUT.mkdir(parents=True, exist_ok=True)
MAXLEN, BS = 512, 64
DEV = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(classifier)
model = AutoModelForSequenceClassification.from_pretrained(classifier, dtype=torch.bfloat16).to(DEV).eval()
dctx = zstandard.ZstdDecompressor()

def read(path):
    return [json.loads(l) for l in dctx.stream_reader(io.BytesIO(path.read_bytes()), read_across_frames=True).read().decode().split("\n") if l]

def score(texts):
    enc = tok(texts, truncation=True, max_length=MAXLEN)["input_ids"]
    order = sorted(range(len(enc)), key=lambda i: len(enc[i]))  # length-sorted batches, less padding
    out = [0.0] * len(texts)
    with torch.no_grad():
        for i in range(0, len(order), BS):
            chunk = [enc[j] for j in order[i:i + BS]]
            L = max(len(c) for c in chunk)
            x = torch.full((len(chunk), L), tok.pad_token_id)
            for r, c in enumerate(chunk):
                x[r, :len(c)] = torch.tensor(c)
            x = x.to(DEV)
            p = model(input_ids=x, attention_mask=(x != tok.pad_token_id)).logits.float().softmax(-1)[:, 1].tolist()
            for j, q in zip(order[i:i + BS], p):
                out[j] = round(q, 4)
    return out

def write(dst, scores):
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(json.dumps(scores))
    os.replace(tmp, dst)

if SRC == "rescrape":
    RS = Path(os.environ.get("NAIJAWEB_RESCRAPE", str(DATA / "wet/rescrape")))
    shards = [p for p in sorted(RS.glob("*.jsonl.zst")) + sorted(RS.glob("*/*.jsonl.zst"))
              if p.parent.name != "dropped_by_decision"]
    if not shards:
        raise SystemExit(f"no finished host files under {RS} -- wrong place, nothing scored")
    already = set()
    for f in OUT.glob("*.jsonl"):
        already |= set(json.load(open(f)))
    hosts = {}
    for p in shards:
        hosts.setdefault(p.name[:-len(".jsonl.zst")], []).append(p)
    n_read = n_skip = n_new = 0
    for host, paths in sorted(hosts.items()):
        rows = {}
        for p in paths:
            for r in read(p):
                n_read += 1
                if r["id"] in already or r["id"] in rows:
                    n_skip += 1
                else:
                    rows[r["id"]] = r["text"]
        if not rows:
            continue
        dst = OUT / f"rescrape_{host}.jsonl"
        prev = json.load(open(dst)) if dst.exists() else {}
        new = dict(zip(rows, score(list(rows.values()))))
        write(dst, {**prev, **new})
        already |= set(new)
        n_new += len(new)
        print(f"{host}: {len(new)} newly scored, {len(prev)} kept from before", flush=True)
    print(f"read {RS}: {len(shards)} files, {len(hosts)} hosts, {n_read} documents; "
          f"{n_skip} already scored, {n_new} newly scored", flush=True)
    raise SystemExit

pool = DATA / f"wet/versions/{SRC}"
PREFIX = SRC.lower() + "_"
already = set()
for f in sorted(OUT.glob("*.jsonl")):
    if not f.name.startswith(PREFIX):
        already |= set(json.load(open(f)))
print(f"{len(already)} documents already scored from other versions, skipping those", flush=True)

for src in sorted(pool.glob("part_*.jsonl.zst")):
    dst = OUT / (PREFIX + src.name.split(".")[0] + ".jsonl")
    if dst.exists():
        continue
    rows = [r for r in read(src) if r["id"] not in already]
    write(dst, dict(zip([r["id"] for r in rows], score([r["text"] for r in rows]))) if rows else {})
    print(f"{src.name}: {len(rows)} docs", flush=True)
