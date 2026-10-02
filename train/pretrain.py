"""Pre-train a small language model from scratch on one corpus version, measuring
held-out loss as it goes.

  python train/pretrain.py --version dedup_within --seed 1 --passes 1
  python train/pretrain.py --version dedup_within --passes 10 --count-only   # check the sampler, no GPU

Training reads the bin as non-overlapping ctx-token chunks, shuffled afresh (with the seed) at every
pass, so one pass reads every chunk exactly once and n passes read each exactly n times. The
tokens after the last whole chunk (fewer than ctx) are never read.

Llama-style decoder, 12 layers / 768 hidden / 12 heads over the shared 32k tokenizer
(85.0M parameters without the input and output embeddings, 134.1M in all). Reads train/bins/<version>.bin.

Held-out loss is read at roughly log-spaced points, starting at step 0 (the untrained
model), on each of the four held-out sets and each language separately, and written as
bits per byte to results/curves_pretrain/<version>_e<passes>_s<seed>.jsonl with the number of
tokens seen. Nothing is pooled and no checkpoint is kept except the last.
Bits per byte = the loss summed over the predicted tokens of a set (tokens 2..ctx of each document) divided by
the bytes of those tokens (a token's bytes = the length of its string in the byte-level vocabulary).
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch
from tokenizers import ByteLevelBPETokenizer
from transformers import LlamaConfig, LlamaForCausalLM

p = argparse.ArgumentParser()
p.add_argument("--version", required=True)
p.add_argument("--seed", type=int, default=1)
p.add_argument("--passes", type=int, default=1)
p.add_argument("--count-only", action="store_true")
p.add_argument("--ctx", type=int, default=1024)
p.add_argument("--bs", type=int, default=32)
p.add_argument("--accum", type=int, default=2)
p.add_argument("--lr", type=float, default=6e-4)
a = p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed)
# smoke-test / CPU support: identical behaviour on a GPU machine (still .cuda(), still compiled);
# falls back to CPU only when no GPU is visible, so smoke/run.sh can exercise this script without one.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if DEVICE.type == "cpu":
    torch.set_num_threads(min(4, torch.get_num_threads()))   # a shared-machine smoke test, not a real run

ROOT = Path(__file__).resolve().parent.parent
data = np.memmap(ROOT / f"train/bins/{a.version}.bin", dtype=np.uint16, mode="r")
n_chunks = len(data) // a.ctx
rng = np.random.RandomState(a.seed)
order = np.concatenate([rng.permutation(n_chunks) for _ in range(a.passes)])   # chunk numbers, a new shuffle each pass
per_step = a.bs * a.accum                                                      # chunks per optimizer step
steps = math.ceil(len(order) / per_step)                                       # the last step may be short

def micro_batches(step):
    idx = order[(step - 1) * per_step:step * per_step]
    return [idx[i:i + a.bs] for i in range(0, len(idx), a.bs)]

if a.count_only:
    visits = np.zeros(n_chunks, dtype=np.int64)
    for step in range(1, steps + 1):
        for mb in micro_batches(step): np.add.at(visits, mb, 1)
    print(f"{a.version}: {len(data):,} tokens, {n_chunks:,} chunks of {a.ctx}, {a.passes} passes, {steps:,} steps; "
          f"visits per chunk min {visits.min()} max {visits.max()}; chunks read {visits.sum():,} = {a.passes} x {n_chunks:,}: "
          f"{visits.sum() == a.passes * n_chunks}; unique tokens read {n_chunks * a.ctx:,}")
    assert visits.min() == visits.max() == a.passes
    raise SystemExit
tok = ByteLevelBPETokenizer(str(ROOT / "train/tok32k/vocab.json"), str(ROOT / "train/tok32k/merges.txt"))

# held-out: one (set, language) group per file, tokenized once
tok_bytes = {i: len(t) for t, i in json.load(open(ROOT / "train/tok32k/vocab.json")).items()}
def scored_bytes(tok, nbytes, text, ctx):
    """Bytes of ids[1:ctx], the tokens the loss is taken on."""
    ids = tok.encode(text).ids
    assert sum(nbytes[i] for i in ids) == len(text.encode()), "byte-level vocabulary does not add up"
    return sum(nbytes[i] for i in ids[1:ctx])
groups = {}
for f in sorted((ROOT / "data/heldout").glob("*.jsonl")):
    texts = [json.loads(l)["text"] for l in open(f)]
    groups[f.stem] = [(tok.encode(t).ids[:a.ctx], scored_bytes(tok, tok_bytes, t, a.ctx)) for t in texts]

cfg = LlamaConfig(vocab_size=32000, hidden_size=768, intermediate_size=2048,
                  num_hidden_layers=12, num_attention_heads=12, max_position_embeddings=a.ctx)
cfg._attn_implementation = "sdpa"
model = LlamaForCausalLM(cfg).to(DEVICE)
model.gradient_checkpointing_disable()
trainer = torch.compile(model) if DEVICE.type == "cuda" else model   # compile only on GPU; on
                                 # CPU (smoke test only) eager is faster for a two-step run than
                                 # paying compilation cost. training only; the held-out reads use
                                 # the eager module, so their varying batch shapes cause no recompilation
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.02, anneal_strategy="cos")
def to_gpu(chunks):
    return torch.from_numpy(np.stack([data[c * a.ctx:(c + 1) * a.ctx].astype(np.int64) for c in chunks])).to(DEVICE)

@torch.no_grad()
def heldout():
    model.eval()
    out = {}
    for name, docs in groups.items():
        nats = nbytes = 0.0
        for i in range(0, len(docs), 8):
            chunk = docs[i:i + 8]
            L = max(len(c[0]) for c in chunk)
            x = torch.zeros((len(chunk), L), dtype=torch.long)
            m = torch.zeros((len(chunk), L), dtype=torch.bool)
            for r, (ids, _) in enumerate(chunk):
                x[r, :len(ids)] = torch.tensor(ids); m[r, :len(ids)] = True
            x, m = x.to(DEVICE), m.to(DEVICE)
            with torch.autocast(DEVICE.type, dtype=torch.bfloat16):
                logits = model(input_ids=x).logits.float()
            ll = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 32000),
                                                   x[:, 1:].reshape(-1), reduction="none")
            nats += (ll * m[:, 1:].reshape(-1)).sum().item()
            nbytes += sum(b for _, b in chunk)
        out[name] = round(nats / nbytes / math.log(2), 4)      # bits per byte
    model.train()
    return out

points = sorted({0} | {min(steps, int(round(2 ** (i / 2)))) for i in range(2, 40)} | {steps})
points = [s for s in points if s <= steps]
# Never truncate a curve: a restart that opened a finished curve file with "w" would leave a step-0 stub.
# An existing file is moved aside (kept, with the time in its name) and the new one is
# created exclusively, so nothing can ever be overwritten.
run = f"{a.version}_e{a.passes}_s{a.seed}"
curve_path = ROOT / f"results/curves_pretrain/{run}.jsonl"
curve_path.parent.mkdir(parents=True, exist_ok=True)
if curve_path.exists():
    curve_path.rename(curve_path.with_name(f"{curve_path.name}.{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}.kept"))
curve = open(curve_path, "x")
def read(step):
    row = {"version": a.version, "passes": a.passes, "seed": a.seed, "step": step, "tokens": min(step * per_step, len(order)) * a.ctx,
           "bits_per_byte": heldout()}
    curve.write(json.dumps(row) + "\n"); curve.flush()
    print(step, json.dumps(row["bits_per_byte"]), flush=True)

t0 = time.time()
read(0)
for step in range(1, steps + 1):
    mbs = micro_batches(step)
    for mb in mbs:
        x = to_gpu(mb)
        with torch.autocast(DEVICE.type, dtype=torch.bfloat16):
            loss = trainer(input_ids=x, labels=x).loss   # HF shifts labels itself
        (loss * len(mb) / sum(map(len, mbs))).backward()   # weight by chunks, so a short last step is a mean too
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    if step in points:
        read(step)
model.save_pretrained(ROOT / f"models/pretrain_{run}")
print("done", round((time.time() - t0) / 60, 1), "minutes", flush=True)
