"""Experiment 4, ground-truth tolerance, phase 1: regenerate the comment and the summary M times per post from the local
generator. The draws are exact replicates of X and Y given Z in the sense of the proposition "Identification with
replicates" (the persona is redrawn per draw: Z is the post, the persona is noise). Draw m = 0 can serve as the observed
data, so that data and replicates come from one engine and share one conditional law exactly.

Engines:  --engine mlx     in-process mlx-lm with continuous batching; run with ~/.venvs/mlx/bin/python
          --engine vllm    in-process vLLM offline API on one GPU (continuous batching, prefix caching); for the vLLM
                           container on a Slurm cluster, --model = local weights directory, --shard i/N for one GPU per shard
          --engine openai  any OpenAI-compatible completions endpoint: llama-server on this machine or vLLM server;
                           --flavor llama|vllm selects the names of the sampler fields
Same prompts, personas and label parsing as exp4_local.py. The chat template is rendered with the HF tokenizer (thinking
disabled) and sent as a string or as token ids, so it is identical across engines. Sampling as in the original run:
temperature 1, top-p 0.95, top-k off, min-p 0, no repetition penalty, at most 160 new tokens.
Output, resumable, one line per (idx, m):  ~/.cache/ci_text/llm/regen_<model>_n<n>_M<M>.jsonl
usage:  python exp4_regen.py --engine mlx --M 5 [--n 3000] [--jobs-per-chunk 128] [--completion-batch 16] [--prefill-batch 2] [--prefill-step 512]
        python exp4_regen.py --engine openai --base-url http://127.0.0.1:8080/v1 --flavor llama --slots 32 --M 5
        python exp4_regen.py --engine vllm --model /path/Qwen3-1.7B --M 20 --source <posts.jsonl> --out-dir <dir> [--shard 0/8]
        python exp4_regen.py --merge --M 20 --out-dir <dir>      # concatenate the shards into the final file
Needs CI_TEXT_CODE_DIR=<repository root> when the script does not live in experiments/ of the repository.
"""
from __future__ import annotations
import argparse, json, os, re, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
CODE = Path(os.environ.get("CI_TEXT_CODE_DIR", HERE.parent)); sys.path.insert(0, str(CODE))
try: import dotenv  # noqa: F401
except ImportError:   # citext.llm_gen loads .env at import; not needed here, and the vLLM container has no python-dotenv
    import types; sys.modules["dotenv"] = types.SimpleNamespace(load_dotenv=lambda *a, **k: None)
from citext.llm_gen import SYSTEM, X_INSTR, Y0_INSTR, PERSONAS
from citext.paths import CACHE_DIR
LLM = CACHE_DIR / "llm"; SOURCE = LLM / "gen_claude-haiku-4-5_n3000.jsonl"
SEED, MAX_NEW, TEMP, TOP_P = 0, 160, 1.0, 0.95
LABEL_RE = re.compile(r"\s*(POSITIVE|NEGATIVE|NEUTRAL)\W*$", re.I)   # as in exp4_local.py
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
def tag(model): return model.split("/")[-1].lower()
def out_path(model, n, M, out_dir=None, shard=None):
    p = Path(out_dir) if out_dir else LLM; s = f".part{shard.replace('/', 'of')}" if shard else ""
    return p / f"regen_{tag(model)}_n{n}_M{M}{s}.jsonl"
def source_posts(n, source=None):
    rows = [json.loads(l) for l in Path(source or SOURCE).read_text().splitlines() if l.strip()]
    return [(r["idx"], r["Z"]) for r in rows][:n]
def persona(idx, m): return PERSONAS[int(np.random.default_rng(idx * 1000 + m + 7).integers(len(PERSONAS)))]   # redrawn per draw, as in the Experiment 3 reference
def parse_label(x):
    mm = LABEL_RE.search(x); return (mm.group(1).upper(), LABEL_RE.sub("", x).strip()) if mm else (None, x)
def chat(tok, user):
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)

class MLXEngine:
    def __init__(self, model, completion_batch, prefill_batch, prefill_step, cache_limit_gb):
        import mlx.core as mx; from mlx_lm import load; from mlx_lm.sample_utils import make_sampler
        self.mx = mx; self.model, self.tok = load(model)
        if cache_limit_gb is not None: mx.set_cache_limit(int(cache_limit_gb * 2**30))   # keep the allocator from hoarding freed buffers on a 16 GB machine
        self.sampler = make_sampler(temp=TEMP, top_p=TOP_P, min_p=0.0, top_k=0)
        self.kw = dict(completion_batch_size=completion_batch, prefill_batch_size=prefill_batch, prefill_step_size=prefill_step)
        w = self.model.model.layers[0].mlp.gate_proj.weight
        log(f"mlx-lm: {model} loaded, weights {w.dtype}, {len(self.model.model.layers)} layers, completion batch {completion_batch}, prefill batch {prefill_batch}, prefill step {prefill_step}")
    def generate(self, prompts, seed):
        from mlx_lm import batch_generate
        self.mx.random.seed(seed)
        ids = [self.tok.encode(p, add_special_tokens=False) for p in prompts]
        order = sorted(range(len(ids)), key=lambda k: len(ids[k]))   # similar prompt lengths per prefill batch: less padding, smaller peak memory
        r = batch_generate(self.model, self.tok, [ids[k] for k in order], max_tokens=MAX_NEW, verbose=False, sampler=self.sampler, **self.kw)
        texts = [None] * len(ids)
        for k, t in zip(order, r.texts): texts[k] = t.strip()
        return texts, r.stats

class VLLMEngine:
    """vLLM offline API on one GPU: continuous batching and prefix caching; token ids are passed, so the prompt is exact."""
    def __init__(self, model, gpu_mem, max_model_len, max_num_seqs):
        from vllm import LLM, SamplingParams; from transformers import AutoTokenizer
        self.SP = SamplingParams; self.tok = AutoTokenizer.from_pretrained(model)
        self.llm = LLM(model=model, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=gpu_mem, enable_prefix_caching=True, max_num_seqs=max_num_seqs, seed=SEED)
        import vllm; log(f"vllm {vllm.__version__}: {model} loaded, max_model_len {max_model_len}, max_num_seqs {max_num_seqs}")
    def generate(self, prompts, seed):
        ids = [{"prompt_token_ids": self.tok.encode(p, add_special_tokens=False)} for p in prompts]
        sp = [self.SP(temperature=TEMP, top_p=TOP_P, top_k=-1, min_p=0.0, repetition_penalty=1.0, max_tokens=MAX_NEW, seed=seed * 100003 + k) for k in range(len(prompts))]
        outs = self.llm.generate(ids, sp, use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outs], None

class HTTPEngine:
    """OpenAI-compatible /completions with the rendered prompt string; `slots` concurrent requests."""
    def __init__(self, model, base_url, flavor, slots):
        from transformers import AutoTokenizer
        self.model, self.url, self.slots = model, base_url.rstrip("/") + "/completions", slots
        self.extra = {"llama": {"top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0}, "vllm": {"top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0}}[flavor]
        self.tok = AutoTokenizer.from_pretrained(model); log(f"http: {self.url} ({flavor}), {slots} slots")
    def _one(self, prompt, seed):
        import urllib.request
        body = json.dumps({"model": self.model, "prompt": prompt, "max_tokens": MAX_NEW, "temperature": TEMP, "top_p": TOP_P, "seed": int(seed), **self.extra}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ.get("CI_TEXT_LOCAL_API_KEY", "none")})
        for attempt in range(6):
            try:
                with urllib.request.urlopen(req, timeout=900) as resp: return json.loads(resp.read())["choices"][0]["text"].strip()
            except Exception as e:
                if attempt == 5: raise
                log(f"  retry {attempt + 1}: {e}"); time.sleep(2 ** attempt)
    def generate(self, prompts, seed):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(self.slots) as ex: texts = list(ex.map(lambda a: self._one(a[1], seed * 100003 + a[0]), enumerate(prompts)))
        return texts, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=("mlx", "vllm", "openai"), default=None); ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--source", default=None, help="posts JSONL (default: the Experiment 3 generation file in the cache)"); ap.add_argument("--out-dir", default=None)
    ap.add_argument("--shard", default=None, help="i/N: this process handles jobs i, i+N, ... and writes a .partXofN file"); ap.add_argument("--merge", action="store_true", help="concatenate the shard files into the final file and exit")
    ap.add_argument("--gpu-mem", type=float, default=0.85); ap.add_argument("--max-model-len", type=int, default=2048); ap.add_argument("--max-num-seqs", type=int, default=512)
    ap.add_argument("--M", type=int, default=5); ap.add_argument("--n", type=int, default=3000); ap.add_argument("--jobs-per-chunk", type=int, default=128)
    ap.add_argument("--completion-batch", type=int, default=16); ap.add_argument("--prefill-batch", type=int, default=2); ap.add_argument("--prefill-step", type=int, default=512)
    ap.add_argument("--cache-limit-gb", type=float, default=1.0)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1"); ap.add_argument("--flavor", choices=("llama", "vllm"), default="llama"); ap.add_argument("--slots", type=int, default=32)
    a = ap.parse_args()
    if a.merge:
        final = out_path(a.model, a.n, a.M, a.out_dir); parts = sorted(final.parent.glob(final.stem + ".part*of*.jsonl"))
        rows = [json.loads(l) for p in parts for l in p.read_text().splitlines() if l.strip()]; rows.sort(key=lambda r: (r["idx"], r["m"]))
        seen = set(); rows = [r for r in rows if not ((r["idx"], r["m"]) in seen or seen.add((r["idx"], r["m"])))]
        final.write_text("".join(json.dumps(r) + "\n" for r in rows)); log(f"merged {len(parts)} shard files, {len(rows)} draws -> {final}"); return
    if a.engine is None: ap.error("--engine is required")
    posts = source_posts(a.n, a.source); path = out_path(a.model, a.n, a.M, a.out_dir, a.shard)
    done = {(r["idx"], r["m"]) for r in (json.loads(l) for l in path.read_text().splitlines() if l.strip())} if path.exists() else set()
    jobs = [(i, z, m) for i, z in posts for m in range(a.M)]   # the M draws of a post are consecutive (prefix caching)
    if a.shard: i_s, n_s = (int(v) for v in a.shard.split("/")); jobs = jobs[i_s::n_s]
    jobs = [j for j in jobs if (j[0], j[2]) not in done]
    log(f"{len(done)} draws done, {len(jobs)} to do ({len(posts)} posts x M={a.M}{', shard ' + a.shard if a.shard else ''}) -> {path}")
    if not jobs: return
    eng = (MLXEngine(a.model, a.completion_batch, a.prefill_batch, a.prefill_step, a.cache_limit_gb) if a.engine == "mlx"
           else VLLMEngine(a.model, a.gpu_mem, a.max_model_len, a.max_num_seqs) if a.engine == "vllm" else HTTPEngine(a.model, a.base_url, a.flavor, a.slots))
    tok = eng.tok; n_done = 0; t_gen = time.time()
    with open(path, "a") as f:
        for c in range(0, len(jobs), a.jobs_per_chunk):
            chunk = jobs[c:c + a.jobs_per_chunk]; tb = time.time()
            prompts = [chat(tok, X_INSTR.format(Z=z, PERSONA=persona(i, m))) for i, z, m in chunk] + [chat(tok, Y0_INSTR.format(Z=z)) for i, z, m in chunk]
            texts, stats = eng.generate(prompts, SEED + c)
            missing = 0
            for k, (i, z, m) in enumerate(chunk):
                lab, x = parse_label(texts[k]); missing += lab is None
                f.write(json.dumps({"idx": i, "m": m, "X": x, "X_label": lab, "persona": persona(i, m), "Y0": texts[len(chunk) + k], "engine": a.engine, "model": tag(a.model)}) + "\n")
            f.flush(); n_done += len(chunk); dt = time.time() - tb; per_text = (time.time() - t_gen) / (2 * n_done)
            log(f"  {n_done}/{len(jobs)} draws, {dt / (2 * len(chunk)):.2f} s per text (chunk), {per_text:.2f} (overall)"
                + (f", gen {stats.generation_tps:.0f} tok/s, prompt {stats.prompt_tps:.0f} tok/s, peak {stats.peak_memory:.1f} GB" if stats else "")
                + f", labels missing {missing}, ETA {(len(jobs) - n_done) * 2 * per_text / 3600:.2f} h")
    log("DONE")

if __name__ == "__main__":
    main()
