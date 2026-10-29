"""Generate (Z, X, Y0, Y1) rows with an LLM.

Z: a short 20 Newsgroups post. X: a 2-3 sentence reader comment reacting to Z, with a
self-reported sentiment label. Y0: a 2-3 sentence newsletter summary of Z (H0: Y
depends on X only through Z, i.e. not at all given Z). Y1: the same summary task, but
the prompt additionally receives the sentiment label of X and asks for a matching
tone (H1: Y depends on X given Z through s(X)).

All three calls see only Z (plus, for Y1, the label of X). Conditional on the model
weights, X is independent of anything else given Z, so (Z, X, Y0) satisfies H0 by
construction.

Usage:
  python -m citext.llm_gen --dry-run
  python -m citext.llm_gen --pilot 20 [--model claude-haiku-4-5]
  python -m citext.llm_gen --n 3000 [--model claude-haiku-4-5]
The API key is read from PROJECT_ROOT/.env (ANTHROPIC_API_KEY) via python-dotenv and is
never printed.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from .paths import CACHE_DIR, PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

LLM_DIR = CACHE_DIR / "llm"
LLM_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM = (
    "You write short pieces of text for a research dataset on online discussions. "
    "Follow the instructions exactly. Write in natural, varied English. Output plain text "
    "only: no preamble, no headings, no quotation marks around the whole answer."
)

X_INSTR = (
    "Below is a post from an online discussion forum.\n\nPOST:\n<<<\n{Z}\n>>>\n\n"
    "Write a reader comment of 2-3 sentences reacting to this post, as a forum member {PERSONA} would. "
    "Then, on a new final line, write exactly one of the words POSITIVE, NEGATIVE or NEUTRAL "
    "to label the overall sentiment of your comment."
)
Y0_INSTR = (
    "Below is a post from an online discussion forum.\n\nPOST:\n<<<\n{Z}\n>>>\n\n"
    "Write a 2-3 sentence summary of this post for a newsletter that digests forum discussions. "
    "Output only the summary."
)
Y1_INSTR = (
    "Below is a post from an online discussion forum.\n\nPOST:\n<<<\n{Z}\n>>>\n\n"
    "Readers reacted to this post with a clearly {LABEL} sentiment. "
    "Write a 2-3 sentence summary of this post for a newsletter that digests forum discussions, "
    "and let the tone of the summary match this {LABEL} reader sentiment. Output only the summary."
)

PERSONAS = ("who tends to be critical and skeptical", "who tends to be appreciative and enthusiastic",
            "who is matter-of-fact and reserved")

# USD per million tokens (input, output, cache write, cache read)
PRICES = {
    "claude-haiku-4-5": (1.00, 5.00, 1.25, 0.10),
    "claude-sonnet-5": (2.00, 10.00, 2.50, 0.20),
    "claude-opus-5": (5.00, 25.00, 6.25, 0.50),
}


def select_posts(n: int, min_words: int = 40, max_words: int = 150, seed: int = 0):
    from .text_semi import load_corpus
    c = load_corpus()
    ok = [i for i, w in enumerate(c["n_words"]) if min_words <= w <= max_words]
    rng = np.random.default_rng(seed)
    pick = rng.permutation(ok)[:n]
    return [(int(i), c["texts"][i]) for i in pick]


def parse_x(text: str):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    label = "NEUTRAL"
    if lines and re.fullmatch(r"\**(POSITIVE|NEGATIVE|NEUTRAL)\**[.!]?", lines[-1].upper()):
        label = re.sub(r"[^A-Z]", "", lines[-1].upper())
        lines = lines[:-1]
    return " ".join(lines), label


def usage_dict(u):
    return {"input": u.input_tokens, "output": u.output_tokens,
            "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0,
            "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0}


def cost_usd(model: str, tot: dict) -> float:
    pin, pout, pcw, pcr = PRICES.get(model, PRICES["claude-opus-5"])
    return (tot["input"] * pin + tot["output"] * pout + tot["cache_write"] * pcw
            + tot["cache_read"] * pcr) / 1e6


async def one_call(client, model, prompt, sem, max_tokens=300):
    async with sem:
        r = await client.messages.create(
            model=model, max_tokens=max_tokens,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
        )
    text = "".join(b.text for b in r.content if b.type == "text")
    return text, usage_dict(r.usage), r.stop_reason


async def generate_row(client, model, idx, Z, sem):
    persona = PERSONAS[int(np.random.default_rng(idx).integers(len(PERSONAS)))]  # fresh randomness per row
    x_text, x_usage, x_stop = await one_call(client, model, X_INSTR.format(Z=Z, PERSONA=persona), sem)
    X, label = parse_x(x_text)
    y0_task = one_call(client, model, Y0_INSTR.format(Z=Z), sem)
    y1_task = one_call(client, model, Y1_INSTR.format(Z=Z, LABEL=label.lower()), sem)
    (Y0, u0, s0), (Y1, u1, s1) = await asyncio.gather(y0_task, y1_task)
    return {"idx": idx, "Z": Z, "X": X, "X_label": label, "persona": persona, "Y0": Y0.strip(), "Y1": Y1.strip(),
            "usage": {"X": x_usage, "Y0": u0, "Y1": u1}, "stop": [x_stop, s0, s1], "model": model}


async def generate_copy(client, model, idx, Z, sem):
    persona = PERSONAS[int(np.random.default_rng(idx + 10**6).integers(len(PERSONAS)))]
    x_text, x_usage, x_stop = await one_call(client, model, X_INSTR.format(Z=Z, PERSONA=persona), sem)
    X, label = parse_x(x_text)
    return {"idx": idx, "X_copy": X, "X_copy_label": label, "persona_copy": persona, "usage": {"X": x_usage},
            "stop": [x_stop], "model": model}


async def run_copies(model: str, src_path: Path, out_path: Path, concurrency: int = 8):
    """One additional independent draw X'' ~ p(X | Z) for every row of src_path (exchangeable
    copy under H0, for the model-X reference test)."""
    import anthropic
    client = anthropic.AsyncAnthropic(max_retries=6, timeout=120.0)
    rows = [json.loads(l) for l in src_path.read_text().splitlines() if l.strip()]
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["idx"])
            except Exception:
                pass
    todo = [(r["idx"], r["Z"]) for r in rows if r["idx"] not in done]
    print(f"copies: {len(done)} present, generating {len(todo)} with {model}", flush=True)
    sem = asyncio.Semaphore(concurrency)
    tot = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
    t0 = time.time()
    with out_path.open("a") as f:
        tasks = [asyncio.create_task(generate_copy(client, model, i, z, sem)) for i, z in todo]
        k = 0
        for coro in asyncio.as_completed(tasks):
            try:
                row = await coro
            except Exception as e:
                print(f"copy failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            f.write(json.dumps(row) + "\n")
            f.flush()
            for key in tot:
                tot[key] += row["usage"]["X"][key]
            k += 1
            if k % 500 == 0 or k == len(todo):
                print(f"  copies {k}/{len(todo)}, {time.time()-t0:.0f}s, running cost {cost_usd(model, tot):.2f} USD", flush=True)
    return tot, len(todo), time.time() - t0


def crt_rows(src_path: Path, n_rows: int, seed: int = 2024):
    rows = [json.loads(l) for l in src_path.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["idx"])
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(len(rows), size=min(n_rows, len(rows)), replace=False).tolist())
    return [rows[i] for i in pick]


async def generate_crt_draw(client, model, idx, b, Z, sem):
    persona = PERSONAS[int(np.random.default_rng(idx * 1000 + b + 7).integers(len(PERSONAS)))]
    x_text, x_usage, x_stop = await one_call(client, model, X_INSTR.format(Z=Z, PERSONA=persona), sem)
    X, label = parse_x(x_text)
    return {"idx": idx, "b": b, "X_prime": X, "X_prime_label": label, "persona": persona, "usage": x_usage,
            "stop": x_stop, "model": model}


async def run_crt(model: str, src_path: Path, out_path: Path, n_rows: int, B: int, concurrency: int = 8,
                  pilot_rows: int = 0):
    """B (the paper's M) independent regenerations X' ~ p(X | Z) of the comment for each of n_rows rows
    (conditional randomization test). Resumable on (idx, b)."""
    import anthropic
    client = anthropic.AsyncAnthropic(max_retries=6, timeout=120.0)
    rows = crt_rows(src_path, n_rows)
    if pilot_rows:
        rows = rows[:pilot_rows]
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                d = json.loads(line)
                done.add((d["idx"], d["b"]))
            except Exception:
                pass
    todo = [(r["idx"], b, r["Z"]) for r in rows for b in range(B) if (r["idx"], b) not in done]
    print(f"crt: {len(done)} draws present, generating {len(todo)} draws for {len(rows)} rows, B={B}, {model}", flush=True)
    sem = asyncio.Semaphore(concurrency)
    tot = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
    t0 = time.time()
    with out_path.open("a") as f:
        tasks = [asyncio.create_task(generate_crt_draw(client, model, i, b, z, sem)) for i, b, z in todo]
        k = 0
        for coro in asyncio.as_completed(tasks):
            try:
                row = await coro
            except Exception as e:
                print(f"draw failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            f.write(json.dumps(row) + "\n")
            f.flush()
            for key in tot:
                tot[key] += row["usage"][key]
            k += 1
            if k % 1000 == 0 or k == len(todo):
                print(f"  crt {k}/{len(todo)} draws, {time.time()-t0:.0f}s, running cost {cost_usd(model, tot):.2f} USD", flush=True)
    return tot, len(todo), time.time() - t0


async def run(model: str, n: int, out_path: Path, concurrency: int = 8, seed: int = 0):
    import anthropic
    client = anthropic.AsyncAnthropic(max_retries=6, timeout=120.0)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            try:
                done.add(json.loads(line)["idx"])
            except Exception:
                pass
    posts = [(i, z) for i, z in select_posts(n, seed=seed) if i not in done]
    print(f"{len(done)} rows already present, generating {len(posts)} rows with {model}", flush=True)
    sem = asyncio.Semaphore(concurrency)
    tot = {"input": 0, "output": 0, "cache_write": 0, "cache_read": 0}
    t0 = time.time()
    with out_path.open("a") as f:
        tasks = [asyncio.create_task(generate_row(client, model, i, z, sem)) for i, z in posts]
        k = 0
        for coro in asyncio.as_completed(tasks):
            try:
                row = await coro
            except Exception as e:  # keep going; report at the end
                print(f"row failed: {type(e).__name__}: {str(e)[:200]}", flush=True)
                continue
            f.write(json.dumps(row) + "\n")
            f.flush()
            for u in row["usage"].values():
                for key in tot:
                    tot[key] += u[key]
            k += 1
            if k % 100 == 0 or k == len(posts):
                el = time.time() - t0
                print(f"  {k}/{len(posts)} rows, {el:.0f}s, running cost {cost_usd(model, tot):.2f} USD",
                      flush=True)
    return tot, len(posts), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--pilot", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--out", default=None)
    ap.add_argument("--copies", default=None, help="path of an existing generation JSONL; draws one extra X per row")
    ap.add_argument("--crt", default=None, help="path of an existing generation JSONL; B regenerations of X for n-rows rows")
    ap.add_argument("--n-rows", type=int, default=500)
    ap.add_argument("--B", type=int, default=39, help="regenerations per row (M in the paper; the file name keeps the tag B)")
    ap.add_argument("--pilot-rows", type=int, default=0)
    args = ap.parse_args()

    if args.dry_run:
        for i, z in select_posts(2):
            print("=" * 70, f"\nrow idx={i}\n--- X prompt ---\n", X_INSTR.format(Z=z, PERSONA=PERSONAS[0]))
            print("--- Y0 prompt ---\n", Y0_INSTR.format(Z=z))
            print("--- Y1 prompt ---\n", Y1_INSTR.format(Z=z, LABEL="negative"))
        return

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set (checked environment and PROJECT_ROOT/.env). "
              "Nothing generated; exiting.")
        sys.exit(0)

    if args.pilot:
        out = LLM_DIR / f"pilot_{args.model}.jsonl"
        if out.exists():
            out.unlink()
        tot, k, el = asyncio.run(run(args.model, args.pilot, out, args.concurrency))
        per_row = {key: v / max(k, 1) for key, v in tot.items()}
        c_row = cost_usd(args.model, per_row)
        print("\nPILOT token usage per row (3 calls):", per_row)
        print(f"pilot cost {cost_usd(args.model, tot):.3f} USD for {k} rows in {el:.0f}s; "
              f"extrapolated to {args.n} rows: {c_row * args.n:.2f} USD, "
              f"about {el / max(k,1) * args.n / 60:.0f} min at this concurrency")
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        labels = [r["X_label"] for r in rows]
        print("label counts:", {l: labels.count(l) for l in set(labels)})
        for r in rows[:3]:
            print("-" * 70)
            print("Z:", r["Z"][:300])
            print("X:", r["X"], f"[{r['X_label']}]")
            print("Y0:", r["Y0"])
            print("Y1:", r["Y1"])
        return

    if args.crt:
        src = Path(args.crt)
        tag = "_pilot" if args.pilot_rows else ""
        out = LLM_DIR / f"crt_{args.model}_n{args.n_rows}_B{args.B}{tag}.jsonl"
        if args.pilot_rows and out.exists():
            out.unlink()
        tot, k, el = asyncio.run(run_crt(args.model, src, out, args.n_rows, args.B, args.concurrency, args.pilot_rows))
        per_call = {key: v / max(k, 1) for key, v in tot.items()}
        full_calls = args.n_rows * args.B
        print(f"\nCRT tokens {tot}; cost {cost_usd(args.model, tot):.3f} USD for {k} draws in {el:.0f}s; "
              f"per call {per_call}; extrapolated to {full_calls} draws: {cost_usd(args.model, per_call) * full_calls:.2f} USD, "
              f"about {el / max(k, 1) * full_calls / 60:.0f} min")
        out.with_suffix(".usage.json").write_text(json.dumps({"tokens": tot, "cost_usd": cost_usd(args.model, tot), "draws": k, "seconds": el,
                                                              "n_rows": args.n_rows, "B": args.B, "pilot_rows": args.pilot_rows}))
        return

    if args.copies:
        src = Path(args.copies)
        out = src.with_name(src.stem + "_copies.jsonl")
        tot, k, el = asyncio.run(run_copies(args.model, src, out, args.concurrency))
        print(f"\nCOPIES tokens {tot}; cost {cost_usd(args.model, tot):.2f} USD; {k} rows; {el/60:.1f} min")
        out.with_suffix(".usage.json").write_text(json.dumps({"tokens": tot, "cost_usd": cost_usd(args.model, tot), "rows": k, "seconds": el}))
        return

    out = Path(args.out) if args.out else LLM_DIR / f"gen_{args.model}_n{args.n}.jsonl"
    tot, k, el = asyncio.run(run(args.model, args.n, out, args.concurrency))
    print(f"\nTOTAL tokens {tot}; cost {cost_usd(args.model, tot):.2f} USD; {k} rows; {el/60:.1f} min")
    (out.with_suffix(".usage.json")).write_text(json.dumps({"tokens": tot, "cost_usd": cost_usd(args.model, tot),
                                                            "rows_generated_this_run": k, "seconds": el}))


if __name__ == "__main__":
    main()
