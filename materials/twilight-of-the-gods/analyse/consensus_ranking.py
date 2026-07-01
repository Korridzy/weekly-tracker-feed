#!/usr/bin/env python3
"""Evaluator consensus & medoid ranking over the remade 11x11 dataset (2026-06-11).

Generates docs/planner-graph-ref/analyse/evaluator-consensus-ranking.md.

Method: analyse-evaluation-method.md (weighted center, closest-to-overall, medoid).
Spec:   .omc/specs/deep-interview-evaluator-consensus-ranking.md

Run (project venv only):
    poetry run python3 docs/planner-graph-ref/analyse/consensus_ranking.py

Flags:
    --rank-only   skip the embedding (semantic) track
    --stdout      additionally print the full report markdown to stdout
"""

# Embedded markdown prose (SECTION_2_6, COMPARISON_DISCUSSION, report strings)
# exceeds the 100-column limit by design — exempt this generator from E501 only.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
import traceback
from datetime import date
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
REPORT_PATH = HERE / "evaluator-consensus-ranking.md"

# ---------------------------------------------------------------------------
# Input data — extracted 2026-06-11 from the remade *-range.md files.
# Every row is a valid permutation of 1..11 (sum 66); verified at extraction.
# ---------------------------------------------------------------------------

EVALUATORS = [
    "deepseek-4-pro",
    "fable-5",
    "gemini-3.1-pro",
    "glm-5.1",
    "gpt-5.4",
    "gpt-5.5",
    "kimi-2.6",
    "mimo-2.5-pro",
    "opus-4.7",
    "qwen-3.6-plus",
    "qwen-3.7-max",
]
PROPOSALS = [
    "deepseek-4-pro",
    "fable-5",
    "gemini-3.1-pro",
    "glm-5.1",
    "gpt-5.4",
    "gpt-5.5",
    "kimi-2.6",
    "mimo-2.5-pro",
    "opus",
    "qwen-3.6-plus",
    "qwen-3.7-max",
]
SHORT_E = {
    "deepseek-4-pro": "deepseek4",
    "fable-5": "fable5",
    "gemini-3.1-pro": "gemini3.1",
    "glm-5.1": "glm5.1",
    "gpt-5.4": "gpt5.4",
    "gpt-5.5": "gpt5.5",
    "kimi-2.6": "kimi2.6",
    "mimo-2.5-pro": "mimo2.5",
    "opus-4.7": "opus4.7",
    "qwen-3.6-plus": "qwen3.6+",
    "qwen-3.7-max": "qwen3.7",
}
SHORT_P = {
    "deepseek-4-pro": "deepseek4",
    "fable-5": "fable5",
    "gemini-3.1-pro": "gemini3.1",
    "glm-5.1": "glm5.1",
    "gpt-5.4": "gpt5.4",
    "gpt-5.5": "gpt5.5",
    "kimi-2.6": "kimi2.6",
    "mimo-2.5-pro": "mimo2.5",
    "opus": "opus",
    "qwen-3.6-plus": "qwen3.6+",
    "qwen-3.7-max": "qwen3.7",
}
EVAL_TO_OWN = {e: e for e in EVALUATORS}
EVAL_TO_OWN["opus-4.7"] = "opus"  # evaluator stem vs proposal file name

# Rows in PROPOSALS order. gemini-3.1-pro row is tier-imputed:
# tier1 {gpt-5.4, fable-5, deepseek-4-pro} -> 2; tier2 {mimo, glm-5.1, gemini} -> 5;
# tier3 {qwen-3.7-max, gpt-5.5, kimi-2.6, qwen-3.6-plus} -> 8.5; tier4 {opus} -> 11.
RANKS: dict[str, list[float]] = {
    "deepseek-4-pro": [4, 2, 7, 5, 6, 3, 1, 8, 11, 10, 9],
    "fable-5": [5, 1, 11, 7, 3, 2, 6, 9, 4, 10, 8],
    "gemini-3.1-pro": [2, 2, 5, 5, 2, 8.5, 8.5, 5, 11, 8.5, 8.5],
    "glm-5.1": [2, 1, 10, 5, 3, 6, 7, 11, 4, 8, 9],
    "gpt-5.4": [5, 2, 8, 6, 1, 3, 7, 9, 4, 10, 11],
    "gpt-5.5": [4, 1, 9, 7, 2, 3, 6, 10, 5, 11, 8],
    "kimi-2.6": [2, 1, 4, 5, 3, 7, 8, 6, 11, 9, 10],
    "mimo-2.5-pro": [4, 1, 8, 9, 3, 2, 6, 7, 5, 10, 11],
    "opus-4.7": [8, 1, 7, 9, 2, 3, 5, 6, 4, 10, 11],
    "qwen-3.6-plus": [7, 1, 10, 6, 3, 2, 5, 11, 4, 8, 9],
    "qwen-3.7-max": [2, 1, 11, 7, 3, 5, 6, 10, 4, 9, 8],
}

N = len(EVALUATORS)
W = 1.0 / N  # equal evaluator weights w_i
EPS_RANK = 0.05  # near-tie threshold, rank-track distances
EPS_COS = 0.002  # near-tie threshold, cosine distances

# Semantic track parameters
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIM = 3072
MIN_BLOCK_TOKENS = 5
MAX_INPUT_TOKENS = 2048
BATCH_SIZE = 50
EMBED_CALLS_BEFORE_SLEEP = 50
SLEEP_SECONDS = 60
INTER_BATCH_SLEEP = 3.0
MAX_RETRIES = 6

_request_count = 0


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Rank track
# ---------------------------------------------------------------------------


def rank_track() -> dict:
    R = np.array([RANKS[e] for e in EVALUATORS], dtype=float)
    C = np.ones_like(R)
    for i, e in enumerate(EVALUATORS):
        C[i, PROPOSALS.index(EVAL_TO_OWN[e])] = 0.0
    # sanity: every full row (incl. self cell) is a permutation of 1..11
    assert np.allclose(R.sum(axis=1), 66.0), "rank rows must sum to 66"

    # consensus s̄_j over the 10 non-author evaluators (equal weights cancel)
    cons = (C * R).sum(axis=0) / C.sum(axis=0)
    var = (C * (R - cons) ** 2).sum(axis=0) / C.sum(axis=0)

    # closest to overall — normalized (RMS) Euclidean over each evaluator's 10 cells
    eucl = np.sqrt(((C * (R - cons) ** 2).sum(axis=1)) / C.sum(axis=1))

    # Spearman cross-check on the same 10 cells
    rho = np.zeros(N)
    for i in range(N):
        mask = C[i] == 1.0
        res = stats.spearmanr(R[i, mask], cons[mask])
        rho[i] = float(getattr(res, "statistic", getattr(res, "correlation", np.nan)))
    spear = 1.0 - rho

    # pairwise RMS Euclidean over the 9 proposals ranked by BOTH
    D = np.zeros((N, N))
    for i in range(N):
        for k in range(i + 1, N):
            mask = (C[i] == 1.0) & (C[k] == 1.0)
            d = float(np.sqrt(np.mean((R[i, mask] - R[k, mask]) ** 2)))
            D[i, k] = D[k, i] = d

    S = W * D.sum(axis=1)  # weighted medoid totals (D[i,i] = 0)

    return {
        "R": R,
        "C": C,
        "cons": cons,
        "var": var,
        "eucl": eucl,
        "rho": rho,
        "spear": spear,
        "D": D,
        "S": S,
    }


# ---------------------------------------------------------------------------
# Semantic track
# ---------------------------------------------------------------------------


def get_token_counter():
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        def count(text: str) -> int:
            return len(enc.encode(text))

        def truncate(text: str, limit: int) -> str:
            toks = enc.encode(text)
            return text if len(toks) <= limit else enc.decode(toks[:limit])

        return count, truncate, "tiktoken cl100k_base (proxy)"
    except Exception:  # pragma: no cover - fallback

        def count(text: str) -> int:
            return len(text.split())

        def truncate(text: str, limit: int) -> str:
            words = text.split()
            return text if len(words) <= limit else " ".join(words[:limit])

        return count, truncate, "whitespace word count (tiktoken unavailable)"


def split_blocks(text: str) -> list[str]:
    """Blank-line-delimited blocks; fenced code/Mermaid kept intact."""
    blocks: list[str] = []
    buf: list[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            buf.append(line)
            continue
        if not stripped and not in_fence:
            if buf:
                blocks.append("\n".join(buf).strip())
                buf = []
            continue
        buf.append(line)
    if buf:
        blocks.append("\n".join(buf).strip())
    return [b for b in blocks if b]


def load_google_api_key() -> str:
    key = os.environ.get("GOOGLE_API_KEY", "")
    if not key:
        env_path = REPO_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("GOOGLE_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not found in environment or .env")
    return key


def embed_batch(client, types_mod, batch: list[str]) -> list[np.ndarray]:
    global _request_count
    backoff = 30.0
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.embed_content(
                model=EMBED_MODEL,
                contents=batch,
                config=types_mod.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
            )
            _request_count += 1
            if _request_count % EMBED_CALLS_BEFORE_SLEEP == 0:
                log(f"[throttle] {_request_count} requests -> sleeping {SLEEP_SECONDS}s")
                time.sleep(SLEEP_SECONDS)
            return [np.asarray(e.values, dtype=float) for e in resp.embeddings]
        except Exception as exc:  # noqa: BLE001 - inspect & retry rate limits
            msg = str(exc).lower()
            transient = any(
                t in msg
                for t in ("429", "resource_exhausted", "rate", "quota", "503", "unavailable")
            )
            if transient and attempt < MAX_RETRIES - 1:
                log(
                    f"[retry] transient embed error (attempt {attempt + 1}): sleeping {backoff:.0f}s"
                )
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
    raise RuntimeError("unreachable")


def semantic_track() -> dict:
    count_tokens, truncate, token_note = get_token_counter()

    docs: dict[str, list[tuple[str, int]]] = {}
    stats_rows = []
    for e in EVALUATORS:
        text = (HERE / f"{e}-range.md").read_text(encoding="utf-8")
        raw = split_blocks(text)
        kept = []
        for b in raw:
            t = count_tokens(b)
            if t < MIN_BLOCK_TOKENS:
                continue
            kept.append((truncate(b, MAX_INPUT_TOKENS), min(t, MAX_INPUT_TOKENS)))
        docs[e] = kept
        toks = [t for _, t in kept]
        stats_rows.append((e, len(raw), len(kept), sum(toks), max(toks)))

    total_chunks = sum(len(v) for v in docs.values())
    total_tokens = sum(t for v in docs.values() for _, t in v)
    log(f"[chunking] {total_chunks} chunks / {total_tokens} tokens across {N} docs ({token_note})")

    # cache pooled doc vectors on the exact file contents + model id
    h = hashlib.sha256(EMBED_MODEL.encode())
    for e in EVALUATORS:
        h.update((HERE / f"{e}-range.md").read_bytes())
    cache = Path("/tmp") / f"consensus_ranking_emb_{h.hexdigest()[:16]}.npz"

    if cache.exists():
        log(f"[cache] reusing pooled embeddings from {cache}")
        E = np.load(cache)["E"]
    else:
        from google import genai
        from google.genai import types as genai_types

        client = genai.Client(api_key=load_google_api_key())
        E = np.zeros((N, EMBED_DIM))
        for i, e in enumerate(EVALUATORS):
            chunks = docs[e]
            vecs: list[np.ndarray] = []
            for b0 in range(0, len(chunks), BATCH_SIZE):
                batch = [c for c, _ in chunks[b0 : b0 + BATCH_SIZE]]
                vecs.extend(embed_batch(client, genai_types, batch))
                log(
                    f"[embed] {e}: {min(b0 + BATCH_SIZE, len(chunks))}/{len(chunks)} chunks "
                    f"(request #{_request_count})"
                )
                time.sleep(INTER_BATCH_SLEEP)
            weights = np.array([t for _, t in chunks], dtype=float)
            V = np.vstack(vecs)
            E[i] = (weights[:, None] * V).sum(axis=0) / weights.sum()
        np.savez_compressed(cache, E=E)
        log(f"[cache] saved pooled embeddings to {cache}")

    centroid = E.mean(axis=0)

    def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
        return float(1.0 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    dcos = np.array([cos_dist(E[i], centroid) for i in range(N)])
    deuc = np.array([float(np.linalg.norm(E[i] - centroid)) for i in range(N)])

    Dcos = np.zeros((N, N))
    for i in range(N):
        for k in range(i + 1, N):
            Dcos[i, k] = Dcos[k, i] = cos_dist(E[i], E[k])
    np.fill_diagonal(Dcos, 0.0)
    Scos = W * Dcos.sum(axis=1)

    return {
        "stats": stats_rows,
        "total_chunks": total_chunks,
        "total_tokens": total_tokens,
        "token_note": token_note,
        "dcos": dcos,
        "deuc": deuc,
        "Dcos": Dcos,
        "Scos": Scos,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------


def fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def fmt_rank(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else str(v)


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def ordered(names: list[str], values: np.ndarray) -> list[tuple[str, float]]:
    return sorted(zip(names, values.tolist(), strict=True), key=lambda p: p[1])


def tie_note(pairs: list[tuple[str, float]], eps: float) -> str:
    groups: list[list[str]] = []
    cur = [pairs[0]]
    for prev, nxt in zip(pairs, pairs[1:], strict=False):  # adjacent pairs, offset by one
        if nxt[1] - prev[1] < eps:
            cur.append(nxt)
        else:
            if len(cur) > 1:
                groups.append([n for n, _ in cur])
            cur = [nxt]
    if len(cur) > 1:
        groups.append([n for n, _ in cur])
    if not groups:
        return f"No near-ties at ε = {eps}."
    parts = ["{" + ", ".join(g) + "}" for g in groups]
    return f"Near-ties (Δ < ε = {eps}): " + "; ".join(parts) + " — treat as effectively tied."


def ranking_table(pairs: list[tuple[str, float]], value_label: str, nd: int = 4) -> str:
    rows = [[str(i + 1), n, fmt(v, nd)] for i, (n, v) in enumerate(pairs)]
    return md_table(["#", "Evaluator", value_label], rows)


def matrix_table(M: np.ndarray, names: list[str], nd: int = 4) -> str:
    header = ["↓ \\ →"] + [SHORT_E[n] for n in names]
    rows = []
    for i, n in enumerate(names):
        rows.append([SHORT_E[n]] + [fmt(M[i, k], nd) for k in range(len(names))])
    return md_table(header, rows)


# Prose finalized from the computed 11x11 numbers (kept identical in the report).
SECTION_2_6 = """### 2.6 Why the two closest-to-overall rankings differ (Euclidean vs Spearman)

The two metrics answer subtly different questions:

- **Normalized Euclidean (§2.2)** measures **magnitude**: it squares each gap `(sᵢⱼ − s̄ⱼ)` between an evaluator's rank and the consensus mean. One big miss costs more than several small ones, and swapping two proposals whose consensus values are nearly tied costs almost nothing.
- **Spearman 1−ρ (§2.3)** measures **order**: it charges for every inversion equally, no matter how small the consensus gap it crosses. A swap across a 0.05-wide consensus gap costs as much as a swap across a 3-point gap.

In this 11×11 run the two rankings agree unusually well — same #1 (**gpt-5.5**: dᵢ = 0.986 *and* 1−ρ = 0.052) and the same bottom three in the same order (deepseek-4-pro, kimi-2.6, gemini-3.1-pro) — because the consensus is sharply stratified at the extremes (fable-5 at 1.30; the qwen pair near 9.5). The divergence is confined to the #2–#4 band, and it is driven by the consensus's near-tied pairs (§2.1):

| Near-tied consensus pair | s̄ⱼ values | Gap |
|---|---|---|
| deepseek-4-pro ≈ gpt-5.5 | 4.10 vs 4.15 | 0.05 |
| gemini-3.1-pro = mimo-2.5-pro | 8.50 vs 8.50 | 0.00 |
| qwen-3.7-max ≈ qwen-3.6-plus | 9.45 vs 9.55 | 0.10 |
| kimi-2.6 / opus / glm-5.1 cluster | 5.75 / 6.30 / 6.60 | ≤ 0.85 |

Poster cases:

- **gpt-5.4** — Euclidean **#2** (dᵢ = 1.138) but Spearman **#4** (1−ρ = 0.100). Numerically a tight fit — its only sizeable miss is opus at rank 4 vs consensus 6.30. But it inverts the near-tied pairs: gpt-5.5 above deepseek-4-pro (consensus gap 0.05), qwen-3.6-plus above qwen-3.7-max (gap 0.10), and promoting opus also lifts it above kimi-2.6 and glm-5.1, adding two more inversions. Each inversion is almost free in squared-gap terms and full price in rank-correlation terms.
- **mimo-2.5-pro** — Spearman **#2** (1−ρ = 0.073) vs Euclidean **#3**; **qwen-3.7-max** — Spearman **#3** (0.088) vs Euclidean **#4**. The mirror image: consensus-consistent stories carrying a few large numeric misses that Euclidean squares — mimo-2.5-pro puts glm-5.1 at 9 vs consensus 6.60 (|gap| = 2.40) and gpt-5.5 at 2 vs 4.15; qwen-3.7-max puts gemini-3.1-pro at 11 vs 8.50 (2.50), opus at 4 vs 6.30 (2.30) and deepseek-4-pro at 2 vs 4.10 (2.10).
- **gpt-5.5** — #1 on both: right magnitudes *and* right order; when an evaluator gets both, the metrics cannot disagree.

Reading guide: Euclidean is the **primary** metric (it is the method's `dᵢ` with the confidence-weighted normalization); Spearman is the **cross-check**. When they disagree about an evaluator, the disagreement itself is informative: Euclidean-better means "right numbers, shuffled near-ties"; Spearman-better means "right story, one or two big numeric misses"."""

COMPARISON_DISCUSSION = """**Agreement across metrics.** The five orderings agree at the extremes of each track, but the two tracks measure different things:

- **gpt-5.5** owns the rank track: #1 on all three rank-based metrics (Euclidean dᵢ = 0.986, Spearman 1−ρ = 0.052, medoid Sᵢ = 1.780). By the method's primary definitions it is both the closest to the overall opinion and the most central evaluator.
- **qwen-3.7-max** owns the semantic track: #1 cosine-to-centroid (0.0029) and #1 cosine medoid (0.0099) — and it is also top-5 on every rank-track metric (#4 Euclidean / #3 Spearman / #5 medoid), making it the only evaluator in the top five of all five columns. If one evaluator had to stand in for the group across both tracks, it is qwen-3.7-max.
- **gemini-3.1-pro** is last on **all five metrics** — the unambiguous outlier. Its coarse 4-tier opinion distorts the rank vector (opus last vs consensus 6.30; the whole gpt-5.5/kimi-2.6 tier parked at 8.5), and its document is also the shortest and least typical text.
- **The tracks genuinely disagree in the middle.** deepseek-4-pro and kimi-2.6 — the rank track's bottom pair after gemini (#9/#10 on closeness, #10/#9 on medoid) — are semantically *central* (#3 and #4): their texts follow the shared template and vocabulary, but their orderings deviate (deepseek-4-pro crowns kimi-2.6 #1 and buries opus at #11; kimi-2.6 lifts gemini-3.1-pro to #4 and likewise buries opus). The mirror image: mimo-2.5-pro and glm-5.1 are strong on the rank track (closeness #3/#2 and #6/#6) but semantically peripheral (#9 and #10). Rank distance measures *what an evaluator concluded*; cosine distance measures *how it wrote* — and here the two are nearly uncorrelated.
- **The semantic track is flat and should be read as weak evidence.** Cosine distances to the centroid span only 0.0029–0.0120 and most adjacent gaps fall under ε = 0.002 (see the near-tie chains flagged in §3.2/§3.4): all eleven documents share format, headings, and subject matter. Only the endpoints — qwen-3.7-max clearly closest; glm-5.1 and gemini-3.1-pro clearly farthest — rise above the noise.

Per the spec, no combined ranking is computed: each metric stands on its own, and the table above is the complete per-metric picture."""


def build_report(rank: dict, sem: dict | None, run_date: str) -> str:
    R, C = rank["R"], rank["C"]
    cons, var = rank["cons"], rank["var"]
    eucl, rho, spear = rank["eucl"], rank["rho"], rank["spear"]
    D, S = rank["D"], rank["S"]

    eucl_o = ordered(EVALUATORS, eucl)
    spear_o = ordered(EVALUATORS, spear)
    medoid_o = ordered(EVALUATORS, S)

    w = []
    w.append("# Evaluator Consensus & Medoid Ranking — 11 Evaluators × 11 Proposals")
    w.append("")
    w.append(
        f"**Run date:** {run_date} · **Dataset:** the 11 remade `*-range.md` analyses "
        "(2026-06-11, including the new `fable-5` proposal/evaluator) · "
        "**Weights:** equal, wᵢ = 1/11 · **Generated by:** `consensus_ranking.py` "
        "(re-run it to regenerate every number below)."
    )
    w.append("")
    w.append("## What this answers")
    w.append("")
    w.append(
        "Each of 11 LLM evaluators ranked the same 11 architectural proposals "
        "(`docs/planner-graph-ref/proposals/`). This report ranks the **evaluators** by how "
        "representative their opinion is of the whole group, two ways, on two tracks:"
    )
    w.append("")
    w.append(
        "1. **Closest to overall** — smallest distance to the group consensus (method §5, `argmin dᵢ`)."
    )
    w.append(
        "2. **Medoid** — smallest total weighted distance to all other evaluators (method §6, "
        "`argmin Σⱼ wⱼ·d(Tᵢ,Tⱼ)`)."
    )
    w.append("")
    w.append(
        "Tracks: a **rank track** (proposals as aspects, ranks as scores) and a **semantic track** "
        "(Gemini text embeddings of each evaluator's full report)."
    )
    w.append("")
    w.append("## Method & parameters")
    w.append("")
    w.append(
        "Mapped to `analyse-evaluation-method.md` (formulas restated below; the variance/agreement "
        "definition sits in its §4 text-generation step):"
    )
    w.append("")
    w.append(
        "- **Scores** `sᵢⱼ`: raw ranks (1 = best … 11 = worst). The method's `[−1,1]` sentiment scale "
        "is an affine image of ranks; applied uniformly it changes no consensus value, distance, "
        "medoid, or ρ, so raw ranks are used for auditability."
    )
    w.append(
        "- **Confidence** `cᵢⱼ ∈ {0,1}` (§2): `cᵢⱼ = 0` on each evaluator's **own** proposal "
        "(self-ranking excluded — 11 cells); 1 elsewhere."
    )
    w.append(
        "- **Weighted center** (§2): `s̄ⱼ = (Σᵢ wᵢ cᵢⱼ sᵢⱼ) / (Σᵢ wᵢ cᵢⱼ)` — with equal wᵢ this is the "
        "mean rank over the 10 non-author evaluators."
    )
    w.append("- **Per-aspect agreement** (§4): `σ²ⱼ = (Σᵢ wᵢ cᵢⱼ (sᵢⱼ − s̄ⱼ)²) / (Σᵢ wᵢ cᵢⱼ)`.")
    w.append(
        "- **Closest to overall** (§5): normalized (confidence-weighted) Euclidean "
        "`dᵢ = √[ Σⱼ cᵢⱼ αⱼ (sᵢⱼ − s̄ⱼ)² / Σⱼ cᵢⱼ αⱼ ]` with αⱼ = 1 — an RMS over the 10 proposals "
        "evaluator i ranked, so evaluators omitting different cells stay comparable. "
        "Cross-check: Spearman `1 − ρ` on the same 10 cells."
    )
    w.append(
        "- **Medoid** (§6): pairwise `d(Tᵢ,Tⱼ)` = RMS Euclidean over the **9** proposals ranked by "
        "both (own(i) and own(j) excluded); totals `Sᵢ = Σⱼ wⱼ d(Tᵢ,Tⱼ)`."
    )
    w.append(
        f"- **Near-ties**: orderings are flagged when adjacent distances differ by less than "
        f"ε = {EPS_RANK} (rank track) / ε = {EPS_COS} (cosine). n = 11 is small; do not over-read "
        "hairline orderings."
    )
    w.append(
        "- **Semantic track**: paragraph-chunked documents (blank-line blocks; fenced code/Mermaid/"
        "tables kept intact; blocks < 5 tokens dropped), embedded with "
        f"`{EMBED_MODEL}` ({EMBED_DIM}-dim, task_type SEMANTIC_SIMILARITY), pooled per document by "
        "**length-weighted mean** (weight = chunk token count). Centroid = equal-weighted mean of the "
        "11 document vectors; cosine distance primary, Euclidean-to-centroid secondary. "
        "Self-exclusion does not apply (an evaluator's self-assessment cannot be excised from its text)."
    )
    w.append("")
    w.append("## 1. Rank matrix R")
    w.append("")
    w.append(
        "Rows = evaluators, columns = proposals, 1 = best. `—` marks the excluded self-ranking cell "
        "(cᵢⱼ = 0). The `gemini-3.1-pro` row is tier-imputed (table below)."
    )
    w.append("")
    header = ["Evaluator ↓ \\ Proposal →"] + [SHORT_P[p] for p in PROPOSALS]
    rows = []
    for i, e in enumerate(EVALUATORS):
        label = e + (" *(imputed)*" if e == "gemini-3.1-pro" else "")
        cells = []
        for j in range(N):
            cells.append("—" if C[i, j] == 0 else fmt_rank(R[i, j]))
        rows.append([label] + cells)
    w.append(md_table(header, rows))
    w.append("")
    w.append(
        "**gemini-3.1-pro imputation.** Its remade analysis groups all 11 proposals into 4 ranked "
        "tiers with no strict intra-tier order; each tier member receives the average of the slot "
        "positions the tier occupies (rank-consistent, row sums to 66 = Σ1..11):"
    )
    w.append("")
    w.append(
        md_table(
            ["Tier", "Members", "Slots", "Imputed rank"],
            [
                ['1 — "Phase-Based Pipeline"', "gpt-5.4, fable-5, deepseek-4-pro", "1–3", "**2**"],
                ['2 — "Minimalist Stage"', "mimo-2.5-pro, glm-5.1, gemini-3.1-pro", "4–6", "**5**"],
                [
                    '3 — "Micro-node Explosion"',
                    "qwen-3.7-max, gpt-5.5, kimi-2.6, qwen-3.6-plus",
                    "7–10",
                    "**8.5**",
                ],
                ['4 — "Edge-Heavy Anti-Pattern"', "opus", "11", "**11**"],
            ],
        )
    )
    w.append("")
    w.append(
        "**Self-ranking exclusion.** The 11 diagonal (author) cells are dropped from the consensus "
        "and every rank-track distance. The excluded values (for the record): "
        + ", ".join(
            f"{e}→{fmt_rank(RANKS[e][PROPOSALS.index(EVAL_TO_OWN[e])])}" for e in EVALUATORS
        )
        + ". Self-bias is visible — 7 of 11 evaluators put their own proposal in their top 5 "
        "(fable-5 and gpt-5.4 self-rank #1)."
    )
    w.append("")
    w.append("## 2. Rank track")
    w.append("")
    w.append("### 2.1 Consensus (weighted center) and per-proposal agreement")
    w.append("")
    w.append(
        "`s̄ⱼ` = mean rank over each proposal's 10 non-author rankers; `σ²ⱼ` = confidence-weighted "
        "variance (lower = stronger agreement)."
    )
    w.append("")
    cons_order = np.argsort(cons)
    rows = []
    for pos, j in enumerate(cons_order, start=1):
        rows.append(
            [str(pos), PROPOSALS[j], fmt(cons[j], 2), fmt(var[j], 2), fmt(np.sqrt(var[j]), 2)]
        )
    w.append(md_table(["Consensus #", "Proposal", "s̄ⱼ (mean rank)", "σ²ⱼ", "σⱼ"], rows))
    w.append("")
    w.append(
        "(Derived by-product: the proposals' own consensus order. The deliverable remains the "
        "evaluator rankings below.)"
    )
    w.append("")
    w.append("### 2.2 Closest to overall — normalized Euclidean dᵢ (primary)")
    w.append("")
    w.append(
        "`dᵢ = √[ Σⱼ cᵢⱼ (sᵢⱼ − s̄ⱼ)² / Σⱼ cᵢⱼ ]` — RMS over evaluator i's 10 ranked proposals."
    )
    w.append("")
    w.append(ranking_table(eucl_o, "dᵢ (RMS rank units)"))
    w.append("")
    w.append(tie_note(eucl_o, EPS_RANK))
    w.append("")
    w.append("### 2.3 Closest to overall — Spearman cross-check")
    w.append("")
    w.append(
        "`1 − ρ` between evaluator i's 10 ranks and the consensus restricted to those proposals."
    )
    w.append("")
    rho_by_name = dict(zip(EVALUATORS, rho.tolist(), strict=True))
    rows = [[str(i + 1), n, fmt(rho_by_name[n]), fmt(v)] for i, (n, v) in enumerate(spear_o)]
    w.append(md_table(["#", "Evaluator", "ρ", "1 − ρ"], rows))
    w.append("")
    w.append(tie_note(spear_o, EPS_RANK))
    w.append("")
    w.append("### 2.4 Pairwise distance matrix (RMS Euclidean, 9 common proposals)")
    w.append("")
    w.append(matrix_table(D, EVALUATORS, nd=4))
    w.append("")
    w.append("### 2.5 Medoid (rank track)")
    w.append("")
    w.append("`Sᵢ = Σⱼ wⱼ·d(Tᵢ,Tⱼ)` with wⱼ = 1/11 (self-distance 0). Smallest total = medoid.")
    w.append("")
    w.append(ranking_table(medoid_o, "Sᵢ (weighted total)"))
    w.append("")
    w.append(tie_note(medoid_o, EPS_RANK))
    w.append("")
    w.append(SECTION_2_6 if SECTION_2_6 else "_(§2.6 pending — filled in the final pass)_")
    w.append("")

    if sem is not None:
        dcos, deuc = sem["dcos"], sem["deuc"]
        Dcos, Scos = sem["Dcos"], sem["Scos"]
        cos_o = ordered(EVALUATORS, dcos)
        cmed_o = ordered(EVALUATORS, Scos)
        w.append("## 3. Semantic track (Gemini embeddings)")
        w.append("")
        w.append(
            f"Model `{EMBED_MODEL}` ({EMBED_DIM}-dim, task_type SEMANTIC_SIMILARITY), batched "
            f"requests ≤ {BATCH_SIZE} inputs, throttle {SLEEP_SECONDS}s per "
            f"{EMBED_CALLS_BEFORE_SLEEP} requests + {INTER_BATCH_SLEEP:.0f}s between batches. "
            "Self-exclusion not applicable (full text embedded)."
        )
        w.append("")
        w.append("### 3.1 Chunking statistics")
        w.append("")
        rows = [
            [e, str(nraw), str(nkept), str(tok), str(mx)]
            for e, nraw, nkept, tok, mx in sem["stats"]
        ]
        rows.append(
            [
                "**total**",
                str(sum(r[1] for r in sem["stats"])),
                f"**{sem['total_chunks']}**",
                f"**{sem['total_tokens']}**",
                "",
            ]
        )
        w.append(
            md_table(
                ["Document", "Raw blocks", "Kept chunks (≥5 tok)", "Tokens", "Max chunk"], rows
            )
        )
        w.append("")
        w.append(
            f"Token counts via {sem['token_note']}; per-input limit {MAX_INPUT_TOKENS} tokens "
            "(no chunk required truncation if max chunk below the limit)."
        )
        w.append("")
        w.append("### 3.2 Closest to centroid — cosine (primary), Euclidean (secondary)")
        w.append("")
        rows = []
        for i, (n, v) in enumerate(cos_o):
            idx = EVALUATORS.index(n)
            rows.append([str(i + 1), n, fmt(v), fmt(deuc[idx])])
        w.append(
            md_table(
                ["#", "Evaluator", "cosine distance to centroid", "Euclidean to centroid"], rows
            )
        )
        w.append("")
        w.append(tie_note(cos_o, EPS_COS))
        w.append("")
        w.append("### 3.3 Pairwise cosine distance matrix")
        w.append("")
        w.append(matrix_table(Dcos, EVALUATORS, nd=4))
        w.append("")
        w.append("### 3.4 Cosine medoid")
        w.append("")
        w.append(ranking_table(cmed_o, "Sᵢ (weighted cosine total)"))
        w.append("")
        w.append(tie_note(cmed_o, EPS_COS))
        w.append("")
        w.append("## 4. Per-metric comparison (no combined winner)")
        w.append("")
        pos_e = {n: i + 1 for i, (n, _) in enumerate(eucl_o)}
        pos_s = {n: i + 1 for i, (n, _) in enumerate(spear_o)}
        pos_c = {n: i + 1 for i, (n, _) in enumerate(cos_o)}
        pos_m = {n: i + 1 for i, (n, _) in enumerate(medoid_o)}
        pos_cm = {n: i + 1 for i, (n, _) in enumerate(cmed_o)}
        rows = [
            [e, str(pos_e[e]), str(pos_s[e]), str(pos_c[e]), str(pos_m[e]), str(pos_cm[e])]
            for e in EVALUATORS
        ]
        w.append(
            md_table(
                [
                    "Evaluator",
                    "Closest: rank-Euclidean",
                    "Closest: Spearman",
                    "Closest: cosine",
                    "Medoid: rank-Euclidean",
                    "Medoid: cosine",
                ],
                rows,
            )
        )
        w.append("")
        w.append(
            COMPARISON_DISCUSSION
            if COMPARISON_DISCUSSION
            else "_(§4 discussion pending — filled in the final pass)_"
        )
        w.append("")
    w.append("## 5. Notes & caveats")
    w.append("")
    w.append(
        "- Two missing-data policies by design: gemini-3.1-pro's tier structure is **imputed** "
        "(completing a coarse-but-total opinion); self-rankings are **excluded** (cᵢⱼ = 0, removing "
        "self-bias). Both flow through the method's confidence term."
    )
    w.append(
        "- n = 11 with one imputed (tiered) row: orderings near the flagged ties are not robust; "
        "the extremes (top and bottom two on each metric) are."
    )
    w.append(
        "- Embedding vectors are not hand-verifiable; the derived distance matrices above are the "
        "auditable artifact. Re-running re-embeds (or reuses a content-keyed cache) and reproduces "
        "the rank track exactly."
    )
    w.append(
        "- Self-exclusion applies to the rank track only; the semantic track embeds each report's "
        "full text including its self-assessment."
    )
    w.append(
        "- Reproduce: `poetry run python3 docs/planner-graph-ref/analyse/consensus_ranking.py` "
        "(project venv; `GOOGLE_API_KEY` required for the semantic track)."
    )
    w.append("")
    return "\n".join(w)


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank-only", action="store_true")
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    rank = rank_track()
    log(
        "[rank] consensus s̄ⱼ: "
        + ", ".join(f"{SHORT_P[p]}={fmt(rank['cons'][j], 2)}" for j, p in enumerate(PROPOSALS))
    )
    for label, vals in (
        ("eucl dᵢ", rank["eucl"]),
        ("spear 1−ρ", rank["spear"]),
        ("medoid Sᵢ", rank["S"]),
    ):
        o = ordered(EVALUATORS, vals)
        log(f"[rank] {label}: " + " < ".join(f"{n}({fmt(v, 3)})" for n, v in o))

    sem = None
    if not args.rank_only:
        try:
            sem = semantic_track()
            for label, vals in (("cosine dᵢ", sem["dcos"]), ("cos-medoid Sᵢ", sem["Scos"])):
                o = ordered(EVALUATORS, vals)
                log(f"[sem] {label}: " + " < ".join(f"{n}({fmt(v, 4)})" for n, v in o))
        except Exception:
            log("[sem] FAILED:\n" + traceback.format_exc())
            return 2

    report = build_report(rank, sem, run_date=str(date.today()))
    REPORT_PATH.write_text(report, encoding="utf-8")
    log(f"[done] report written to {REPORT_PATH} ({len(report)} chars)")
    if args.stdout:
        print("===REPORT START===")
        print(report)
        print("===REPORT END===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
