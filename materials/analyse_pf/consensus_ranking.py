"""Evaluator consensus & medoid ranking — executes the deep-interview spec
(.omc/specs/deep-interview-evaluator-consensus-ranking.md).

Two tracks, equal evaluator weights (wᵢ = 1/n):
  * Rank track  — proposals as aspects; self-rankings excluded (cᵢⱼ = 0);
                  confidence-weighted consensus; normalized (RMS) Euclidean
                  closest-to-overall + Spearman cross-check; RMS-Euclidean medoid.
  * Semantic    — paragraph-chunk each report (code/tables intact, drop <5-token
                  blocks), embed chunks with Gemini gemini-embedding-001 (batched),
                  length-weighted-mean pool to one vector/doc; cosine distance to
                  centroid + cosine medoid (Euclidean-to-centroid secondary).

Prints the complete report markdown to stdout between REPORT markers; logs to stderr.
Run: poetry run python3 docs/planner-graph-ref/analyse/consensus_ranking.py
"""

from __future__ import annotations

import datetime as _dt
import re
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

EMBED_MODEL = "gemini-embedding-001"
BATCH_SIZE = 50
MIN_BLOCK_TOKENS = 5
SLEEP_BETWEEN_BATCHES = 2.0
EMBED_CALLS_BEFORE_SLEEP = 50
SLEEP_SECONDS = 60

# Near-tie thresholds (flag adjacent ranks within these gaps).
EPS = {"rank_eucl": 0.05, "spearman": 0.02, "cosine": 0.003}

PROPOSALS = [
    "deepseek-4-pro", "gpt-5.4", "gpt-5.5", "glm-5.1", "kimi-2.6",
    "opus", "qwen-3.6-plus", "qwen-3.7-max", "mimo-2.5-pro", "gemini-3.1-pro",
]
PSHORT = {  # column headers
    "deepseek-4-pro": "ds4", "gpt-5.4": "g54", "gpt-5.5": "g55", "glm-5.1": "glm",
    "kimi-2.6": "kimi", "opus": "opus", "qwen-3.6-plus": "q36", "qwen-3.7-max": "q37",
    "mimo-2.5-pro": "mimo", "gemini-3.1-pro": "gem",
}

# Evaluator (range-file stem) -> {proposal: rank}; gemini row imputed (see spec).
RANKS: dict[str, dict[str, float]] = {
    "deepseek-4-pro": {"deepseek-4-pro": 1, "gpt-5.4": 2, "gpt-5.5": 4, "glm-5.1": 6, "kimi-2.6": 5, "opus": 3, "qwen-3.6-plus": 9, "qwen-3.7-max": 7, "mimo-2.5-pro": 8, "gemini-3.1-pro": 10},
    "glm-5.1-pro": {"deepseek-4-pro": 2, "gpt-5.4": 4, "gpt-5.5": 1, "glm-5.1": 6, "kimi-2.6": 5, "opus": 3, "qwen-3.6-plus": 9, "qwen-3.7-max": 10, "mimo-2.5-pro": 8, "gemini-3.1-pro": 7},
    "gpt-5.4": {"deepseek-4-pro": 4, "gpt-5.4": 1, "gpt-5.5": 2, "glm-5.1": 3, "kimi-2.6": 5, "opus": 6, "qwen-3.6-plus": 8, "qwen-3.7-max": 10, "mimo-2.5-pro": 9, "gemini-3.1-pro": 7},
    "gpt-5.5": {"deepseek-4-pro": 4, "gpt-5.4": 1, "gpt-5.5": 2, "glm-5.1": 3, "kimi-2.6": 5, "opus": 6, "qwen-3.6-plus": 10, "qwen-3.7-max": 7, "mimo-2.5-pro": 9, "gemini-3.1-pro": 8},
    "kimi-2.6": {"deepseek-4-pro": 1, "gpt-5.4": 2, "gpt-5.5": 9, "glm-5.1": 4, "kimi-2.6": 3, "opus": 8, "qwen-3.6-plus": 7, "qwen-3.7-max": 10, "mimo-2.5-pro": 5, "gemini-3.1-pro": 6},
    "mimo-2.5-pro": {"deepseek-4-pro": 1, "gpt-5.4": 2, "gpt-5.5": 5, "glm-5.1": 3, "kimi-2.6": 4, "opus": 9, "qwen-3.6-plus": 7, "qwen-3.7-max": 10, "mimo-2.5-pro": 8, "gemini-3.1-pro": 6},
    "opus-4.7": {"deepseek-4-pro": 3, "gpt-5.4": 1, "gpt-5.5": 2, "glm-5.1": 4, "kimi-2.6": 5, "opus": 8, "qwen-3.6-plus": 6, "qwen-3.7-max": 7, "mimo-2.5-pro": 9, "gemini-3.1-pro": 10},
    "qwen-3.6": {"deepseek-4-pro": 5, "gpt-5.4": 1, "gpt-5.5": 10, "glm-5.1": 2, "kimi-2.6": 8, "opus": 3, "qwen-3.6-plus": 9, "qwen-3.7-max": 4, "mimo-2.5-pro": 6, "gemini-3.1-pro": 7},
    "qwen-3.7-max": {"deepseek-4-pro": 2, "gpt-5.4": 1, "gpt-5.5": 3, "glm-5.1": 4, "kimi-2.6": 5, "opus": 6, "qwen-3.6-plus": 7, "qwen-3.7-max": 8, "mimo-2.5-pro": 10, "gemini-3.1-pro": 9},
    "gemini-3.1-pro": {"deepseek-4-pro": 2, "gpt-5.4": 1, "gpt-5.5": 7.5, "glm-5.1": 4, "kimi-2.6": 7.5, "opus": 7.5, "qwen-3.6-plus": 7.5, "qwen-3.7-max": 10, "mimo-2.5-pro": 4, "gemini-3.1-pro": 4},
}
EVALUATORS = list(RANKS.keys())
N = len(EVALUATORS)
W = np.full(N, 1.0 / N)

EVAL_TO_OWN = {
    "deepseek-4-pro": "deepseek-4-pro", "glm-5.1-pro": "glm-5.1", "gpt-5.4": "gpt-5.4",
    "gpt-5.5": "gpt-5.5", "kimi-2.6": "kimi-2.6", "mimo-2.5-pro": "mimo-2.5-pro",
    "opus-4.7": "opus", "qwen-3.6": "qwen-3.6-plus", "qwen-3.7-max": "qwen-3.7-max",
    "gemini-3.1-pro": "gemini-3.1-pro",
}
GEMINI_IMPUTED = "gemini-3.1-pro"  # evaluator whose row is imputed

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")
    def ntok(s: str) -> int: return len(_enc.encode(s))
    TOKENIZER = "tiktoken/cl100k_base"
except Exception:  # pragma: no cover
    def ntok(s: str) -> int: return max(1, len(s) // 4)
    TOKENIZER = "char/4 estimate"


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# RANK TRACK
# --------------------------------------------------------------------------- #
def rank_track():
    R = np.array([[float(RANKS[e][p]) for p in PROPOSALS] for e in EVALUATORS])  # (N,M)
    C = np.array([[0.0 if p == EVAL_TO_OWN[e] else 1.0 for p in PROPOSALS] for e in EVALUATORS])
    M = len(PROPOSALS)

    # consensus per proposal: weighted mean over non-author evaluators
    consensus = np.array([
        (W * C[:, j] * R[:, j]).sum() / (W * C[:, j]).sum() for j in range(M)
    ])
    variance = np.array([
        (W * C[:, j] * (R[:, j] - consensus[j]) ** 2).sum() / (W * C[:, j]).sum() for j in range(M)
    ])

    # closest-to-overall: normalized (RMS) Euclidean over each evaluator's rated proposals
    eucl = np.array([
        np.sqrt((C[i] * (R[i] - consensus) ** 2).sum() / C[i].sum()) for i in range(N)
    ])

    # Spearman cross-check on each evaluator's 9 rated proposals vs consensus
    spear_rho = np.empty(N)
    for i in range(N):
        mask = C[i] > 0
        res = spearmanr(R[i][mask], consensus[mask])
        spear_rho[i] = float(getattr(res, "statistic", res[0]))
    spear_dist = 1.0 - spear_rho

    # pairwise RMS Euclidean over commonly-rated proposals (exclude both authors)
    D = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            if i == k:
                continue
            common = (C[i] > 0) & (C[k] > 0)
            D[i, k] = np.sqrt(((R[i][common] - R[k][common]) ** 2).sum() / common.sum())
    medoid = D @ W  # weighted total distance to others

    return dict(R=R, C=C, consensus=consensus, variance=variance, eucl=eucl,
                spear_rho=spear_rho, spear_dist=spear_dist, D=D, medoid=medoid)


# --------------------------------------------------------------------------- #
# SEMANTIC TRACK
# --------------------------------------------------------------------------- #
def split_blocks(text: str) -> list[str]:
    """Blank-line blocks, keeping fenced code/Mermaid intact (blank lines inside a
    ``` fence do not split). Tables have no internal blank lines so stay whole."""
    blocks, cur, in_fence = [], [], False
    for line in text.split("\n"):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            cur.append(line)
            continue
        if in_fence:
            cur.append(line)
            continue
        if line.strip() == "":
            if cur:
                blocks.append("\n".join(cur).strip())
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur).strip())
    return [b for b in blocks if b.strip()]


def load_google_key() -> str:
    for line in (REPO_ROOT / ".env").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("GOOGLE_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("GOOGLE_API_KEY not found in .env")


def embed_batch(client, batch, attempt=0):
    from google.genai import types
    try:
        try:
            resp = client.models.embed_content(
                model=EMBED_MODEL, contents=batch,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"))
        except TypeError:
            resp = client.models.embed_content(model=EMBED_MODEL, contents=batch)
        return [list(e.values) for e in resp.embeddings]
    except Exception as e:  # pragma: no cover - network/limits
        msg = str(e).lower()
        if attempt < 6 and ("resource_exhausted" in msg or "429" in msg or "quota" in msg or "rate" in msg):
            wait = min(70, 10 * (attempt + 1))
            log(f"  backoff {wait}s ({type(e).__name__}) attempt {attempt+1}")
            time.sleep(wait)
            return embed_batch(client, batch, attempt + 1)
        raise


def semantic_track():
    from google import genai
    client = genai.Client(api_key=load_google_key())

    # build chunk list with provenance
    chunk_texts, chunk_doc, chunk_w, stats = [], [], [], []
    for di, e in enumerate(EVALUATORS):
        text = (HERE / f"{e}-range.md").read_text(encoding="utf-8")
        raw = split_blocks(text)
        kept = [(b, ntok(b)) for b in raw if ntok(b) >= MIN_BLOCK_TOKENS]
        stats.append((e, len(raw), len(kept), sum(t for _, t in kept)))
        for b, t in kept:
            chunk_texts.append(b)
            chunk_doc.append(di)
            chunk_w.append(float(t))
    log(f"semantic: {len(chunk_texts)} chunks across {N} docs; embedding (batch={BATCH_SIZE})…")

    vecs, req = [], 0
    for s in range(0, len(chunk_texts), BATCH_SIZE):
        vecs.extend(embed_batch(client, chunk_texts[s:s + BATCH_SIZE]))
        req += 1
        log(f"  batch {req} done ({min(s + BATCH_SIZE, len(chunk_texts))}/{len(chunk_texts)})")
        if req % EMBED_CALLS_BEFORE_SLEEP == 0:
            time.sleep(SLEEP_SECONDS)
        else:
            time.sleep(SLEEP_BETWEEN_BATCHES)
    E = np.array(vecs)
    dim = E.shape[1]

    # length-weighted mean pool per document
    doc_vec = np.zeros((N, dim))
    wsum = np.zeros(N)
    for p in range(len(chunk_texts)):
        d = chunk_doc[p]
        doc_vec[d] += chunk_w[p] * E[p]
        wsum[d] += chunk_w[p]
    doc_vec /= wsum[:, None]

    centroid = doc_vec.mean(axis=0)  # equal weights

    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

    cos_cent = np.array([1.0 - cos(doc_vec[i], centroid) for i in range(N)])
    eucl_cent = np.array([float(np.linalg.norm(doc_vec[i] - centroid)) for i in range(N)])
    Dcos = np.array([[1.0 - cos(doc_vec[i], doc_vec[k]) for k in range(N)] for i in range(N)])
    np.fill_diagonal(Dcos, 0.0)  # avoid -0.0000 float noise on the diagonal
    cos_medoid = Dcos @ W

    return dict(dim=dim, stats=stats, n_chunks=len(chunk_texts), cos_cent=cos_cent,
                eucl_cent=eucl_cent, Dcos=Dcos, cos_medoid=cos_medoid)


# --------------------------------------------------------------------------- #
# rendering helpers
# --------------------------------------------------------------------------- #
def order(vals, ascending=True):
    return list(np.argsort(vals if ascending else -np.asarray(vals), kind="stable"))


def ranked_list(vals, eps, fmt="{:.4f}"):
    idx = order(vals)
    lines, prev = [], None
    for pos, i in enumerate(idx, 1):
        tie = "  ‹≈ tie with above›" if prev is not None and abs(vals[i] - prev) < eps else ""
        lines.append(f"| {pos} | {EVALUATORS[i]} | {fmt.format(vals[i])} |{tie}")
        prev = vals[i]
    return idx, "\n".join(lines)


def positions(vals, ascending=True):
    idx = order(vals, ascending)
    pos = {}
    for p, i in enumerate(idx, 1):
        pos[i] = p
    return pos


def md_matrix(Mx, fmt="{:.3f}"):
    head = "| eval \\ eval | " + " | ".join(PSHORT.get(EVAL_TO_OWN[e], e[:4]) for e in EVALUATORS) + " |"
    sep = "|" + "---|" * (N + 1)
    rows = []
    for i, e in enumerate(EVALUATORS):
        rows.append("| " + e + " | " + " | ".join(fmt.format(Mx[i, k]) for k in range(N)) + " |")
    return "\n".join([head, sep] + rows)


SECTION_2_6 = """### 2.6 Why the two closest-to-overall rankings differ (Euclidean vs Spearman)

The two metrics answer different questions, so they rank evaluators differently:

- **Euclidean (RMS, §2.2)** measures the *magnitude* of rank deviations. It squares each gap, so it is driven by the single largest miss and is nearly indifferent to swapping two proposals the consensus treats as near-equal.
- **Spearman (1−ρ, §2.3)** measures *ordinal* agreement. Every inversion costs the same regardless of how close the two proposals are, and at n = 9 each one moves ρ noticeably.

This consensus has **near-tied clusters** — exactly what splits the two metrics (a swap inside a cluster is ~free for Euclidean but a full penalty for Spearman):

| near-tie | consensus mean-ranks | gap |
|---|---|---|
| kimi-2.6 ≈ opus | 5.50 / 5.72 | 0.22 |
| mimo-2.5-pro ≈ gemini-3.1-pro ≈ qwen-3.6-plus | 7.56 / 7.78 / 7.83 | ≤ 0.27 |

Two illustrative cases:

- **`gpt-5.5` — Euclidean #1, Spearman #4.** Numerically the closest opinion (largest single miss only 2.17, on qwen-3.6-plus), yet its 5 order-inversions are almost all swaps *inside* near-ties — mimo/gemini (gap 0.22), qwen-3.6/qwen-3.7 (0.50), qwen-3.7/gemini (0.56), qwen-3.7/mimo (0.78). Euclidean ignores them; Spearman counts each. *It nails the numbers but shuffles the tied clusters.*
- **`mimo-2.5-pro` — Spearman #1, Euclidean #6.** Its ordering is nearly perfect (4 inversions), but one large magnitude miss — opus = 9 vs consensus 5.72 (|d| = 3.28) — dominates its squared Euclidean distance. *It gets the order right but badly misplaces one proposal.*

For contrast, **`qwen-3.7-max` is #2 on both** (good order *and* good magnitudes — its few inversions are all within the ≤0.28-gap cluster), and **`deepseek-4-pro`** is weak on both (Euclidean #5, Spearman #8: both big magnitude misses *and* the most inversions, 9/36).

**Reading guide.** Euclidean answers *"whose actual ranking is numerically closest to consensus?"*; Spearman answers *"who got the gross ordering right?"*. Because the consensus genuinely contains near-ties (swapping them *should* be near-free), Euclidean is the more faithful closeness measure here — which is why it is the **primary** and Spearman the cross-check.
"""


def main():
    today = _dt.date.today().isoformat()
    rt = rank_track()
    log("rank track computed.")
    try:
        st = semantic_track()
        sem_ok = True
    except Exception as exc:  # pragma: no cover
        import traceback
        log("SEMANTIC TRACK FAILED:\n" + traceback.format_exc())
        st, sem_ok = None, False

    R, C, cons, var = rt["R"], rt["C"], rt["consensus"], rt["variance"]
    out = []
    w = out.append

    w("===REPORT START===")
    w("# Evaluator Consensus & Medoid Ranking — Planner-Graph Proposal Reviews\n")
    w(f"*Generated {today}. Method: `analyse-evaluation-method.md`. "
      f"n = {N} evaluators, equal weights wᵢ = {1/N:.2f}. "
      f"Spec: `.omc/specs/deep-interview-evaluator-consensus-ranking.md`.*\n")

    w("## What this answers\n")
    w("Each of the 10 `*-range.md` files is an **evaluator** that ranked the same 10 architectural "
      "**proposals**. Treating each evaluator's ranking as their *opinion*, this report finds, two ways:\n")
    w("1. **Closest opinion to the overall** — whose ranking is nearest the weighted consensus (`argmin dᵢ`).")
    w("2. **Medoid** — whose ranking is nearest *all other* rankings (`argmin Σⱼ wⱼ·d(Tᵢ,Tⱼ)`).\n")
    w("Two independent tracks are reported side by side with no forced single winner: a **rank track** "
      "(aspect-based: proposals are the aspects) and a **semantic track** (Gemini embeddings of the report text).\n")

    w("## Method & parameters\n")
    w("- **Weights:** equal, wᵢ = 1/10 (method §2/§8).")
    w("- **Self-rankings excluded** (rank track): each evaluator's rank of its *own* proposal is dropped "
      "via the method's confidence term cᵢⱼ = 0 (10 cells). Consensus and distances use the remaining 9.")
    w("- **gemini-3.1-pro** gave tiers for only 6 proposals; its row is **imputed** with rank-consistent "
      "tier-averages (row sums to 55).")
    w("- **Scores** are raw ranks (1 = best … 10 = worst). The method's [−1,1] sentiment scale is an affine "
      "image, so every consensus value, distance, medoid and Spearman ρ is identical — ranks are used directly.")
    w("- **Rank-track distances:** consensus `s̄ⱼ = Σwᵢcᵢⱼsᵢⱼ / Σwᵢcᵢⱼ` (§2); closest = normalized (RMS) "
      "weighted Euclidean `dᵢ = √[Σⱼcᵢⱼ(sᵢⱼ−s̄ⱼ)² / Σⱼcᵢⱼ]` (§5, αⱼ=1) + Spearman 1−ρ cross-check; "
      "medoid = `argmin Σⱼwⱼd(Tᵢ,Tⱼ)` (§6), pairwise RMS Euclidean over commonly-rated proposals.")
    w(f"- **Semantic track:** paragraph-chunk each report (code/tables intact, drop <{MIN_BLOCK_TOKENS}-token "
      f"blocks), embed chunks with Gemini `{EMBED_MODEL}`, length-weighted-mean pool to one vector/doc; "
      "cosine distance to the centroid (primary) + Euclidean (secondary); cosine medoid. "
      f"Token counts via {TOKENIZER}.\n")

    # ---- input matrix ----
    w("## 1. Input — rank matrix R (1 = best; — = self-ranking excluded)\n")
    head = "| Evaluator ↓ \\ Proposal → | " + " | ".join(PSHORT[p] for p in PROPOSALS) + " |"
    w(head)
    w("|" + "---|" * (len(PROPOSALS) + 1))
    for i, e in enumerate(EVALUATORS):
        cells = []
        for j, p in enumerate(PROPOSALS):
            cells.append("—" if C[i, j] == 0 else (f"{R[i,j]:.1f}".rstrip("0").rstrip(".")))
        tag = " *(imputed)*" if e == GEMINI_IMPUTED else ""
        w(f"| {e}{tag} | " + " | ".join(cells) + " |")
    w("")
    w("Column keys: " + ", ".join(f"`{PSHORT[p]}`={p}" for p in PROPOSALS) + ".\n")

    # ---- rank track ----
    w("## 2. Rank track\n")
    w("### 2.1 Weighted consensus & agreement (per proposal, over its 9 non-author rankers)\n")
    w("| Proposal | consensus mean-rank | variance σ²ⱼ |")
    w("|---|---|---|")
    for j in order(cons):
        w(f"| {PROPOSALS[j]} | {cons[j]:.3f} | {var[j]:.3f} |")
    w("")
    cons_order = order(cons)
    w("**Consensus ordering of proposals** (best→worst): "
      + " → ".join(PROPOSALS[j] for j in cons_order) + ".")
    hi = max(range(len(PROPOSALS)), key=lambda j: var[j])
    lo = min(range(len(PROPOSALS)), key=lambda j: var[j])
    w(f"Strongest agreement on **{PROPOSALS[lo]}** (σ²={var[lo]:.3f}); most contested is "
      f"**{PROPOSALS[hi]}** (σ²={var[hi]:.3f}).\n")

    w("### 2.2 Closest to overall — normalized Euclidean to consensus *(primary)*\n")
    idx_e, body = ranked_list(rt["eucl"], EPS["rank_eucl"])
    w("| # | Evaluator | dᵢ (RMS) |")
    w("|---|---|---|")
    w(body)
    w(f"\n→ **Closest to the overall (rank track): `{EVALUATORS[idx_e[0]]}`** "
      f"(dᵢ = {rt['eucl'][idx_e[0]]:.4f}); farthest: `{EVALUATORS[idx_e[-1]]}` "
      f"({rt['eucl'][idx_e[-1]]:.4f}).\n")

    w("### 2.3 Closest to overall — Spearman 1−ρ *(cross-check)*\n")
    idx_s, body = ranked_list(rt["spear_dist"], EPS["spearman"])
    w("| # | Evaluator | 1−ρ | ρ |")
    w("|---|---|---|---|")
    for pos, i in enumerate(order(rt["spear_dist"]), 1):
        w(f"| {pos} | {EVALUATORS[i]} | {rt['spear_dist'][i]:.4f} | {rt['spear_rho'][i]:.4f} |")
    w("")

    w("### 2.4 Pairwise Euclidean distance matrix (RMS, over common proposals)\n")
    w(md_matrix(rt["D"], "{:.3f}"))
    w("")

    w("### 2.5 Medoid — minimum weighted total distance to all others *(rank track)*\n")
    idx_m, body = ranked_list(rt["medoid"], EPS["rank_eucl"])
    w("| # | Evaluator | Sᵢ = Σⱼwⱼd(i,j) |")
    w("|---|---|---|")
    w(body)
    w(f"\n→ **Rank-track medoid: `{EVALUATORS[idx_m[0]]}`** (Sᵢ = {rt['medoid'][idx_m[0]]:.4f}).\n")

    w(SECTION_2_6)

    # ---- semantic ----
    if sem_ok:
        w(f"## 3. Semantic track — Gemini `{EMBED_MODEL}` ({st['dim']}-dim)\n")
        w("### 3.1 Chunking (blank-line blocks; code/tables intact; <5-token blocks dropped)\n")
        w("| Evaluator | raw blocks | chunks kept | tokens |")
        w("|---|---|---|---|")
        for e, raw, kept, tk in st["stats"]:
            w(f"| {e} | {raw} | {kept} | {tk} |")
        w(f"\nTotal embedded chunks: **{st['n_chunks']}**; pooled by length-weighted mean to one vector/doc.\n")

        w("### 3.2 Closest to overall — cosine distance to centroid *(primary)*\n")
        idx_c, body = ranked_list(st["cos_cent"], EPS["cosine"], "{:.6f}")
        w("| # | Evaluator | cosine dist |")
        w("|---|---|---|")
        w(body)
        w(f"\n→ **Closest to the overall (semantic): `{EVALUATORS[idx_c[0]]}`** "
          f"(cos = {st['cos_cent'][idx_c[0]]:.6f}); farthest: `{EVALUATORS[idx_c[-1]]}`.\n")
        w("Euclidean-to-centroid (secondary): " +
          ", ".join(f"{EVALUATORS[i]} {st['eucl_cent'][i]:.3f}" for i in order(st["eucl_cent"])) + ".\n")

        w("### 3.3 Pairwise cosine distance matrix\n")
        w(md_matrix(st["Dcos"], "{:.4f}"))
        w("")

        w("### 3.4 Cosine medoid — minimum total cosine distance to others\n")
        idx_cm, body = ranked_list(st["cos_medoid"], EPS["cosine"], "{:.6f}")
        w("| # | Evaluator | Sᵢ (cosine) |")
        w("|---|---|---|")
        w(body)
        w(f"\n→ **Semantic medoid: `{EVALUATORS[idx_cm[0]]}`** (Sᵢ = {st['cos_medoid'][idx_cm[0]]:.6f}).\n")
    else:
        w("## 3. Semantic track — UNAVAILABLE\n")
        w("> The Gemini embedding call failed at run time; only the rank track is reported above.\n")

    # ---- comparison ----
    w("## 4. Per-metric comparison (no forced single winner)\n")
    pe = positions(rt["eucl"]); ps = positions(rt["spear_dist"])
    pme = positions(rt["medoid"])
    if sem_ok:
        pc = positions(st["cos_cent"]); pcm = positions(st["cos_medoid"])
    w("**Closest-to-overall — position by metric** (1 = closest):\n")
    w("| Evaluator | rank-Euclidean | Spearman | " + ("cosine |" if sem_ok else "") )
    w("|---|---|---|" + ("---|" if sem_ok else ""))
    for i in order(rt["eucl"]):
        row = f"| {EVALUATORS[i]} | {pe[i]} | {ps[i]} |"
        if sem_ok:
            row += f" {pc[i]} |"
        w(row)
    w("")
    w("**Medoid — position by metric** (1 = most central):\n")
    w("| Evaluator | rank-Euclidean | " + ("cosine |" if sem_ok else ""))
    w("|---|---|" + ("---|" if sem_ok else ""))
    for i in order(rt["medoid"]):
        row = f"| {EVALUATORS[i]} | {pme[i]} |"
        if sem_ok:
            row += f" {pcm[i]} |"
        w(row)
    w("")

    # agreement observations
    closest_rank = EVALUATORS[order(rt['eucl'])[0]]
    closest_spear = EVALUATORS[order(rt['spear_dist'])[0]]
    medoid_rank = EVALUATORS[order(rt['medoid'])[0]]
    obs = [f"Rank-track **closest** and **medoid** agree on `{medoid_rank}`/`{closest_rank}`"
           + (" (same)." if medoid_rank == closest_rank else " (differ).")]
    obs.append(f"Euclidean and Spearman closest agree: {'yes — both ' + closest_rank if closest_rank == closest_spear else 'no (' + closest_rank + ' vs ' + closest_spear + ')'}.")
    if sem_ok:
        closest_cos = EVALUATORS[order(st['cos_cent'])[0]]
        medoid_cos = EVALUATORS[order(st['cos_medoid'])[0]]
        obs.append(f"Semantic **closest** = `{closest_cos}`, semantic **medoid** = `{medoid_cos}`.")
        obs.append("Rank vs semantic closest "
                   + ("agree." if closest_cos == closest_rank else f"differ (`{closest_rank}` rank vs `{closest_cos}` semantic) — expected, since they measure different things (ordinal agreement vs textual similarity)."))
    w("**Observations:** " + " ".join(obs) + "\n")

    # ---- notes ----
    w("## 5. Notes & caveats\n")
    w("**Self-ranking bias (observation).** Self-rankings were excluded from the math; for the record, the "
      "rank each evaluator gave its *own* proposal:\n")
    w("| Evaluator | own proposal | self-rank |")
    w("|---|---|---|")
    for e in EVALUATORS:
        own = EVAL_TO_OWN[e]
        w(f"| {e} | {own} | {RANKS[e][own]:g} |")
    w("")
    w("- **Two missing-data policies (by design):** gemini's tier omissions are *imputed*; self-rankings are "
      "*excluded* (cᵢⱼ=0). The method's confidence term would zero both; the split is deliberate.")
    w("- **Method scope:** this implements the method's consensus (§2), closest (§5) and medoid (§6) with "
      "variance/agreement (§4). Claim-extraction (§3) and a synthesized overall-opinion essay (§4 text "
      "generation) are intentionally out of scope (spec Non-Goals).")
    w("- **Small n (10)** and one imputed opinion: treat near-ties (flagged ‹≈›) as effectively equal.")
    if sem_ok:
        w("- **Semantic vs rank** measure different things: ordinal agreement vs full-text similarity. "
          "Divergence between the two tracks is informative, not an error.")
    w(f"\n*Reproducible via `docs/planner-graph-ref/analyse/consensus_ranking.py` "
      f"(model `{EMBED_MODEL}`, equal weights, {today}).*")
    w("===REPORT END===")

    print("\n".join(out))


if __name__ == "__main__":
    main()
