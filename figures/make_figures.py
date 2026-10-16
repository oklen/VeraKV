"""README figures. Run from anywhere: python figures/make_figures.py

Writes assets/{how-it-works,results,ttft}.png and their -dark variants. Sources:
- results: the AMA-Bench table in README.md (leaderboard and published scores as of 2026-07-15), and
  results/ama_paired/analysis.json + a3_analysis.json (the paired re-run, docs/AMA_AGENT_PAIRED.md).
- ttft: the paper's TTFT figure (kvmemory/kv_efficiency.py, dense Qwen3-8B) and its 896-question
  accuracy check (kvmemory/kv_equiv.py, results/kv_equiv_896/).
"""
import json
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _style as S  # noqa: E402

ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "assets")


def box(ax, x0, x1, yc, h, lines, t, edge, weight="normal", dashed=False, size=10):
    ax.add_patch(FancyBboxPatch((x0, yc - h / 2), x1 - x0, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=t["box"], ec=edge, lw=1.6 if edge != t["muted"] else 1.0,
                                ls=(0, (4, 3)) if dashed else "-"))
    ax.text((x0 + x1) / 2, yc, "\n".join(lines), ha="center", va="center", fontsize=size,
            color=t["fg"], fontweight=weight, linespacing=1.35)


def arrow(ax, x0, x1, y, t, dashed=False):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=t["muted"], lw=1.1, mutation_scale=11,
                                ls=(0, (4, 3)) if dashed else "-", shrinkA=0, shrinkB=0))


def how_it_works(theme):
    t = S.use(theme)
    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")
    cols = ((0.5, 11), (14, 37), (40, 61.5), (64.5, 86), (89, 99.5))
    gaps = ((11, 14), (37, 40), (61.5, 64.5), (86, 89))

    # Most agent memories: a model decides what the reader will never see.
    ya = 39.5
    ax.text(0.5, 47.5, "Most agent memories: a model decides what the reader never sees", fontsize=11,
            fontweight="bold", color=t["fg"], va="center")
    box(ax, *cols[0], ya, 7.5, ["Trajectory"], t, t["muted"])
    box(ax, *cols[1], ya, 7.5, ["A model extracts facts", "or writes summaries"], t, S.RED)
    box(ax, *cols[2], ya, 7.5, ["Retrieve top-k"], t, t["muted"])
    box(ax, *cols[3], ya, 7.5, ["A model decides it", "has read enough"], t, S.RED)
    box(ax, *cols[4], ya, 7.5, ["Reader"], t, t["muted"])
    for x0, x1 in gaps:
        arrow(ax, x0, x1, ya, t)
    ax.text(25.5, ya - 5.7, "before the question is known", ha="center", fontsize=9, color=S.RED)
    ax.text(75.25, ya - 5.7, "without seeing what is missing", ha="center", fontsize=9, color=S.RED)

    # VeraKV: no model call in the memory; only the context budget cuts.
    yb = 19.5
    ax.text(0.5, 29.5, "VeraKV: no model decides; only the context budget cuts", fontsize=11,
            fontweight="bold", color=S.BLUE, va="center")
    box(ax, *cols[0], yb, 11, ["Trajectory"], t, t["muted"])
    box(ax, *cols[1], yb, 11, ["Store every step", "verbatim, indexed by", "a short head of each"], t, S.BLUE, size=9.5)
    box(ax, *cols[2], yb, 11, ["Pick steps by number,", "word overlap (top 5)", "and recency (last 4)"], t, S.BLUE, size=9.5)
    box(ax, *cols[3], yb, 11, ["Heads of all steps", "+ picked steps in full", "~10k median, 22k cap"], t, S.BLUE, size=9.5)
    box(ax, *cols[4], yb, 11, ["Reader"], t, t["muted"])
    for x0, x1 in gaps:
        arrow(ax, x0, x1, yb, t)
    ax.text(25.5, yb - 7.4, "no model call", ha="center", fontsize=9, color=S.BLUE)
    ax.text(50.75, yb - 7.4, "no model call", ha="center", fontsize=9, color=S.BLUE)

    # Optional KV path: the picked steps go to the reader as their cached KV instead of text.
    yc = 4.6
    box(ax, 14, 86, yc, 8.4,["Optional, when you host the model: serve the picked steps",
                                "as their own cached KV, gathered at their original positions.",
                                "Same accuracy; first token up to 10.9× faster."],
        t, t["muted"], dashed=True, size=9.5)
    ax.annotate("", xy=(58.5, yc + 4.2), xytext=(58.5, yb - 5.5),
                arrowprops=dict(arrowstyle="-|>", color=t["muted"], lw=1.0, mutation_scale=10, ls=(0, (4, 3)),
                                shrinkA=0, shrinkB=0))
    ax.annotate("", xy=(94.25, yb - 5.5), xytext=(86, yc),
                arrowprops=dict(arrowstyle="-|>", color=t["muted"], lw=1.0, mutation_scale=10, ls=(0, (4, 3)),
                                shrinkA=0, shrinkB=0, connectionstyle="angle,angleA=0,angleB=90,rad=0"))
    S.save(fig, OUT, "how-it-works", theme)


LEADERBOARD = [  # README table; VeraKV's row is our run, verified on the leaderboard (#1 as of 2026-07-15)
    ("VeraKV", "0.6478", True),
    ("HybridFocus", "0.6246", False),
    ("AMA-Agent", "0.5722", False),
    ("Long context, no memory", "0.52", False),
    ("MemoRAG", "0.4606", False),
    ("HippoRAG2", "0.4480", False),
    ("MemGPT", "0.3304", False),
    ("Mem0", "0.2104", False),
]
SLICES = [("All questions", "all"), ("State-update questions", "state_update"), ("Web", "dom_WEB"),
          ("Embodied AI", "dom_EMBODIED_AI"), ("Software", "dom_SOFTWARE"), ("Text2SQL", "dom_TEXT2SQL"),
          ("Game", "dom_Game")]


def results(theme):
    t = S.use(theme)
    paired = json.load(open(os.path.join(ROOT, "results", "ama_paired", "analysis.json")))["paired"]
    a3 = json.load(open(os.path.join(ROOT, "results", "ama_paired", "a3_analysis.json")))["paired"]["V1-A3|all"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 4.1), gridspec_kw={"width_ratios": [1, 1.15], "wspace": 0.7})

    names = [n for n, _, _ in LEADERBOARD][::-1]
    vals = [float(v) for _, v, _ in LEADERBOARD][::-1]
    cols = [S.BLUE if ours else S.GREY for _, _, ours in LEADERBOARD][::-1]
    ax1.barh(range(len(vals)), vals, color=cols, height=0.62)
    for i, (_, v, ours) in enumerate(LEADERBOARD[::-1]):
        ax1.text(float(v) + 0.01, i, v, va="center", fontsize=9, color=t["fg"], fontweight="bold" if ours else "normal")
    ax1.set_yticks(range(len(names)), names)
    ax1.set_xlim(0, 0.78)
    ax1.set_xticks([0, 0.2, 0.4, 0.6])
    ax1.set_xlabel("accuracy")
    ax1.grid(axis="x", lw=0.6)
    ax1.set_axisbelow(True)
    ax1.set_title("AMA-Bench, all 2,496 questions\nscores as of 2026-07-15, when VeraKV was #1", loc="left", fontsize=10.5)

    rows = [(label, paired["V1-A1|" + key]) for label, key in SLICES]
    ys = list(range(len(rows) + 1, 1, -1))
    for (label, r), y in zip(rows, ys):
        d, lo, hi = 100 * r["diff"], 100 * r["lo"], 100 * r["hi"]
        c = S.BLUE if lo > 0 else S.GREY
        ax2.plot([lo, hi], [y, y], color=c, lw=2.2, solid_capstyle="butt")
        ax2.plot(d, y, "o", color=c, ms=6.5)
        ax2.text(hi + 0.8, y, "%+.1f" % d, va="center", fontsize=9, color=t["fg"])
    d, lo, hi = 100 * a3["diff"], 100 * a3["lo"], 100 * a3["hi"]
    ax2.plot([lo, hi], [0, 0], color=S.GREY, lw=2.2, solid_capstyle="butt")
    ax2.plot(d, 0, "o", color=S.GREY, ms=6.5)
    ax2.text(hi + 0.8, 0, ("%+.1f, but it takes 207 s per\nquestion to VeraKV's 63 s" % d).replace("-", "−"),
             va="center", fontsize=9, color=t["fg"])
    ax2.axhline(1, color=t["grid"], lw=0.8)
    ax2.axvline(0, color=t["muted"], lw=0.9)
    ax2.set_yticks(ys + [0], [label for label, _ in rows] + ["vs AMA-Agent without its\nearly answer (all questions)"])
    ax2.set_xlim(-8, 27)
    ax2.set_ylim(-0.7, len(rows) + 1.6)
    ax2.set_xlabel("VeraKV minus AMA-Agent, accuracy points (95% CI)")
    ax2.grid(axis="x", lw=0.6)
    ax2.set_axisbelow(True)
    ax2.set_title("Head-to-head with AMA-Agent, same server and judge\n2,136 paired questions", loc="left", fontsize=10.5)
    S.save(fig, OUT, "results", theme)


TTFT = {  # paper, KV sub-selection figure: per-query TTFT (ms) vs selected evidence tokens, dense Qwen3-8B
    "tokens": [575, 1916, 3008, 4387],
    "text": [66, 189, 295, 448],
    "kv": [41, 40, 41, 41],
}


def ttft(theme):
    t = S.use(theme)
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    x = TTFT["tokens"]
    ax.plot(x, TTFT["text"], "o-", color=S.GREY, lw=2, ms=6, label="re-read the selected steps as text")
    ax.plot(x, TTFT["kv"], "s-", color=S.BLUE, lw=2, ms=6, label="gather their cached KV (VeraKV)")
    for xi, a, b in zip(x, TTFT["text"], TTFT["kv"]):
        ax.text(xi, a * 1.13, "%.1f×" % (a / b), ha="center", fontsize=9, color=t["fg"])
    ax.set_yscale("log")
    ax.set_yticks([30, 50, 100, 200, 500], ["30", "50", "100", "200", "500"])
    ax.minorticks_off()
    ax.set_ylim(28, 700)
    ax.set_xlim(0, 4900)
    ax.set_xlabel("selected evidence per question (tokens)")
    ax.set_ylabel("time to first token (ms)")
    ax.grid(axis="y", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    ax.set_title("First-token latency stays flat with cached KV (Qwen3-8B)\n"
                 "accuracy is the same either way: 17.3% (KV) vs 17.4% (text), 896 questions",
                 loc="left", fontsize=10.5)
    S.save(fig, OUT, "ttft", theme)


if __name__ == "__main__":
    for theme in ("light", "dark"):
        how_it_works(theme)
        results(theme)
        ttft(theme)
