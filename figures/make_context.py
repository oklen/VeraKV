"""README figure: what each reader's prompt contains for the demo question, section by section.

    python figures/make_context.py [--lang en|zh]   # -> assets/context-case[-zh][-dark].png (both languages by default)

Reads the verbatim excerpts in figures/demo_case.json (built by figures/build_demo_case.py from the logged
prompts of the paired re-run). Lines are cut with "…" where the prompt goes on; nothing is reworded. In the
Chinese copy only the labels around the excerpts are translated; the excerpts stay as logged."""
import json
import os
import sys
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _style as S  # noqa: E402
from _style import L  # noqa: E402

ROOT = os.path.dirname(HERE)
C = json.load(open(os.path.join(HERE, "demo_case.json"), encoding="utf-8"))
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"
W, DPI = 1280, 100
PT = DPI / 72.0
MONO_PT, LABEL_PT = 12.5, 12.5
LINE = MONO_PT * PT * 1.42          # px per mono line
WRAP = 52                           # mono characters per line in a column
ACCENT = {"dark": {"green": "#56d364", "red": "#ff7b72", "blue": "#79b8ff"},
          "light": {"green": "#1a7f37", "red": "#cf222e", "blue": "#0969da"}}
DECIDING = ("<step 5>", "<step 17>")
ZH = {  # section titles and sizes of demo_case.json, for the Chinese copy
    "Role": "Role", "Query": "Query", "Retrieved turns": "Retrieved turns", "Note": "Note",
    "Decision procedure": "Decision procedure", "Output formats": "Output formats",
    "Overview: one line per step": "Overview：每个 step 一行", "The last 4 steps in full": "最后 4 个 step 全文",
    "Selected steps in full": "选中 step 全文", "Question": "Question", "Instruction": "Instruction",
    "each turn also lists its available actions, left out here": "每个 turn 还列了 available actions，这里略去",
    "steps 0–145": "step 0–145", "steps 146–149": "step 146–149",
    "steps 21, 67, 77, 136, 141": "step 21、67、77、136、141",
}


def tr(s):
    return L(s, ZH[s])


def wrap(line):
    body = line.lstrip(" ")
    indent = " " * (len(line) - len(body))
    return textwrap.wrap(body, WRAP, initial_indent=indent, subsequent_indent=indent + "    ",
                         break_on_hyphens=False) or [""]


def column(side):
    """Layout ops for one column: ('label', text, colour_key) and ('box', [(line, colour_key)])."""
    ops = []
    for sec in C["contexts"][side]:
        title = tr(sec["title"]) + (" · " + tr(sec["size"]) if sec.get("size") else "")
        ops.append(("label", title, "muted"))
        rows = []
        if sec.get("header"):
            rows.append((sec["header"], "fg"))
        for t in sec.get("turns", []):
            rows += [(t[0], "fg"), (t[1], "fg"), (t[2], "fg")]
        gaps = set(sec.get("gaps_after", []))
        for line in sec.get("lines", []):
            rows.append((line, "green" if line.startswith(DECIDING) else "muted" if line == "…" else "fg"))
            n = line.split(">")[0].replace("<step ", "") if line.startswith("<step ") else None
            if n is not None and n.isdigit() and int(n) in gaps:
                rows.append(("…", "muted"))
        ops.append(("box", rows))
        if sec["title"] == "Retrieved turns":
            ops.append(("note", L("No step 5 or 17 (open drawer 1) among these turns.",
                                  "这些 turn 里没有 step 5 和 17（open drawer 1）。"), "red"))
        if sec["title"].startswith("Overview"):
            ops.append(("note", L("Steps 5 and 17 (open drawer 1) are here, one line each.",
                                  "step 5 和 17（open drawer 1）都在这里，各占一行。"), "green"))
    return ops


def height(ops):
    h = 0.0
    for op in ops:
        if op[0] == "box":
            h += 16 + LINE * sum(len(wrap(l)) for l, _ in op[1]) + 14
        else:
            h += LABEL_PT * PT * 1.6 + (12 if op[0] == "note" else 0)
    return h


def draw(theme):
    t = S.use(theme)
    acc = ACCENT[theme]
    colours = {"fg": t["fg"], "muted": t["muted"], **acc}
    cols = {"ama_agent": column("ama_agent"), "verakv": column("verakv")}
    top = 150.0
    H = top + max(height(o) for o in cols.values()) + 30
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=t["bg"])

    def txt(x_px, y_px, s, size, colour, family=None, weight="normal"):
        fig.text(x_px / W, 1 - y_px / H, s, fontsize=size, color=colour, family=family or S.sans(SANS),
                 weight=weight, va="top")

    txt(38, 22, L("What each reader sees for this question", "这道题，两边的 reader 各看到了什么"), 16, t["fg"],
        weight="bold")
    txt(38, 56, L("Excerpts from the logged prompts of the paired re-run, section by section, in order. "
                  "“…” marks text left out.",
                  "摘自 paired re-run 日志里的真实 prompt，按顺序分段摘录，原文未改；“…” 表示省略。"), 12.5, t["muted"])
    for (side, ops), x0, name, colour in zip(cols.items(), (38, 662), (L("What AMA-Agent's reader sees",
                                                                      "AMA-Agent 的 reader 看到的"),
                                                                    L("What VeraKV's reader sees",
                                                                      "VeraKV 的 reader 看到的")),
                                             (t["fg"], acc["blue"])):
        txt(x0, 100, name, 15, colour, weight="bold")
        y = top
        for op in ops:
            if op[0] == "label":
                txt(x0, y, op[1], LABEL_PT, colours[op[2]])
                y += LABEL_PT * PT * 1.6
            elif op[0] == "note":
                txt(x0, y - 4, op[1], LABEL_PT, colours[op[2]], weight="bold")
                y += LABEL_PT * PT * 1.6 + 12
            else:
                n_lines = sum(len(wrap(l)) for l, _ in op[1])
                h_box = 16 + LINE * n_lines
                fig.patches.append(Rectangle((x0 / W, 1 - (y + h_box) / H), 580 / W, h_box / H,
                                             transform=fig.transFigure, color=t["box"], lw=0))
                yy = y + 8
                for line, ck in op[1]:
                    for part in wrap(line):
                        txt(x0 + 10, yy, part, MONO_PT, colours[ck], family=MONO)
                        yy += LINE
                y += h_box + 14
    S.save(fig, os.path.join(ROOT, "assets"), "context-case", theme)
    # Text on flat colours: a 64-colour palette looks the same at under a third of the size.
    path = os.path.join(ROOT, "assets", S.name_for("context-case", theme) + ".png")
    Image.open(path).convert("RGB").quantize(colors=64, method=Image.Quantize.MEDIANCUT,
                                             dither=Image.Dither.NONE).save(path, optimize=True)


if __name__ == "__main__":
    for lang in S.langs():
        S.set_lang(lang)
        for theme in ("dark", "light"):
            draw(theme)
