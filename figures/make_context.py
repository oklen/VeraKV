"""README figure: what each reader's prompt contains for the demo question, section by section.

    python figures/make_context.py       # -> assets/context-case.png and assets/context-case-dark.png

Reads the verbatim excerpts in figures/demo_case.json (built by figures/build_demo_case.py from the logged
prompts of the paired re-run). Lines are cut with "…" where the prompt goes on; nothing is reworded."""
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


def wrap(line):
    body = line.lstrip(" ")
    indent = " " * (len(line) - len(body))
    return textwrap.wrap(body, WRAP, initial_indent=indent, subsequent_indent=indent + "    ",
                         break_on_hyphens=False) or [""]


def column(side):
    """Layout ops for one column: ('label', text, colour_key) and ('box', [(line, colour_key)])."""
    ops = []
    for sec in C["contexts"][side]:
        title = sec["title"] + (" · " + sec["size"] if sec.get("size") else "")
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
            ops.append(("note", "No step 5 or 17 (open drawer 1) among these turns.", "red"))
        if sec["title"].startswith("Overview"):
            ops.append(("note", "Steps 5 and 17 (open drawer 1) are here, one line each.", "green"))
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

    def txt(x_px, y_px, s, size, colour, family=SANS, weight="normal"):
        fig.text(x_px / W, 1 - y_px / H, s, fontsize=size, color=colour, family=family, weight=weight, va="top")

    txt(38, 22, "What each reader sees for this question", 16, t["fg"], weight="bold")
    txt(38, 56, "Excerpts from the logged prompts of the paired re-run, section by section, in order. "
                "“…” marks text left out.", 12.5, t["muted"])
    for (side, ops), x0, name, colour in zip(cols.items(), (38, 662), ("What AMA-Agent's reader sees",
                                                                    "What VeraKV's reader sees"),
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
    path = os.path.join(ROOT, "assets", "context-case" + ("" if theme == "light" else "-dark") + ".png")
    Image.open(path).convert("RGB").quantize(colors=64, method=Image.Quantize.MEDIANCUT,
                                             dither=Image.Dither.NONE).save(path, optimize=True)


if __name__ == "__main__":
    for theme in ("dark", "light"):
        draw(theme)
