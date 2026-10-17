"""README demo: one real AMA-Bench question, AMA-Agent and VeraKV side by side, on a shared clock.

    python figures/make_demo.py [--mp4]          # -> assets/demo-case.gif (needs ffmpeg)

Reads figures/demo_case.json (built by figures/build_demo_case.py from the logged paired re-run). Every quoted
string and every number comes from that file; the rest are plain labels. The answering clock shows real
seconds and runs faster than real time by the factor printed under it."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import _style as S  # noqa: E402

ROOT = os.path.dirname(HERE)
C = json.load(open(os.path.join(HERE, "demo_case.json"), encoding="utf-8"))
A, V, A3, M = C["ama_agent"], C["verakv"], C["ama_agent_no_early_answer"], C["memory_build"]
N = C["num_steps"]

W, H, DPI, FPS = 1280, 900, 100, 10
T = S.THEMES["dark"]
BG, FG, MUTED, GRID, BOX = T["bg"], T["fg"], T["muted"], T["grid"], T["box"]
BLUE_TXT, ORANGE, RED, GREEN, BLUE = "#79b8ff", S.ORANGE, "#ff7b72", "#56d364", S.BLUE
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"
L0, R0, PW = 0.03, 0.52, 0.45          # panel left edges and width (figure fraction)

# timeline (seconds of animation)
T_TITLE, T_MEM, T_MEM_END, T_Q, T_LISTS, T_CLOCK = 0.2, 1.0, 6.5, 7.5, 8.0, 9.0
SPEED = 9.0                              # the answering clock runs this many times faster than real time
T_V = T_CLOCK + V["seconds"] / SPEED
T_A = T_CLOCK + A["seconds"] / SPEED
T_REF, T_A3, T_END = T_A + 1.5, T_A + 5.0, T_A + 13.0


def plain(s):
    return s.replace("**", "")


def first_sentences(s, n):
    """The first n sentences of an answer, verbatim apart from markdown emphasis."""
    rest, out = plain(s).replace("\n", " "), []
    for _ in range(n):
        i = rest.find(". ")
        if i < 0:
            out.append(rest.strip())
            break
        out.append(rest[:i + 1].strip())
        rest = rest[i + 2:]
    return " ".join(out)


def text(fig, x, y, s, size=12.5, color=FG, weight="normal", family=SANS, style="normal", **kw):
    return fig.text(x, y, s, fontsize=size, color=color, fontweight=weight, family=family, style=style,
                    va="top", **kw)


def wrapped(fig, x, y, s, width, size=12.5, color=FG, step=0.026, **kw):
    lines = textwrap.wrap(s, width)
    for i, line in enumerate(lines):
        text(fig, x, y - i * step, line, size=size, color=color, **kw)
    return y - len(lines) * step


def cell_x(x0, i):
    return x0 + i * PW / N


def box(fig, x, y, w, h, colour, alpha=1.0):
    fig.patches.append(Rectangle((x, y), w, h, transform=fig.transFigure, color=colour, alpha=alpha, lw=0))


def outline(fig, x, y, w, h, colour, lw=2.2):
    fig.patches.append(Rectangle((x, y), w, h, transform=fig.transFigure, fill=False, ec=colour, lw=lw))


def step_line(s, obs=None):
    return "%3d  %-15s %s" % (s["step"], s["action"], obs or s["observation"])


def frame(t):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    text(fig, L0, 0.978, "AMA-Bench · %s · one real question · model calls, times and answers from the logged "
         "paired re-run" % C["source"]["task_type"], size=12.5, color=MUTED)
    if t < T_TITLE:
        return fig
    for x0, name, sub, colour in ((L0, "AMA-Agent", "built for this benchmark", FG),
                                  (R0, "VeraKV", "no model call in its memory", BLUE_TXT)):
        text(fig, x0, 0.94, name, size=18, weight="bold", color=colour)
        text(fig, x0, 0.902, sub, size=12.5, color=MUTED)

    # 1. building memory for the trajectory
    if t >= T_MEM:
        text(fig, L0, 0.862, "1   Building memory for the %d-step trajectory" % N, size=13.5, weight="bold",
             color=MUTED)
        ys, hs = 0.778, 0.034
        ranges = M["ama_agent_turn_ranges"]
        done = min(len(ranges), int((t - T_MEM - 0.3) / ((T_MEM_END - T_MEM - 0.3) / len(ranges))) + 1) \
            if t >= T_MEM + 0.3 else 0
        for k, (a, b) in enumerate(ranges):
            x, w = cell_x(L0, a), cell_x(L0, b + 1) - cell_x(L0, a) - 0.002
            box(fig, x, ys, w, hs, ORANGE if k < done else GRID)
        ready = t >= T_MEM + 0.3
        box(fig, R0, ys, PW, hs, BLUE if ready else GRID, alpha=0.75 if ready else 1.0)
        text(fig, L0, 0.762, "%d LLM call%s" % (done, "" if done == 1 else "s"), size=16, weight="bold",
             color=ORANGE)
        text(fig, L0, 0.726, "one state summary per ~%s characters" % format(M["ama_agent_session_chars"], ","),
             size=12.5, color=MUTED)
        if ready:
            text(fig, R0, 0.762, "%d model calls · ready" % M["verakv_model_calls"], size=16, weight="bold",
                 color=GREEN)
            text(fig, R0, 0.726, "its index: the first lines of each step", size=12.5, color=MUTED)
        if t >= T_REF:
            for x0, colour in ((L0, RED), (R0, GREEN)):
                for d in C["deciding_steps"]:
                    if "open" in d["action"]:
                        outline(fig, cell_x(x0, d["step"]) - 0.002, ys - 0.006, PW / N + 0.004, hs + 0.012, colour)

    # 2. answering one question, on a shared clock
    if t >= T_Q:
        text(fig, L0, 0.688, "2   Answering one question", size=13.5, weight="bold", color=MUTED)
        wrapped(fig, L0, 0.658, C["question"], 100, size=14, step=0.03, weight="bold")
    if t >= T_CLOCK:
        sec = min((t - T_CLOCK) * SPEED, max(A["seconds"], V["seconds"]))
        text(fig, 0.5, 0.596, "%d s" % round(sec), size=24, weight="bold", ha="center")
        text(fig, 0.5, 0.557, "real seconds · the clock runs %d× faster" % SPEED, size=12.5, color=MUTED, ha="center")
    if t >= T_LISTS:
        text(fig, L0, 0.522, "It reads steps %s:" % ", ".join(str(s["step"]) for s in A["steps"]), size=12.5,
             color=MUTED)
        y = 0.495
        for s in A["steps"]:
            if s["step"] in (6, 16):
                obs = "… The drawer 1 is closed." if "is closed" in s["observation"] else s["observation"]
                text(fig, L0, y, step_line(s, obs), size=12.5, family=MONO)
                y -= 0.026
        text(fig, R0, 0.522, "It reads the first lines of every step, among them:", size=12.5, color=MUTED)
        y = 0.495
        for s in C["deciding_steps"]:
            head = V["head_lines"][str(s["step"])]
            assert s["action"] in head and s["observation"].split(". ")[0] in head
            obs = "… The drawer 1 is open." if "is open" in s["observation"] else s["observation"]
            text(fig, R0, y, step_line(s, obs), size=12.5, family=MONO, color=GREEN if "open" in s["action"] else FG)
            y -= 0.026

    for x0, side, t_ans, colour in ((L0, A, T_A, ORANGE), (R0, V, T_V, BLUE_TXT)):
        if t < T_CLOCK:
            continue
        if t < t_ans:
            text(fig, x0, 0.415, "working …", size=13, color=MUTED, style="italic")
            continue
        text(fig, x0, 0.415, "answers at %d s · %d model call%s" % (round(side["seconds"]), side["model_calls"],
                                                                     "" if side["model_calls"] == 1 else "s"),
             size=14, weight="bold", color=colour)
        y = wrapped(fig, x0, 0.382, "“%s”" % first_sentences(side["answer"], 2), 62, size=12.5, step=0.025)
        if side is A:
            wrapped(fig, x0, y - 0.006, "its check said %s; its reasoning: “%s”" % (A["check_output"], A["reasoning_quote"]),
                    66, size=12.5, color=MUTED, step=0.025, style="italic")
        if t >= T_REF:
            mark = "✓" if side["score"] == 1.0 else "✗"
            text(fig, x0 + PW, 0.415, "%s judge: %s" % (mark, side["judge"]), size=15, weight="bold",
                 color=GREEN if side["score"] == 1.0 else RED, ha="right")

    if t >= T_REF:
        text(fig, L0, 0.245, "steps 5 and 17 (open drawer 1): not among the steps it read", size=12.5, color=RED)
        text(fig, R0, 0.245, "steps 5 and 17 (open drawer 1): in its index", size=12.5, color=GREEN)
        box(fig, L0, 0.115, R0 + PW - L0, 0.09, BOX)
        wrapped(fig, L0 + 0.01, 0.193, "Reference answer: " + C["reference"], 118, size=12.5, step=0.027)
    if t >= T_A3:
        text(fig, L0, 0.085, "Forced past its early answer, AMA-Agent takes %d model calls and %d s on this question,"
             % (A3["model_calls"], round(A3["seconds"])), size=13, color=FG)
        text(fig, L0, 0.052, "and still answers: “%s”" % first_sentences(A3["answer"], 1), size=13, color=FG)
    return fig


def main():
    out = os.path.join(ROOT, "assets", "demo-case.gif")
    tmp = tempfile.mkdtemp(prefix="vkdemo_")
    try:
        n = int(T_END * FPS)
        for i in range(n):
            fig = frame(i / float(FPS))
            fig.savefig(os.path.join(tmp, "f%04d.png" % i), facecolor=BG)
            plt.close(fig)
        frames = os.path.join(tmp, "f%04d.png")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", frames,
                        "-filter_complex", "split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none",
                        "-loop", "0", out], check=True)
        print("wrote", out, os.path.getsize(out), "bytes,", n / float(FPS), "s")
        if "--mp4" in sys.argv:
            mp4 = os.path.join(os.path.dirname(tmp), "demo-case.mp4")
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", frames,
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", mp4], check=True)
            print("wrote", mp4)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--frame":
        frame(float(sys.argv[2])).savefig(sys.argv[3], facecolor=BG)
    else:
        main()
