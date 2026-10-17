"""README demo: one real AMA-Bench question, played through AMA-Agent and VeraKV side by side.

    python figures/make_demo.py            # -> assets/demo-case.gif (needs ffmpeg)

Reads figures/demo_case.json (built by figures/build_demo_case.py from the logged prompts and answers of
the paired re-run). Every quoted string comes from that file; the rest are plain labels."""
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
A, V = C["ama_agent"], C["verakv"]
N = C["num_steps"]

W, H, DPI, FPS = 1280, 900, 100, 10
T = S.THEMES["dark"]
BG, FG, MUTED, GRID, BOX = T["bg"], T["fg"], T["muted"], T["grid"], T["box"]
BLUE_TXT, ORANGE, RED, GREEN, BLUE = "#79b8ff", S.ORANGE, "#ff7b72", "#56d364", S.BLUE
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"
L0, R0, PW = 0.03, 0.52, 0.45          # panel left edges and width (figure fraction)

# timeline (seconds)
T_Q, T_STRIP, T_SEL, T_SEL_END, T_READER, T_CHECK, T_ANS, T_REF, T_END = 0.3, 3.0, 5.0, 11.0, 12.5, 17.5, 24.0, 30.0, 42.0


def plain(s):
    return s.replace("**", "")


def first_sentences(s, n):
    """The first n sentences of an answer, verbatim apart from markdown emphasis."""
    s = plain(s).replace("\n", " ")
    out, rest = [], s
    for _ in range(n):
        i = rest.find(". ")
        if i < 0:
            out.append(rest.strip())
            break
        out.append(rest[:i + 1].strip())
        rest = rest[i + 2:]
    return " ".join(out)


def lerp(t, t0, t1):
    return 0.0 if t <= t0 else 1.0 if t >= t1 else (t - t0) / (t1 - t0)


def text(fig, x, y, s, size=12.5, color=FG, weight="normal", family=SANS, style="normal", **kw):
    return fig.text(x, y, s, fontsize=size, color=color, fontweight=weight, family=family, style=style,
                    va="top", **kw)


def wrapped(fig, x, y, s, width, size=12.5, color=FG, step=0.026, **kw):
    lines = textwrap.wrap(s, width)
    for i, line in enumerate(lines):
        text(fig, x, y - i * step, line, size=size, color=color, **kw)
    return y - len(lines) * step


def strip(fig, x0, y0, lit, dim=(), outline=(), marks=(), mark_color=RED):
    """The trajectory: one cell per step; `lit` bright, `dim` faint, `outline` boxed, `marks` boxed in mark_color."""
    w, h = PW / N, 0.034
    for i in range(N):
        colour, alpha = GRID, 1.0
        if i in dim:
            colour, alpha = BLUE, 0.62
        if i in lit:
            colour, alpha = lit[i], 1.0
        fig.patches.append(Rectangle((x0 + i * w, y0), w * 0.82, h, transform=fig.transFigure, color=colour,
                                     alpha=alpha, lw=0))
    for i in outline:
        fig.patches.append(Rectangle((x0 + i * w - 0.0015, y0 - 0.004), w * 0.82 + 0.003, h + 0.008,
                                     transform=fig.transFigure, fill=False, ec=FG, lw=1.4))
    for i in marks:
        fig.patches.append(Rectangle((x0 + i * w - 0.002, y0 - 0.006), w * 0.82 + 0.004, h + 0.012,
                                     transform=fig.transFigure, fill=False, ec=mark_color, lw=2.2))


def frame(t):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    text(fig, L0, 0.975, "AMA-Bench · %s · one real question from the paired re-run (logged prompts and answers)"
         % C["source"]["task_type"], size=12.5, color=MUTED)
    if t >= T_Q:
        wrapped(fig, L0, 0.94, C["question"], 96, size=15, step=0.032, weight="bold")

    for x0, name, sub, colour, found in ((L0, "AMA-Agent (as released)", "built for this benchmark", FG,
                                          ("steps 5 and 17 (open drawer 1): never retrieved", RED)),
                                         (R0, "VeraKV (released config)", "no model call in the memory", BLUE_TXT,
                                          ("steps 5 and 17 (open drawer 1): in the overview", GREEN))):
        text(fig, x0, 0.855, name, size=16, weight="bold", color=colour)
        if t >= T_REF:
            text(fig, x0, 0.822, found[0], size=12.5, color=found[1])
        else:
            text(fig, x0, 0.822, sub, size=12.5, color=MUTED)

    # 2. the trajectory
    if t >= T_STRIP:
        sel = lerp(t, T_SEL, T_SEL_END)
        a_steps = A["retrieved"] + A["named"]
        a_lit = {n: ORANGE for k, n in enumerate(a_steps) if sel * len(a_steps) > k}
        v_full = V["full"] + V["hot"]
        v_dim = set(range(N)) if sel > 0.15 else set()
        v_lit = {n: BLUE_TXT for k, n in enumerate(v_full) if sel * len(v_full) > k}
        reveal = t >= T_REF
        deciding = [d["step"] for d in C["deciding_steps"] if "open" in d["action"]]
        strip(fig, L0, 0.745, a_lit, outline=A["named"] if sel > 0.9 else (),
              marks=deciding if reveal else (), mark_color=RED)
        strip(fig, R0, 0.745, v_lit, dim=v_dim, outline=A["named"] if sel > 0.9 else (),
              marks=deciding if reveal else (), mark_color=GREEN)
        for x0 in (L0, R0):
            text(fig, x0, 0.733, "step 0", size=12, color=MUTED)
            text(fig, x0 + PW, 0.733, "step %d" % (N - 1), size=12, color=MUTED, ha="right")
            text(fig, x0 + PW / 2, 0.733, "%d steps · %s tokens" % (N, format(C["trajectory_tokens"], ",")),
                 size=12, color=MUTED, ha="center")
        if sel > 0:
            text(fig, L0, 0.70, "similarity search: top %d + the step the question names" % len(A["retrieved"]),
                 size=12.5, color=ORANGE)
            text(fig, R0, 0.70, "a short head of every step + %d in full + the last %d"
                 % (len(V["full"]), len(V["hot"])), size=12.5, color=BLUE_TXT)
            for x0, n_tok, colour in ((L0, A["prompt_tokens"], ORANGE), (R0, V["prompt_tokens"], BLUE_TXT)):
                frac = n_tok / float(V["prompt_tokens"]) * sel
                fig.patches.append(Rectangle((x0, 0.655), PW * 0.55, 0.012, transform=fig.transFigure,
                                             color=GRID, lw=0))
                fig.patches.append(Rectangle((x0, 0.655), PW * 0.55 * frac, 0.012, transform=fig.transFigure,
                                             color=colour, lw=0))
                if sel >= 1:
                    text(fig, x0 + PW * 0.57, 0.669, "%s tokens to its reader" % format(n_tok, ","), size=12.5,
                         color=colour)

    # 4. what each reader has about drawer 1
    if t >= T_READER:
        text(fig, L0, 0.625, "What its reader gets (%d steps):" % len(A["steps"]), size=12.5, color=MUTED)
        y = 0.597
        for s in A["steps"]:
            obs = s["observation"]
            if s["step"] in A["retrieved"] and "drawer 1 is closed" in obs:
                obs = "… The drawer 1 is closed."
            elif len(obs) > 29:
                obs = obs.split(". ")[0] + ". …"
            text(fig, L0, y, "%3d  %-18s %s" % (s["step"], s["action"], obs), size=12.5, family=MONO)
            y -= 0.026
        text(fig, R0, 0.625, "In its overview of all %d steps:" % N, size=12.5, color=MUTED)
        y = 0.597
        for s in C["deciding_steps"]:
            head = V["head_lines"][str(s["step"])]
            assert s["action"] in head and s["observation"].split(". ")[0] in head
            obs = "… The drawer 1 is open." if "is open" in s["observation"] else s["observation"]
            text(fig, R0, y, "%3d  %-18s %s" % (s["step"], s["action"], obs), size=12.5, family=MONO,
                 color=GREEN if "open" in s["action"] else FG)
            y -= 0.026
        text(fig, R0, y - 0.004, "+ the other %d steps, and %d steps in full" % (N - len(C["deciding_steps"]),
                                                                                 len(V["full"]) + len(V["hot"])),
             size=12.5, color=MUTED)

    # 5. the check
    if t >= T_CHECK:
        text(fig, L0, 0.425, "Its check, before answering:", size=12.5, color=MUTED)
        y = wrapped(fig, L0, 0.398, "“%s”" % A["check_question"], 62, size=12.5, step=0.025, style="italic")
        if t >= T_CHECK + 1.5:
            text(fig, L0, y - 0.004, "→ %s" % A["check_output"], size=15, weight="bold", color=ORANGE)
        if t >= T_CHECK + 3.0:
            y2 = y - 0.035
            for q in A["reasoning_quotes"]:
                y2 = wrapped(fig, L0, y2, "“%s”" % q, 62, size=12.5, color=MUTED, step=0.025, style="italic")
        text(fig, R0, 0.425, "Before answering:", size=12.5, color=MUTED)
        text(fig, R0, 0.398, "No model decides what to leave out.", size=15, weight="bold", color=BLUE_TXT)
        text(fig, R0, 0.365, "Only the context budget cuts.", size=12.5, color=FG)

    # 6. the answers
    if t >= T_ANS:
        for x0, side, colour in ((L0, A, RED), (R0, V, GREEN)):
            text(fig, x0, 0.222, "Its answer:", size=12.5, color=MUTED)
            wrapped(fig, x0, 0.195, "“%s”" % first_sentences(side["answer"], 2), 62, size=12.5, step=0.025)
            if t >= T_REF:
                mark = "✓" if side["score"] == 1.0 else "✗"
                text(fig, x0 + PW, 0.222, "%s  judge: %s" % (mark, side["judge"]), size=15, weight="bold",
                     color=colour, ha="right")

    # 7. the reference
    if t >= T_REF:
        fig.patches.append(Rectangle((L0, 0.025), R0 + PW - L0, 0.072, transform=fig.transFigure, color=BOX, lw=0))
        wrapped(fig, L0 + 0.01, 0.088, "Reference answer: " + C["reference"], 120, size=12.5, step=0.026)
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
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", os.path.join(tmp, "f%04d.png"),
                        "-filter_complex", "split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=none",
                        "-loop", "0", out], check=True)
        print("wrote", out, os.path.getsize(out), "bytes,", n / float(FPS), "s")
        if "--mp4" in sys.argv:
            mp4 = os.path.join(tmp, "..", "demo-case.mp4")
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", os.path.join(tmp, "f%04d.png"),
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", mp4], check=True)
            print("wrote", os.path.abspath(mp4))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--frame":
        f = frame(float(sys.argv[2]))
        f.savefig(sys.argv[3], facecolor=BG)
    else:
        main()
