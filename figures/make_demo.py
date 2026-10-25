"""README demo: one real AMA-Bench question, AMA-Agent and VeraKV side by side, on a shared clock.

    python figures/make_demo.py [--mp4] [--lang en|zh]   # -> assets/demo-case[-zh].gif (needs ffmpeg)

Reads figures/demo_case.json (built by figures/build_demo_case.py from the logged paired re-run, and VeraKV's
build time from figures/time_index_build.py). Every quoted string and every number comes from that file; the
rest are plain labels. Both clocks (building memory, answering) show real seconds and run faster than real
time by the factor printed in the frame. AMA-Agent's memory segments fill in when their logged calls finished.
Between the two phases the build times are compared, with their ratio rounded down to one significant figure.
Both languages are drawn by default; the Chinese copy (demo-case-zh.gif, for the Chinese README) translates the
labels only, and the question, answers and step lines stay as logged."""
import json
import os
import re
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
from _style import L  # noqa: E402

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
SPEED = 12.0                             # both clocks run this many times faster than real time
T_TITLE, T_MEM, T_B0 = 0.2, 1.0, 1.5
T_B1 = T_B0 + M["ama_agent_wall_seconds"] / SPEED
T_CMP = T_B1 + 0.5                       # the two build times compared, held for 3 s
T_Q = T_CMP + 3.0
T_LISTS, T_CLOCK = T_Q + 0.5, T_Q + 1.5
T_V = T_CLOCK + V["seconds"] / SPEED
T_A = T_CLOCK + A["seconds"] / SPEED
T_REF, T_A3, T_END = T_A + 1.5, T_A + 5.0, T_A + 13.0


ZH = {  # strings from demo_case.json that appear in the labels
    "a laptop CPU (Apple M3 Pro)": "笔记本 CPU（Apple M3 Pro）",
    "an 8×A100 node": "一台 8×A100 机器",
    "yes": "对", "no": "错",
}


def tr(s):
    return L(s, ZH.get(s, s))


def dwidth(s):
    """Display width in Latin character cells: CJK characters count two."""
    return sum(2 if ord(c) >= 0x2E80 else 1 for c in s)


def wrap(s, width):
    """textwrap for English; in Chinese, break between CJK characters too and measure by display width."""
    if S.LANG != "zh":
        return textwrap.wrap(s, width)
    lines, cur = [], ""
    for tok in re.findall(r"[\u2E80-\uFFFF]|[^\s\u2E80-\uFFFF]+\s*|\s+", s):
        if cur.strip() and dwidth((cur + tok).rstrip()) > width:
            lines.append(cur.rstrip())
            cur = tok.lstrip()
        else:
            cur += tok
    if cur.strip():
        lines.append(cur.rstrip())
    return lines


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


def text(fig, x, y, s, size=13, color=FG, weight="normal", family=None, style="normal", **kw):
    return fig.text(x, y, s, fontsize=size, color=color, fontweight=weight, family=family or S.sans(SANS),
                    style=style, va="top", **kw)


def wrapped(fig, x, y, s, width, size=13, color=FG, step=0.026, **kw):
    lines = wrap(s, width)
    for i, line in enumerate(lines):
        text(fig, x, y - i * step, line, size=size, color=color, **kw)
    return y - len(lines) * step


def cell_x(x0, i):
    return x0 + i * PW / N


def box(fig, x, y, w, h, colour, alpha=1.0):
    fig.patches.append(Rectangle((x, y), w, h, transform=fig.transFigure, color=colour, alpha=alpha, lw=0))


def outline(fig, x, y, w, h, colour, lw=2.2):
    fig.patches.append(Rectangle((x, y), w, h, transform=fig.transFigure, fill=False, ec=colour, lw=lw))


def centred(fig, y, parts, size, weight="bold"):
    """One line in several colours, centred on the frame."""
    r = fig.canvas.get_renderer()
    objs = [text(fig, 0, y, s, size=size, color=c, weight=weight) for s, c in parts]
    widths = [o.get_window_extent(r).width / W for o in objs]
    x = 0.5 - sum(widths) / 2
    for o, w in zip(objs, widths):
        o.set_x(x)
        x += w


def build_comparison(fig):
    """The two build times side by side; the ratio is rounded down to one significant figure."""
    wall, ms, serving = M["ama_agent_wall_seconds"], M["verakv_median_ms"], M["ama_agent_serving"]
    ratio = int(wall / (ms / 1000.0))
    about = int(str(ratio)[0]) * 10 ** (len(str(ratio)) - 1)
    assert "laptop CPU" in M["verakv_machine"]
    box(fig, L0, 0.515, R0 + PW - L0, 0.14, BOX)
    centred(fig, 0.632, [(L("Building memory for this trajectory: ", "构建这条轨迹的 memory："), FG),
                         (L("%d s", "%d 秒") % wall, ORANGE), (" vs ", FG),
                         (L("%.1f ms", "%.1f 毫秒") % ms, GREEN),
                         (L(" — about %s× less time", "，约快 %s 倍") % format(about, ","), FG)], size=18)
    text(fig, 0.5, 0.572, L("AMA-Agent: %d LLM calls (%s, logged run on %s) · VeraKV: no model call (laptop CPU, "
                            "median of %d runs)",
                            "AMA-Agent：%d 次 LLM 调用（%s，在%s上的运行日志）· VeraKV：不调模型（笔记本 CPU，"
                            "%d 次运行的中位数）")
         % (M["ama_agent_llm_calls"], serving["model"], tr(serving["hardware"]), M["verakv_runs"]),
         color=MUTED, ha="center")


def step_line(s, obs=None):
    return "%3d  %-15s %s" % (s["step"], s["action"], obs or s["observation"])


def frame(t):
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    text(fig, L0, 0.978, L("AMA-Bench · %s · one real question · from the logged paired re-run, except VeraKV's "
                           "build time (measured)",
                           "AMA-Bench · %s · 一道真实的题 · 数据来自 paired re-run 日志，VeraKV 构建 memory 的耗时另行实测")
         % C["source"]["task_type"], size=13, color=MUTED)
    if t < T_TITLE:
        return fig
    for x0, name, sub, colour in ((L0, "AMA-Agent", L("built for this benchmark", "专为这个 benchmark 打造"), FG),
                                  (R0, "VeraKV", L("no model call in its memory", "memory 层不调用模型"), BLUE_TXT)):
        text(fig, x0, 0.945, name, size=18, weight="bold", color=colour)
        text(fig, x0, 0.907, sub, size=13, color=MUTED)

    # 1. building memory for the trajectory
    if t >= T_MEM:
        text(fig, L0, 0.871, L("1   Building memory for the %d-step trajectory", "1   为这条 %d 步的轨迹构建 memory") % N,
             size=13.5, weight="bold", color=MUTED)
        text(fig, R0 + PW, 0.871, L("clock: real seconds, %d× faster", "时钟：真实秒数，%d 倍速播放") % SPEED, size=13,
             color=MUTED, ha="right")
        text(fig, L0, 0.841, L("one state summary per ~%s characters", "每 ~%s 字符写一段 state summary")
             % format(M["ama_agent_session_chars"], ","), size=13, color=MUTED)
        text(fig, R0, 0.841, L("its index: the first lines of each step", "index：每个 step 的开头几行"), size=13,
             color=MUTED)
        ys, hs = 0.781, 0.034
        ranges, secs = M["ama_agent_turn_ranges"], M["ama_agent_call_seconds"]
        clock = min((t - T_B0) * SPEED, M["ama_agent_wall_seconds"]) if t >= T_B0 else None
        for (a, b), dt in zip(ranges, secs):        # running: faint; done when its logged call returned
            x, w = cell_x(L0, a), cell_x(L0, b + 1) - cell_x(L0, a) - 0.002
            box(fig, x, ys, w, hs, GRID if clock is None else ORANGE, alpha=1.0 if clock is None or clock >= dt else 0.3)
        box(fig, R0, ys, PW, hs, GRID if clock is None else BLUE, alpha=1.0 if clock is None else 0.75)
        if clock is not None:
            done = sum(dt <= clock for dt in secs)
            label = (L("%d LLM calls · %d s", "%d 次 LLM 调用 · %d 秒") % (len(ranges), M["ama_agent_wall_seconds"])
                     if clock >= M["ama_agent_wall_seconds"] else
                     L("%d of %d LLM calls done · %d s", "%d / %d 次 LLM 调用已完成 · %d 秒") % (done, len(ranges), clock))
            text(fig, L0, 0.765, label, size=16, weight="bold", color=ORANGE)
            text(fig, L0, 0.729, L("logged run: all %d in parallel, %d–%d s each", "运行日志：%d 次并行发出，每次 %d–%d 秒")
                 % (len(ranges), round(min(secs)), round(max(secs))), size=13, color=MUTED)
            text(fig, R0, 0.765, L("%d model calls · %.1f ms", "%d 次 LLM 调用 · %.1f ms")
                 % (M["verakv_model_calls"], M["verakv_median_ms"]), size=16, weight="bold", color=GREEN)
            text(fig, R0, 0.729, L("median of %d runs on %s", "%d 次运行的中位数，%s")
                 % (M["verakv_runs"], tr(M["verakv_machine"])), size=13, color=MUTED)
        if t >= T_REF:
            for x0, colour in ((L0, RED), (R0, GREEN)):
                for d in C["deciding_steps"]:
                    if "open" in d["action"]:
                        outline(fig, cell_x(x0, d["step"]) - 0.002, ys - 0.006, PW / N + 0.004, hs + 0.012, colour)

    if T_CMP <= t < T_Q:
        build_comparison(fig)

    # 2. answering one question, on a shared clock
    if t >= T_Q:
        text(fig, L0, 0.684, L("2   Answering one question", "2   回答一道题"), size=13.5, weight="bold",
             color=MUTED)
        wrapped(fig, L0, 0.655, C["question"], 100, size=14, step=0.03, weight="bold")
    if t >= T_CLOCK:
        sec = min((t - T_CLOCK) * SPEED, max(A["seconds"], V["seconds"]))
        text(fig, 0.5, 0.596, L("%d s", "%d 秒") % round(sec), size=24, weight="bold", ha="center")
        text(fig, 0.5, 0.557, L("real seconds · the clock runs %d× faster", "真实秒数 · 时钟 %d 倍速播放") % SPEED,
             size=13, color=MUTED, ha="center")
    if t >= T_LISTS:
        text(fig, L0, 0.522, L("It reads steps %s:", "它读的是 step %s：")
             % L(", ", "、").join(str(s["step"]) for s in A["steps"]), size=13, color=MUTED)
        y = 0.495
        for s in A["steps"]:
            if s["step"] in (6, 16):
                obs = "… The drawer 1 is closed." if "is closed" in s["observation"] else s["observation"]
                text(fig, L0, y, step_line(s, obs), size=13, family=MONO)
                y -= 0.026
        text(fig, R0, 0.522, L("It reads the first lines of every step, among them:", "它读的是每个 step 的开头几行，其中有："),
             size=13, color=MUTED)
        y = 0.495
        for s in C["deciding_steps"]:
            head = V["head_lines"][str(s["step"])]
            assert s["action"] in head and s["observation"].split(". ")[0] in head
            obs = "… The drawer 1 is open." if "is open" in s["observation"] else s["observation"]
            text(fig, R0, y, step_line(s, obs), size=13, family=MONO, color=GREEN if "open" in s["action"] else FG)
            y -= 0.026

    for x0, side, t_ans, colour in ((L0, A, T_A, ORANGE), (R0, V, T_V, BLUE_TXT)):
        if t < T_CLOCK:
            continue
        if t < t_ans:
            text(fig, x0, 0.407, L("working …", "处理中 …"), size=13, color=MUTED, style="italic")
            continue
        text(fig, x0, 0.407, L("answers at %d s · %d model call%s" % (round(side["seconds"]), side["model_calls"],
                                                                       "" if side["model_calls"] == 1 else "s"),
                               "%d 秒作答 · %d 次 LLM 调用" % (round(side["seconds"]), side["model_calls"])),
             size=14, weight="bold", color=colour)
        y = wrapped(fig, x0, 0.374, "“%s”" % first_sentences(side["answer"], 2), 62, size=13, step=0.025)
        if side is A:
            wrapped(fig, x0, y - 0.006, L("its check said %s; its reasoning: “%s”",
                                          "它的「够不够」检查判 %s；它自己的 reasoning 里写着：“%s”")
                    % (A["check_output"], A["reasoning_quote"]), 66, size=13, color=MUTED, step=0.025, style="italic")
        if t >= T_REF:
            mark = "✓" if side["score"] == 1.0 else "✗"
            text(fig, x0 + PW, 0.407, L("%s judge: %s", "%s judge：%s") % (mark, tr(side["judge"])), size=15,
                 weight="bold", color=GREEN if side["score"] == 1.0 else RED, ha="right")

    if t >= T_REF:
        text(fig, L0, 0.245, L("steps 5 and 17 (open drawer 1): not among the steps it read",
                               "step 5、17（open drawer 1）：不在它读的 step 里"), size=13, color=RED)
        text(fig, R0, 0.245, L("steps 5 and 17 (open drawer 1): in its index",
                               "step 5、17（open drawer 1）：在它的 index 里"), size=13, color=GREEN)
        box(fig, L0, 0.115, R0 + PW - L0, 0.09, BOX)
        wrapped(fig, L0 + 0.01, 0.193, L("Reference answer: ", "参考答案：") + C["reference"], 118, size=13,
                step=0.027)
    if t >= T_A3:
        text(fig, L0, 0.085, L("Forced past its early answer, AMA-Agent takes %d model calls and %d s on this question,",
                               "去掉 early answer 后，AMA-Agent 在这道题上要 %d 次 LLM 调用、%d 秒，")
             % (A3["model_calls"], round(A3["seconds"])), size=13, color=FG)
        text(fig, L0, 0.052, L("and still answers: “%s”", "答案仍然是：“%s”") % first_sentences(A3["answer"], 1),
             size=13, color=FG)
    return fig


def main():
    out = os.path.join(ROOT, "assets", "demo-case%s.gif" % L("", "-zh"))
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
            mp4 = os.path.join(os.path.dirname(tmp), "demo-case%s.mp4" % L("", "-zh"))
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", frames,
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", mp4], check=True)
            print("wrote", mp4)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--frame":       # --frame <seconds> <out.png> [--lang zh]
        S.set_lang(S.langs()[0] if "--lang" in sys.argv else "en")
        frame(float(sys.argv[2])).savefig(sys.argv[3], facecolor=BG)
    else:
        for lang in S.langs():
            S.set_lang(lang)
            main()
