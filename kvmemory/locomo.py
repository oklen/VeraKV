"""kvmemory.locomo — LOCOMO long-term-dialogue-memory loader (a second benchmark family; mirrors
ama_bench.py for agent trajectories).

LOCOMO (Maharana et al., ACL'24; the public 10-conversation / 1,986-QA release) tests memory over very
long multi-session dialogues — a different shape from AMA-Bench's agent trajectories. Each conversation
becomes a `LocomoEpisode` whose dialogue turns are `Segment`s (one per turn; `seg_id` = the LOCOMO
`dia_id` e.g. "D3:7"; `text` = "[timestamp] speaker: utterance" so temporal questions can be answered;
`meta` carries speaker/session/timestamp). QA carry the gold `evidence` dia_ids, which pin each answer
to specific turns — giving the clean selection-vs-reasoning attribution that AMA-Bench's synthesized
golds could not (FINDINGS §10.1 caveat).

Categories (int): 1 multi-hop · 2 temporal · 3 open-domain · 4 single-hop · 5 adversarial. cats 1-4 have
`{question, answer, evidence, category}`; cat 5 is *unanswerable* (has `adversarial_answer`, no `answer`)
— the abstention showcase, scored separately. The standard metric is Mem0's "J" (lenient LLM-judge
binary CORRECT/WRONG) over cats 1-4.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .core import Segment


@dataclass
class LocomoEpisode:
    conv_id: str
    sample_id: str
    speakers: tuple
    segments: list           # list[Segment], one per dialogue turn, chronological across sessions
    qa: list                 # list[dict]: {question, answer|adversarial_answer, evidence, category}


def _session_numbers(conv: dict) -> list[int]:
    return sorted(int(k.split("_")[1]) for k in conv if re.fullmatch(r"session_\d+", k))


def load_locomo(path: str) -> list[LocomoEpisode]:
    data = json.load(open(path))
    episodes = []
    for i, el in enumerate(data):
        conv = el["conversation"]
        segs: list[Segment] = []
        turn = 0
        for n in _session_numbers(conv):
            ts = conv.get(f"session_{n}_date_time", "")
            for t in conv[f"session_{n}"]:
                utt = (t.get("text") or "").strip()
                cap = t.get("blip_caption")
                if cap:
                    utt += f"  [shares an image: {cap}]"
                speaker = t.get("speaker", "")
                body = f"[{ts}] {speaker}: {utt}" if ts else f"{speaker}: {utt}"
                segs.append(Segment(
                    seg_id=t.get("dia_id", f"D{n}:{turn}"), turn=turn, text=body, kind="dialogue",
                    meta={"speaker": speaker, "session": n, "timestamp": ts, "dia_id": t.get("dia_id")},
                ))
                turn += 1
        episodes.append(LocomoEpisode(
            conv_id=str(i), sample_id=str(el.get("sample_id", i)),
            speakers=(conv.get("speaker_a"), conv.get("speaker_b")),
            segments=segs, qa=list(el.get("qa", [])),
        ))
    return episodes
