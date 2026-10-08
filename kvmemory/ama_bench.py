"""kvmemory.ama_bench — load AMA-Bench episodes and slice into Segments.

AMA-Bench (ICLR 2026, github.com/AMA-Bench/AMA-Bench): each record =
  {episode_id, domain, task_type, num_turns, total_tokens, success,
   trajectory:[{turn_idx, action, observation}],
   qa_pairs:[{question, answer, type, question_uuid}]}.
One turn -> one Segment (skip empty leading/padding turns). ColdIndex defaults to framework
auto-gist (these trajectories carry no separate scratchpad; reasoning is sometimes embedded in
`action`). QA type 'A' = exact recall from an old step = the verbatim-rehydration sweet spot.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .core import Segment


@dataclass
class Episode:
    episode_id: int
    domain: str
    task_type: str
    task: str
    num_turns: int
    total_tokens: int
    success: bool
    segments: list = field(default_factory=list)
    qa: list = field(default_factory=list)  # [{question, answer, type, question_uuid}]


def _seg_text(action: str, observation: str) -> str:
    parts = []
    if action:
        parts.append(f"action: {action}")
    if observation:
        parts.append(f"observation: {observation}")
    return "\n".join(parts)


def load_episodes(path: str, max_tokens: int | None = None, limit: int | None = None,
                  domains: set | None = None) -> list[Episode]:
    """Load AMA-Bench jsonl; filter by context length / domain; slice turns into Segments."""
    eps: list[Episode] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if max_tokens is not None and r["total_tokens"] > max_tokens:
                continue
            if domains is not None and r["domain"] not in domains:
                continue
            segs = []
            for t in r["trajectory"]:
                act = (t.get("action") or "").strip()
                obs = (t.get("observation") or "").strip()
                if not act and not obs:
                    continue  # skip empty padding turns
                segs.append(Segment(
                    seg_id=f"{r['episode_id']}:{t['turn_idx']}",
                    turn=t["turn_idx"],
                    text=_seg_text(act, obs),
                    kind="step",
                    meta={"action": act, "observation": obs},
                ))
            if not segs:
                continue
            eps.append(Episode(
                episode_id=r["episode_id"], domain=r["domain"], task_type=r["task_type"],
                task=r.get("task", ""), num_turns=r["num_turns"],
                total_tokens=r["total_tokens"], success=bool(r.get("success", False)),
                segments=segs, qa=r["qa_pairs"],
            ))
            if limit is not None and len(eps) >= limit:
                break
    return eps
