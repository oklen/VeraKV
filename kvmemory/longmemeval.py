"""kvmemory.longmemeval — LongMemEval loader (a THIRD benchmark family: long-context chat-assistant
memory; Wu et al., ICLR'25, arXiv 2410.10813). Complements AMA-Bench (agent trajectories) and LOCOMO
(multi-session dialogue) with the long-context *assistant*-memory shape: each of the 500 questions carries
its OWN ~115k-token haystack of 30-53 chat sessions, only a few of which hold the evidence. The haystack
dwarfs the 32k window, so the `full` arm must truncate — exactly the budget under which selection should
win (cf. §3.1 / §10.3).

One question → one `LMEEpisode` whose haystack SESSIONS are `Segment`s (one per session; `seg_id` =
`haystack_session_id`; `turn` = chronological index, so the recency tier = the latest sessions — what the
**knowledge-update** type needs; `text` = "[date]\n role: content" over the session's turns; `meta`
carries date/session_id). The gold `answer_session_ids` pin each answer to specific sessions → the clean
session-level selection-vs-reasoning attribution LOCOMO gave us via evidence dia_ids.

Six question types: single-session-user / -assistant / -preference, multi-session, temporal-reasoning,
knowledge-update. Abstention questions (`question_id` ending "_abs") are unanswerable — correct behaviour
is to say "I don't know"; scored separately as the abstention showcase (recall-grounded memory *knows*
whether evidence was retrieved). Public `_s` release = 500 Qs (`longmemeval_s_cleaned.json`).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .core import Segment


@dataclass
class LMEEpisode:
    qid: str
    qtype: str
    question: str
    question_date: str
    answer: str
    evidence_ids: list       # answer_session_ids — the gold evidence session(s)
    is_abstention: bool      # question_id ends "_abs": unanswerable, correct = abstain
    segments: list           # list[Segment], one per haystack SESSION, chronological (recent = later turn)


def _session_text(session: list) -> str:
    parts = []
    for t in session or []:
        role = t.get("role", "")
        content = (t.get("content") or "").strip()
        if content:
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def load_longmemeval(path: str) -> list[LMEEpisode]:
    data = json.load(open(path))
    eps: list[LMEEpisode] = []
    for q in data:
        sids = q.get("haystack_session_ids") or []
        sessions = q.get("haystack_sessions") or []
        dates = q.get("haystack_dates") or [""] * len(sessions)
        segs: list[Segment] = []
        for i, (sid, sess, date) in enumerate(zip(sids, sessions, dates)):
            body = _session_text(sess)
            text = f"[{date}]\n{body}" if date else body
            segs.append(Segment(seg_id=str(sid), turn=i, text=text, kind="session",
                                meta={"date": date, "session_id": str(sid)}))
        qid = str(q.get("question_id", ""))
        eps.append(LMEEpisode(
            qid=qid, qtype=str(q.get("question_type", "")), question=str(q.get("question", "")),
            question_date=str(q.get("question_date", "")), answer=str(q.get("answer", "")),
            evidence_ids=[str(x) for x in (q.get("answer_session_ids") or [])],
            is_abstention=qid.endswith("_abs"), segments=segs,
        ))
    return eps
