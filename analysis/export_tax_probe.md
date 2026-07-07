# The export tax: why memory-side reasoning cannot be exported without paying for it

**Probe wave (2026-07-06), small-set screening. n=144 QA (12 episodes, 2/domain stratified
from the h0 half, seed 0), Qwen3-32B reader+judge, flagship memory (hybrid+pin), all arms
same batch on identical questions (143 common after joining). CIs are ±7pp per arm — every
delta below is a paired, directional reading, not a headline number.**

## Question

The relay ladder (`plan_poisoning_case_study.md`) ended at: the structured pass's derivation
never leaves the thinking block (99% of artifacts begin `Answer[1]:`), and even the correct
answer barely transfers to a default reader. Follow-up: can careful prompt + few-shot work
force the derivation *out* (blocker 2), and a verification-framed reader instruction make it
*adopted* (blocker 3)?

## Arms

| arm | upstream instruction | artifact | reader instruction | acc | Δ vs PRSTR (paired) |
|---|---|---|---|---|---|
| PRDEF | — (in place) | — | default | 0.671 | +0.007 |
| PRSTR | structured (in place) | — | — | 0.664 | — |
| PRTRU | structured | bare answer | trust | 0.636 | −2.8 |
| PRAV | structured | bare answer | verify-lite | 0.615 | −4.9 |
| PRX1 | structured + **export demand** | Evidence/Derivation/Answer | verify (named-flaw) | 0.608 | −5.6 |
| PRXC | structured + citation demand | answer + `[evidence: steps N,M]` | verify-lite | 0.601 | −6.3 |
| PRXD | structured + export demand | Evidence/Derivation/Answer | trust | 0.573 | −9.1 |
| **PRSXP** | structured + **export demand** | **answered in place — no relay** | — | **0.587** | **−7.7** |

Export demand = format requirement + schematic few-shot: visible reply must contain
`Evidence:` (verbatim quotes with step numbers), `Derivation:` (numbered steps ending in a
no-later-step-overturns check), `Answer[1]:`.

## Mechanism readings (from per-question artifact dumps)

| arm | format compliance | all-quotes-verbatim | reader adoption |
|---|---|---|---|
| PRTRU (bare answer) | 0% (n/a) | 54% | 39% |
| PRAV (bare answer) | 0% (n/a) | 50% | 41% |
| PRX1 (export+verify) | **99%** | **23%** | **94%** |
| PRXD (export+trust) | 99% | 19% | 67% |
| PRXC (citation only) | n/a | 57% | 83% |
| PRSXP (export, in place) | 99% | — | 100% (by construction) |

(Adoption is a string heuristic between the upstream answer and the reader's final answer; it
under-detects on verbose bare-answer artifacts — compare within-column, not across.)

## Findings

1. **Both original blockers fall mechanically to prompting.** The format demand + one
   schematic few-shot exports the derivation in 99% of artifacts (default behavior: ~1%).
   The verify-with-named-flaw instruction achieves 94% adoption (blanket trust: 67% on the
   same artifacts). Prompt engineering *can* pierce the thinking block and *can* make the
   reader adopt.

2. **But the export demand itself taxes the answering pass: −7.7pp with no relay involved**
   (PRSXP vs PRSTR, same batch, paired). This is the cleanest single number of the wave:
   identical memory, identical questions, identical structured instruction — the only change
   is "your visible reply must lay out the derivation," and the in-place score drops from
   0.664 to 0.587. PRX1 (0.608) ≈ the taxed upstream faithfully relayed (94% adoption); its
   deficit is inherited, not added.

3. **The tax has a fingerprint: quote decay and soft-type victims.** Only the two
   derivation-demand arms collapse to 19–23% all-quotes-verbatim (every other arm ~50–57%):
   the model derives in `<think>`, then *re-writes* the Evidence section from memory of its
   thinking rather than re-copying from context. Judged victim cases (PRX1 adopted-wrong ∧
   PRSTR-right) concentrate in counterfactual / strategic-next-action question types, where
   committing prose sub-conclusions early locks phrasing that the rubric then rejects.

4. **Even the minimal demand pays.** A one-line citation requirement
   (`[evidence: steps N,M]`) — no quotes, no derivation prose — still nets −6.3 relative to
   in-place structured and −3.5 relative to the bare-answer relay, despite buying 83%
   adoption.

5. **Verification framing does not help bare answers.** PRAV (verify-lite) ≤ PRTRU (trust)
   (−2.1, directional): a verification procedure on an artifact with nothing checkable
   invites re-derivation; blanket trust adopts more. The verify instruction's 94% only
   appears when there is a derivation to check — which only exists if you pay the tax.

6. **Nothing beats the dumbest relay.** Bare answer + blanket trust (PRTRU, −2.8) tops every
   artifact enrichment tried; and it in turn loses to just answering in place.

## Revised verdict on blocker 2

The original phrasing — "the derivation is architecturally non-exportable" — was about
default behavior. The corrected, stronger form: **a tax-free export does not exist.** The
visible derivation either does not exist (default), or exists already corrupted by the
demand that it exist. The thinking block is not hiding the derivation from the pipeline; it
is *protecting* it — the freedom to derive without committing prose is where the structured
instruction's value lives, and any demand that pierces that veil pays back the gain it was
trying to move. This is the export-side dual of the relay ladder's conclusion: reasoning
must happen in the answering pass, *unobserved*.

## Caveats

- Small-set screening (n=144, single batch): per-arm CIs ±7pp; the 8-arm consistency, the
  quote-decay signature, and the victim-type concentration carry the story, not any single
  delta. A full-half (n=1248) PRSXP-vs-anchor run would make the tax number publishable.
- This stratified subset shows **no default→structured gap** (PRDEF 0.671 ≈ PRSTR 0.664;
  on the full h0 the gap is +7.8pp) — equal domain weights dilute TEXT2SQL and n is small —
  so this set measures relay/export *friction*, not gap *recovery*. Do not read PRDEF≈PRSTR
  as contradicting the factorial.
- Files: `results/mu_merged_PR{STR,DEF,TRU,AV,X1,XC,XD,SXP}.json`,
  `results/sel_PR{TRU,AV,X1,XC,XD,SXP}_full.jsonl`; analysis in
  `analysis/s1_verdict.py`-style scripts (worker `stage_w7/`).

---

# Full-scale confirmation (n=1248, test_h0, same-batch — 2026-07-06 15:18)

The small-set screening under-estimated the tax. Same-batch full-half run,
paired on 1,239 common questions:

| arm | config | acc |
|---|---|---|
| RESTR4 | structured, in place (anchor; band 0.629–0.642) | **0.6354** |
| SXPH | structured + export demand, in place, no relay | **0.5192** |
| HOFX | export demand + verify-adopt relay | **0.5304** |

**1. The tax at scale: −11.8pp, CI[−14.4, −9.1]** (paired bootstrap; fixed 74 /
broke 220). Not only does the export demand refund the structured instruction's
entire +7–8pp gain — it **overshoots into net harm**: SXPH 0.519 sits *below
every default-instruction anchor* (REDEF2 0.564, HOFS 0.583, REDEF 0.597).
Demanding a visible derivation makes the structured instruction worse than the
one-line default.

**2. The mechanism numbers replicate exactly**: compliance 98–99%, reader
adoption 95%, all-quotes-verbatim 21% (small set: 99% / 94% / 23%).

**3. The relay is faithful and irrelevant: HOFX − SXPH = +1.1pp, CI[−1.1, +3.5]**
(zero-inclusive). The verify reader adopts what it is given and inherits the
damage. HOFX 0.528 ≪ HOFT 0.593 (trust-relay of the *untaxed* pass's bare
answer, joint n=1239): adoption was never the binding constraint — upstream
quality is.

**4. The tax lands on exactly the questions the instruction exists to fix.**
On the F+ pool (default-wrong ∧ structured-right, n=180): RESTR4 re-roll 0.722
→ SXPH **0.411** (−31pp; HOFX 0.433) — the demand destroys ~43% of what
structured wins there, while HOFT holds 0.589. On F− (n=83) SXPH ≈ anchor
(0.422 vs 0.434): pure harm, no compensating class.

**5. Concentration confirms the small-set fingerprint.** By qa_type, Type B
(strategic/reasoning) pays −20.6pp, A (exact recall) −11.1, D −9.1, C −6.0.
By domain: TEXT2SQL −15.7, OPENWORLD −15.0, SOFTWARE −14.4 vs EMBODIED −4.7,
WEB −6.1.

**Verdict, final form.** Forcing the derivation into the visible reply is not a
lossy export of the reasoning — it is a *different, worse reasoning process*:
the model commits prose sub-conclusions as it thinks (Type B pays most),
re-writes evidence from memory instead of re-copying (21% quote fidelity), and
the visible-format constraint binds precisely where the instruction's freedom
was producing the gain (F+). The thinking block is not hiding the derivation
from the pipeline; it is protecting the conditions under which the derivation
is any good. Reasoning must happen in the answering pass, unobserved.

Files: `results/mu_merged_{RESTR4,SXPH,HOFX}.json`,
`results/sel_{SXPH,HOFX}_full.jsonl`; analysis `analysis/w8_verdict.py`
(paired-bootstrap CIs, F± pools, domain/type splits).

---

# Decomposition correction (2026-07-06 16:30) — the tax is mostly answer-form, not process

**User audit challenge: "差的也太多了" — and the audit vindicated it.** SXPH's final
answers are 3.5× shorter than the anchor's (median 869→247 chars): the terse few-shot
(`Answer[1]: desk 1`) re-shaped the final answer, confounding reasoning damage with
completeness-credit loss at the judge. Two same-day controls (n=1239 paired):

| arm | manipulation | med chars | Δ vs RESTR4 [CI] |
|---|---|---|---|
| SXPH | export demand, terse few-shot | 247 | −11.8 [−14.4,−9.1] |
| SBRF | brevity-only ("≤3 sentences"), **no derivation** | 389 | **−7.5 [−10.1,−4.9]** |
| SXPF | export demand + complete-answer demand & long few-shot | 513 | **−5.3 [−8.0,−2.7]** |

- The four arms sit on one monotone curve: penalty ∝ how far the reply is compressed
  away from the instruction's natural output (247/389/513/869 → −11.8/−7.5/−5.3/0).
- **F+ pool (n=180): brevity-only loses the same as full export (0.722→0.500 vs 0.511)** —
  on exactly the questions the structured instruction fixes, the damage is carried by
  answer completeness, not derivation corruption.
- SXPF's −5.3 residual, whose answers are still 41% shorter than anchor, is therefore an
  **upper bound** on any pure "observation corrupts the process" effect; the data are
  consistent with full mediation by answer completeness. The earlier "Verdict, final form"
  section above is **superseded** on attribution (its mechanism numbers — 99% compliance,
  95% adoption, 21% verbatim quotes, faithful relay — all stand).
- What remains export-specific: quote fidelity (21% all-verbatim) — the visible Evidence
  section is post-hoc reconstruction, not a record of the thinking. But the accuracy loss
  can no longer be attributed to that.

**Corrected moral.** The eliciting instruction's *visible output form is part of its
mechanism*: its gain flows partly through the complete, multi-part final answers it
elicits (note SBRF ≈ default anchors — structured with short answers ≈ default). Any
secondary demand that re-shapes the reply — derivation sections, citation lines,
brevity — trades that away, roughly in proportion to the compression. Paper updated
accordingly (law clause "nor observed without corrupting it" → "its reply cannot be
re-shaped"). Pre-registered windows: SBRF 0.56–0.59 (actual 0.559 ✓ edge), SXPF
0.58–0.61 (actual 0.581 ✓).

Files: `results/mu_merged_{SBRF,SXPF}.json`, `results/sel_{SBRF,SXPF}_full.jsonl`,
`analysis/w9_verdict.py`, `analysis/w8_audit.py` (the length/leak audit that caught it).

---

# Endgame (2026-07-07 00:40) — one curve, one constant, one trade, one winner

Final wave (all same-batch n=1239 paired unless noted; full-set n=2476):

| arm | config | med chars | acc | Δ [CI] |
|---|---|---|---|---|
| SXPV | export + completeness v2, in place | 786 | 0.6239 | −1.1 [−3.6,+1.6] vs RESTR4 (tax fully recovered) |
| RVERB | completeness only, in place | 1443 | 0.6360 | +0.2 [−2.2,+2.5] (flat above natural length) |
| HOFV | zero-tax artifact + verify relay | 666 | 0.5900 | −4.4 (= HOFT2: relay fixed point) |
| HOFT2 | bare artifact + trust relay | 522 | 0.5900 | −4.4 — **both on the compression curve** |
| **HOFTW** | trust relay + complete re-emission | 1664 | **0.6586** | **+2.4 [+0.1,+4.7]** |
| **HOFTW2** | replicate | — | **0.6635** | **+2.8 [+0.7,+5.1]**; agreement 85% |
| **HOFTWF** | full 2496 set | 1638 | **0.6631** | **+1.7 [+0.1,+3.3]** vs RESTRF 0.6454 |
| DVERB | default + completeness (appended) | 1111 | 0.6126 | +3.0 [+0.4,+5.6] vs REDEF3 |
| DVERB2 | default + completeness (prefix replaced) | 1542 | 0.6320 | −0.2 [−2.7,+2.3] vs RESTR4 — **tie** |
| REDEF3 | default anchor (same batch) | 348 | 0.5827 | — |
| DMANI | requirements checklist as context | 379 | 0.5681 | −1.8 [−4.2,+0.7] null; length unmoved |
| DSCAF | evidence worksheet as context | 353 | 0.5777 | −0.7 [−2.9,+1.5] null; guard rejects 74%, served subset −3.1 |

**The four claims.**
1. **One curve**: score ≈ anchor − f(final-answer compression); monotone below the
   natural output length, flat above (RVERB). The relay "friction" (0.593 fixed
   point) was reader-side compression, not relay loss.
2. **One constant**: answer completeness — transferable to the default instruction
   by one sentence (DVERB2 ties structured), an additive constant any system can
   add; we disclose it and recommend leaderboards fix the answer instruction.
3. **One trade**: structured's reasoning content = win F+ (0.372→0.722, of which
   completeness alone reaches 0.650) / lose F− (0.699→0.434); net ≈ 0 beyond
   completeness.
4. **One winner**: the composed relay (structured pass upstream → trust + complete
   re-emission downstream) keeps the F+ wins in full AND repairs part of F−
   (0.506-0.627 vs anchor 0.434) — the only configuration above the in-place
   anchor, three independent zero-excluded confirmations incl. the full official
   set (0.6631 > headline 0.6466 > prior best 0.6246). Not submitted: its edge is
   architecture × completeness, disclosed as mechanism evidence, not leaderboard
   fodder.

**Memory-side presentation is closed**: neither the checklist nor the
verbatim-evidence worksheet moves the reply length (348→353–379) — content
cannot reach emission behavior; the worksheet also reprises the fabrication
signature (26% guard-pass; 4th independent sighting: packet 31%, export 21–23%,
DSCAF 26–28% faithful).

Paper updated (relay-ladder passage v3, law v3, reader-axis framing paragraph
with the irreversibility asymmetry, tab:reader +2 rows, abstract decomposition
note). Files: mu_merged_{SXPV,RVERB,HOFV,HOFT2,HOFW,HOFTW,HOFTW2,HOFTWF,RESTRF,
DVERB,DVERB2,REDEF3,DMANI,DSCAF}.json + sel dumps.

---

# Evidence-priced override: the gate arm (2026-07-07 02:05, HOFTC)

User follow-up ("让 reader 更采信记忆侧?") inverted: maximizing adoption is the wrong
objective (perfect adoption ⇒ score ≈ upstream ≈ anchor, forfeiting the relay's
error-correction surplus). The right objective is override *calibration*: HOFTC =
HOFTW + one sentence — "Depart from its final answer ONLY if you can name the
specific step whose text contradicts it."

**HOFTC 0.6747 [0.649, 0.701] — project best. +4.0 [+1.5,+6.5] over the same-batch
structured anchor; +1.1 [−1.2,+3.6] over HOFTW2.** Mechanism: override *rate* on the
anchor-right pool unchanged (85 vs 83) — the gate raises override *precision*:
F+ 0.722→0.767 (now above the anchor on structured's home turf) and F− 0.518→0.554.
Requiring named evidence makes departures right more often, everywhere. Pre-registered
window 0.665–0.675 (calibration works) hit at the top edge.

Files: `results/mu_merged_HOFTC.json`, `results/sel_HOFTC_full.jsonl`. Full-set
confirmation (HOFTCF) + router-attribution verification arms queued overnight.

---

# Overnight finale (2026-07-07 04:00–07:30)

**Full-set confirmations (n=2496, same-batch):** HOFTCF (gate relay) **0.6715**
[0.653,0.690] vs RESTRF 0.6454 (+2.6; h0 0.6747 replicates); DVERB2F 0.6414 vs
REDEFF 0.5809 (+6.1) vs RESTRF (−0.4 tie) — the completeness attribution closes
on the official set.

**Different-family judge (Llama-3.1-8B re-grade, 12.5k rows):** every ordering
preserved; completeness margin grows to +12.7 (DVERB2 0.680 vs REDEF3 0.552),
relay margin to +12.3 (HOFTW2 0.769 vs RESTR4 0.647), HOFTC top at 0.776;
compression curve stays monotone (0.473/0.552/0.557/0.655/0.692). Neither
effect is a same-family-judge artifact. `results/xjudge_out{,2}.json`.

**Mini case rig (11 cases × G0–G3, results/minicase_out.txt):** G3
(assume-correct) adopts 7/8 bad overrides but kills 3/3 good ones — the
max-trust trade in miniature; gates G1/G2 intermediate; pool membership drifts
with upstream rolls (case 2's "overrides" all land the correct answer tonight).

**Router audit (the correction):** deployed default "hybrid" = RRF{lexical,
model-pick}, whose 32-token pick calls emit thinking-truncated stubs. Same-day
paired triple (pin fixed, structured reader): lexical-only **0.6506** >
+embedding 0.6410 > +model-pick (deployed) 0.6226; lexical−deployed = **+2.7
[+0.5,+5.0]** zero-excluded; lexical≈embedding. All paper differences
unaffected (config constant within batch); absolutes carry ~1–3pp handicap,
reported as-run; recommended default → pin over lexical base. First embed-arm
attempt crashed (4 embedders OOM on vLLM-full GPU0) — fixed layout NINST=3 +
embedder on GPU6. Paper corrected (6 sites + audit paragraph); tab:config/
tab:ablation/tab:router labels fixed.

Files: mu_merged_{HOFTCF,DVERB2F,REDEFF,RESTR5,RESTREMB,RESTRLEX}.json,
sel_HOFTCF_full.jsonl.

---

# Full-set router replication: the pin+lexical headline rerun (2026-07-07 12:28, w20)

User: "pin+lexical 重跑 headline 吧,跑完对比一下." Same-batch trio (n=2496, SRC=test,
structured reader): RESTRLEXF (pin over lexical), RESTRF2 (deployed cfg_flagship
re-anchor), HOFTCL (lexical base + gate relay). Pre-registered windows: RESTRLEXF
0.66–0.68 (the audit's +2.7 carried over), RESTRF2 0.635–0.655, HOFTCL 0.68–0.70.

**RESTRLEXF 0.6466 [0.628,0.665] — below window. The half-set handicap does NOT
replicate:** RESTRLEXF−RESTRF2 (0.6538) = **−0.8 [−2.3,+0.8] n.s.** (wins 181 /
losses 200); vs the earlier full-set anchor RESTRF 0.6454, +0.2 n.s.; RESTRF2−RESTRF
(same config, two batches) +0.9 n.s. — run-to-run scale ≈1pp. Per-domain the two
bases trade: lexical wins EMBODIED_AI +2.6 / Game +1.7, loses OPENWORLD_QA −3.1 /
TEXT2SQL −3.1. Verdict: at full scale the model-pick fusion is score-neutral (≈0),
not a 1–3pp tax; the recommended-default switch to lexical now rests on cost (one
LLM pick-call per retrieval saved, no malformed stubs), not on score. No official
resubmission warranted — the submitted 0.6466 headline stands (RESTRLEXF lands on
the same number by coincidence).

**HOFTCL 0.6707 [0.653,0.690]:** the gate relay on the lexical base — **+2.3
[+0.7,+3.9] zero-excluded** over its same-batch in-place anchor (4th independent
relay confirmation), −0.2 n.s. vs HOFTCF (hybrid base) → the relay margin is
router-base-indifferent. HOFTCL−RESTRF2 +1.5 [−0.0,+3.1] grazes zero.

Paper: audit paragraph rewritten (pre-registered full-set replication sentence,
handicap claim withdrawn → "score-neutral at full scale" + cost-based default
rationale), tab:config ≤0 → ≈0, gate sentence gains the router-base-indifference
clause. Methodology note: a zero-excluded n=1248 CI (+2.7 [+0.5,+5.0]) failed to
replicate at n=2496 — treat half-set zero-exclusion as screening, only full-set
paired runs as confirmatory.

Files: mu_merged_{RESTRLEXF,RESTRF2,HOFTCL}.json, sel_HOFTCL_full.jsonl;
paired analysis: /home/tiger/w20_verdict.py.

---

# HOFTC 失误解剖:官方尺度 2×2 + rider 措辞 mini 筛(2026-07-07 16:00,j2x2-j2x4 + mc2)

用户 goal:HOFTC 还错什么、是不是记忆侧、能否提升。三轮判分(严判 → 官方 prompt 复刻
agreement 99.5%,3-worker 分片 → 发现 dump 截断 bug:up_ans 被截 300 字符、96.4% 触顶,
从 art 重抽全文重判)后的官方尺度 2×2(n=2356,上游=中继的记忆侧 structured 趟):

**KEPT 1428 (60.6%) / BOTHWRONG 651 (27.6%) / REPAIRED 157 (6.7%) / DESTROYED 105 (4.5%)。
上游 0.651 ≈ 同批原地锚 0.6466(记忆侧作答满血,无隐藏损失);中继净 +52 题(修复:摧毁
1.5:1),分域 SOFTWARE +30 / TEXT2SQL +23 / Game −3 / OPENWORLD −8。**

错例(764)判词:85% = BOTHWRONG(证据在场、双趟都推错:归因偏步、计数滑落、反事实带偏),
其中 32% 从未被 73 arm 任一做对(gold/judge 噪声——ep150 实锤 gold 步号错位:gold 称 step 15
用 grep,原始轨迹 turn 15 是 str_replace_editor view、grep 在 turn 16,73 个 arm 全按轨迹答)、
20% 仅 1-2 arm 碰对(重掷边缘)、34%(222 题,~9pp)稳定可解但均匀摊在全部域×题型 → 无单一
杠杆 = 已八连攻确认的推理天花板。14% = DESTROYED(上游对、reader 重发射毁掉):86% 稳定可解,
模式 = 完整性 rider 的 "5-12 句" 在短 gold 题上逼出过度展开(5 步写成 4 步、坐标写错)+ 多要素
改写漂移 + judge 边界。**记忆侧检索与作答均已饱和——剩余错误不是记忆侧的锅。**

**mini 筛(mc2,12 DESTROYED + 12 KEPT,上游 artifact 用实跑原件,官方 judge):现行 rider
CUR 自发救回 4/12(DESTROYED 池同样有 ~33% 重掷边缘性)、KEPT 12/12;长度自适应 ADA 救回
7/12(配对 +3)但 KEPT 破 1/12。基率算术:KEPT 体量是 DESTROYED 的 13.6×,破 1/12(8.3%)
的点估计 = 4.8× 盈亏线(1.8%),外推净 −3.9pp → 与 ADAPT 同构("拿身体换尾巴")。判决:
不上规模,杠杆关闭;DESTROYED 是修复引擎(+157)的价格,不是可摘除的缺陷。**

工程:dump 截断已根治(agentic_reader.py art[:9000]/up_ans[:4000]/ans[:2000],同步 worker);
官方 judge prompt 完整复刻件在 j2x3/j2x4_run.py;3-worker 分片判分 ~5 分钟/轮。
Files: analysis/hoftcl_{autopsy,2x2}.py, results/j2x{2,3,4}_*, /home/tiger/mc2.log。
