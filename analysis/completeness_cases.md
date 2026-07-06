# Why does completeness score? Two mechanisms, not one (case audit, 2026-07-07)

Pools: REDEF3 (default, med 348 chars) vs DVERB2 (default + completeness demand, med
1542 chars), same batch, n=1239 paired. Fixed (short-wrong → long-right) n=171;
broke n=110. Eight randomly sampled cases (seed 1) annotated below; raw rows in
`results/mu_merged_{REDEF3,DVERB2}.json`.

## Mechanism A — coverage credit (the judge counts gold elements)

- **FIXED 3 (TEXT2SQL/C, filter definition)**: the short answer is substantively
  right (step 14; all three filter conditions) but omits the persistence argument
  ("remained consistent across 14–20") that the gold names; the long answer adds it
  and passes. Right-but-incomplete → 0.
- **FIXED 5 (TEXT2SQL/B, preconditions)**: short lists 3 preconditions; gold wants
  "several interdependent" with more elements. Element counting.

## Mechanism B — forced thoroughness (the demand changes the derivation itself)

- **FIXED 6 (EMBODIED/C, pan locations)**: short answer asserts "pan 1 and pan 2
  both on countertop 1" — factually wrong, not an omission. The
  enumerate-per-object-with-citations obligation forces a per-item evidence walk
  (finds `take pan 1...`, tracks pan 2 into inventory) and lands right.
- **FIXED 1 (TEXT2SQL/A, causal chain)**: short cites the wrong steps (3/7) with the
  wrong mechanism; long re-walks from step 2 and recovers the actual grep-empty
  chain.
- **FIXED 2 (OPENWORLD/C, workspace state)**: short misses the step1 file's
  existence; per-file enumeration catches it.

## Mechanism B in reverse — why the broke pool breaks

- **BROKE 1 (OPENWORLD/A, exact query)**: short = the exact Japanese string
  (correct). Long appends invented context ("results from the official website...")
  and fails. Forced elaboration adds claims a terse correct answer never risked.
- **BROKE 2 (WEB/C, domain count)**: short counts 3 domains (right); the forced
  detailed re-derivation merges one and answers "two" (wrong). Re-derivation under
  elaboration pressure changed a correct conclusion.

## Verdict

The "completeness constant" is a fusion: **(A) coverage-credit recovery** for
elements the model knew but omitted (the grading-surface part), and **(B) a mild
reasoning elicitation smuggled inside an output-form demand** — the
enumerate-with-citations obligation forces per-element evidence walks that fix
flatly-wrong state-tracking (F+ side) and corrupt quick correct answers (F− side,
symmetric). In this 8-case sample the split is roughly even, with B carrying the
hard reversals. Honesty note: fixed/broke pools also carry judge borderline churn
(~1/3 by the earlier spontaneous-recovery control); the unambiguous B-type
reversals (pan 2, domain count) are the load-bearing evidence. Paper updated with
the two-sided sentence in the attribution passage.
