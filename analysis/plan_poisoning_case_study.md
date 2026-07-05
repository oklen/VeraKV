# Why memory-side reasoning fails where answering-pass reasoning succeeds
## (plan-poisoning case study — the goal investigation)

**Setup.** PLDEF = a memory-side call generates an EvidencePlan (sub-questions + verbatim
citations + local conclusions + composition; expressly forbidden to answer), prepended for a
default reader. Same model, same evidence, same day as the anchors. Result was −5.6pp.

**Forensics that CLEARED the mundane suspects** (all 1,248 archived plans):
- think-leak (unclosed `</think>` → raw thinking shipped as artifact): **0/1248**
- format truncation: 95% contain `Sub-question 1`, **87% reach `Composition`** — plans are
  structurally complete, not budget-mangled (though the 2048-token cap incl. thinking vs the
  answer call's 8192 remains an asymmetry worth fixing on principle).

**The sharp slice: 88 questions where the plain default reader was RIGHT, structured was RIGHT,
and the plan made the answer WRONG.** Five read in depth below. The plans are well-formed,
evidence-cited, professional-looking — and each poisons the answer through one of five faces of
a single mechanism:

1. **Absence commitments freeze early** — "not found" is an acceptable plan output; once written,
   the reader inherits the denial. In the answering pass "not found" is a failure state the model
   resists (it keeps searching, and finds Turn 4).
2. **The plan's sub-question framing displaces the question's own rubric** — the reader answers
   the plan, not the question.
3. **Modality misreads get frozen** ("what had to be true for success" answered as "what went
   wrong") — in the answering pass the final compose step re-collides with the question and
   often self-corrects; the plan never has to cash out.
4. **Over-collected side-facts steer composition wrong** (a constant RootWebArea ID pushed the
   answer to "no reload" against gold's "new view loaded").
5. **Concrete micro-errors are inherited verbatim** (plan counts take=1, gold take=2; the
   no-plan reader miscounts differently and still passes).

**Mechanism (one sentence): the plan is produced without answer-accountability and consumed with
authority — derivation and commitment are separated, so nothing upstream is ever checked against
"does this answer the question", and nothing downstream re-checks it.**

**VERDICT (all cells run; same-batch h0):** the fix experiments completed the picture with two
more independent blockers beyond the accountability diagnosis:

- **Layer 2 — the process is not exportable.** When the memory side runs the full structured pass
  (HOFA), 99% of visible artifacts BEGIN with the final answer: with a reasoning model the
  derivation happens inside the thinking block and never appears in the output. There is no
  process to hand over — only the product.
- **Layer 3 — even the product barely transfers.** Correct answer + evidence in the default
  reader's context: conditional retention 0.797 vs 0.773 with no artifact (+0.4pp overall). An
  explicit trust-relay sentence: 0.816 (0.593 overall, still −4.7 below structured-in-place).
  The reader re-litigates regardless; behavior is set by the instruction, not context content.
- **Few-shot works as a poison-reducer, not a cure:** PLAN2 (anti-absence + modality-mirroring +
  worked contrasts + full budget) −5.6 → −2.1pp, poisoned-pool recovery 46/88 (v1: 0/88).
- HOFS implementation note: the answer-strip emptied 98.7% of artifacts (answer-first format);
  it served as a third default re-roll anchor (0.5811; default re-roll band ±2pp).

Ladder: 0.562 (no artifact) → 0.567 (answer in context) → 0.593 (answer + trust) → 0.640
(structured in place). Monotone in how much of the answering pass the reader keeps.

**The original fix prediction (for the record):** make the memory-side call answer-accountable —
run the full structured derivation *including committing to an answer* at full budget upstream;
hand the artifact to the default reader. HOFS deterministically strips the final-answer line from
the SAME upstream generation, separating "the answer transfers" from "the derivation transfers".

---


## Poisoned case 1 [OPENWORLD_QA | C | ep 84]
**Q:** In which turn did the agent first extract the detail that Tidal produced the singles 'Shadowboxer', 'Sleep to Dream', and 'Criminal'?

**GOLD:** Turn 4

**Default, no plan (1):** The detail about Tidal producing the singles 'Shadowboxer', 'Sleep to Dream', and 'Criminal' is not explicitly mentioned in any of the provided steps. The steps focus on verifying album release dates, Christgau grades, and reconciliation of discographies, but 

**Structured (1):** The detail that Tidal produced the singles "Shadowboxer," "Sleep to Dream," and "Criminal" is **not explicitly mentioned** in any of the provided steps. The steps reference Fiona Apple's *Tidal* (1996) album (e.g., steps 9, 15, 25, 33, 35, 43, 45), but none of

**THE PLAN (memory-side artifact):**
```
Sub-question 1: Did any step explicitly mention the singles "Shadowboxer," "Sleep to Dream," or "Criminal" in relation to Fiona Apple's *Tidal*?  
  Evidence: None of the steps (0–47) reference these specific singles.  
  Local conclusion: The agent never explicitly extracted or recorded these singles in the provided steps.  

Sub-question 2: Was there a step where *Tidal* (Fiona Apple) was identified as an album but without details about its singles?  
  Evidence: <step 9> "Fiona Apple - Tidal (1996)" and <step 35> (AllMusic search for *Tidal*) list the album but do not mention singles.  
  Local conclusion: The agent confirmed *Tidal* as an album but did not extract its singles.  

Composition: The agent never extracted the detail about the singles "Shadowboxer," "Sleep to Dream," and "Criminal" in any of the provided steps. The question’s premise is not supported by the evidence.
```
**Plan+default answer (0):** The agent never extracted the detail about the singles "Shadowboxer," "Sleep to Dream," and "Criminal" in any of the provided steps (0–47). The evidence confirms that while *Tidal* was identified as an album, no step explicitly mentioned its singles.


## Poisoned case 2 [SOFTWARE | D | ep 152]
**Q:** What were the key steps between discovering the location of the bug and successfully implementing the fix?

**GOLD:** The key steps were: 1) Using grep to search for the error message 'Cannot filter a query once a slice has been taken' which led to finding the problem in related_descriptors.py at line 1007, 2) Creating a reproduction script that confirmed the issue existed, 3) Analyzing the get_prefetch_queryset me

**Default, no plan (1):** The key steps were: 1. **Identifying the issue**: Located the bug in `get_prefetch_queryset` where sliced querysets (`is_sliced`) caused errors due to Django's "sticky" queryset behavior. 2. **Designing the fix**: Created a helper function `filter_sliced_query

**Structured (1):** The key steps between discovering the bug location and implementing the fix were: 1. **Identifying the error source** (Steps 15-18, 27-29, 121): - Located the `TypeError("Cannot filter a query once a slice has been taken")` in `django/db/models/query.py` (Step

**THE PLAN (memory-side artifact):**
```
Sub-question 1: How was the location of the bug identified in the Django codebase?  
  Evidence: <step 10> "find . -name \"*.py\" -type f | xargs grep -l \"Prefetch\" obs=... django/db/models/query.py ... tests/prefetch_related/tests.py"  
  Evidence: <step 13> "grep -r \"Cannot filter a query once a slice has been taken\" obs=django/db/models/query.py: raise TypeError(...)"  
  Local conclusion: The bug was traced to `django/db/models/query.py` and `django/db/models/fields/related_descriptors.py` via code searches for "Prefetch" and the error message.  

Sub-question 2: What was the core issue with sliced querysets and `Prefetch`?  
  Evidence: <step 36> "The issue is that when using `Prefetch()` with a sliced queryset, Django raises 'Cannot filter a query once a slice has been taken.'"  
  Evidence: <step 119> "view_range [75,105] shows `filter_sliced_queryset` temporarily removes slices to apply filters."  
  Local conclusion: Sliced querysets could not be modified (e.g., filtered) during `Prefetch`, requiring temporary slice removal.  

Sub-question 3: How was the fix implemented in the code?  
  Evidence: <step 110> "Replaced old code with logic to save/clear/reapply slices in `get_prefetch_queryset`."  
  Evidence: <step 119> "filter_sliced_queryset saves low/high marks, clears limits, applies filters, then restores limits."  
  Local conclusion: The fix introduced a helper function to handle slices during filtering in prefetch logic.  

Sub-question 4: How was the fix 
```
**Plan+default answer (0):** The key steps were: 1. **Bug Location Identification**: Code searches for "Prefetch" and error messages traced the issue to `django/db/models/query.py` and `django/db/models/fields/related_descriptors.py` (steps 10, 13). 2. **Core Issue Analysis**: Recognized 


## Poisoned case 3 [WEB | C | ep 188]
**Q:** Compare the 'ID' of the 'Next' button in Step 14 ([19083]) and Step 15 ([21032]). The button label and function are identical. What does the change in the numeric ID imply about the underlying page state between these two steps?

**GOLD:** The ID change implies that the page was re-rendered or a new view was loaded (likely moving from one review record to the next), even though the button label remained the same.

**Default, no plan (1):** The change in numeric IDs implies that the page or a portion of the UI was re-rendered or reloaded between steps, likely due to navigation or state updates (e.g., menu interactions or data fetching). This is common in dynamic web apps where elements are regene

**Structured (1):** The change in numeric IDs between Step 14 ([19083]) and Step 15 ([21032]) for the "Next" button implies that the underlying page state was dynamically updated or re-rendered between these steps. This is evident because: 1. **Element Re-creation**: Web applicat

**THE PLAN (memory-side artifact):**
```
Sub-question 1: What is the numeric ID of the "Next" button in Step 14?  
  Evidence: <step 14> "Observation: ... [19083] button 'Next'"  
  Local conclusion: The "Next" button in Step 14 has ID [19083].  

Sub-question 2: What is the numeric ID of the "Next" button in Step 15?  
  Evidence: <step 15> "Observation: ... [21032] button 'Next'"  
  Local conclusion: The "Next" button in Step 15 has ID [21032].  

Sub-question 3: What is the RootWebArea ID in Step 14?  
  Evidence: <step 14> "Observation: ... [56235] RootWebArea '#000000299 / Orders / Operations / Sales / Magento Admin'"  
  Local conclusion: The RootWebArea in Step 14 is #000000299.  

Sub-question 4: What is the RootWebArea ID in Step 15?  
  Evidence: <step 15> "Observation: ... [56235] RootWebArea '#000000299 / Orders / Operations / Sales / Magento Admin'"  
  Local conclusion: The RootWebArea in Step 15 remains #000000299.  

Sub-question 5: What action occurs between Step 14 and Step 15?  
  Evidence: <step 14> "Action: click [56332] where [56332] is link '\ue60b SALES'"  
  Evidence: <step 15> "Action: click [57042] where [57042] is link 'Orders'"  
  Local conclusion: The user navigates from the SALES menu to the Orders submenu, staying within the same RootWebArea.  

Composition: The numeric ID of the "Next" button changes from [19083] to [21032] between steps, but the RootWebArea ID (#000000299) remains constant. This implies the page state (e.g., order context) is preserved, but the UI dynamically upda
```
**Plan+default answer (0):** The change in the numeric ID of the "Next" button between Step 14 and Step 15 indicates that the UI element was re-rendered dynamically without a full page reload. The RootWebArea ID (#000000299) remaining constant confirms the underlying page state (e.g., ord


## Poisoned case 4 [TEXT2SQL | B | ep 94]
**Q:** Before the step 9 query could compile successfully, what multiple schema-related preconditions had to be true about the four push-notification tables and the columns referenced (name at least 3 distinct conditions)?

**GOLD:** At least these preconditions must hold: 1) Each referenced column must exist in the specific table/CTE where it is selected. Step 9 selects "AD_TRACKING_ENABLED" from all four sources, but later evidence (step 18) implies USERS_MESSAGES_PUSHNOTIFICATION_INFLUENCEDOPEN_VIEW does not have that column,

**Default, no plan (1):** 1. The `AD_TRACKING_ENABLED` column must exist in all four tables (SEND, BOUNCE, OPEN, INFLUENCEDOPEN) to support the `COALESCE` logic in the final `SELECT`. 2. The join keys (`USER_ID`, `CAMPAIGN_ID`, `MESSAGE_VARIATION_ID`, `PLATFORM`) must be present in all

**Structured (1):** Three schema-related preconditions required for the step 9 query to compile successfully: 1. **Column Consistency Across Tables**: The `AD_TRACKING_ENABLED` column must exist in all four push-notification tables (`USERS_MESSAGES_PUSHNOTIFICATION_SEND_VIEW`, `B

**THE PLAN (memory-side artifact):**
```
Sub-question 1: Did the `AD_TRACKING_ENABLED` column exist in all four push-notification tables (send, bounce, open, influenced_open)?  
  Evidence: <step 9> "invalid identifier 'AD_TRACKING_ENABLED'"  
  Local conclusion: The `AD_TRACKING_ENABLED` column was missing in at least one table (likely `INFLUENCEDOPEN_VIEW`), causing the compilation error.  

Sub-question 2: Were the join keys (`USER_ID`, `CAMPAIGN_ID`, `MESSAGE_VARIATION_ID`, `PLATFORM`, `AD_TRACKING_ENABLED`) present in all tables involved in the joins?  
  Evidence: <step 13> "SELECT "TIME", "AD_TRACKING_ENABLED", "PLATFORM" [...] Query executed successfully"  
  Local conclusion: `AD_TRACKING_ENABLED` and `PLATFORM` existed in `SEND_VIEW`, but their absence in other tables (e.g., `INFLUENCEDOPEN_VIEW`) caused join failures.  

Sub-question 3: Were the `CARRIER`, `BROWSER`, and `DEVICE_MODEL` columns present in the `OPEN_VIEW` and `INFLUENCEDOPEN_VIEW` tables?  
  Evidence: <step 13> "SELECT [...] "PLATFORM" [...] Query executed successfully" and <step 15> `"CARRIER": "BROWSER": "DEVICE_MODEL":`  
  Local conclusion: These columns existed in `OPEN_VIEW` but were missing in `INFLUENCEDOPEN_VIEW`, leading to inconsistencies in the schema.  

Composition: The query required (1) consistent column existence across all tables for `AD_TRACKING_ENABLED`, (2) matching join keys in all joined tables, and (3) presence of device-related fields in both `OPEN_VIEW` and `INFLUENCEDOPEN_VIEW`. The error in step 9 arose because 
```
**Plan+default answer (0):** 1. **Column Consistency**: The `AD_TRACKING_ENABLED` column must exist in all four tables (send, bounce, open, influenced_open) to avoid compilation errors from missing identifiers. 2. **Join Key Availability**: All join keys (`USER_ID`, `CAMPAIGN_ID`, `MESSAG


## Poisoned case 5 [EMBODIED_AI | D | ep 46]
**Q:** Until step 75, what types of actions has the agent performed and how frequently?

**GOLD:** Action frequency: go (34 times), look (25 times), examine (3 times), help (3 times), open (3 times), close (3 times), take (2 times), move (2 times), inventory (1 times)

**Default, no plan (1):** The agent has performed the following actions up to step 75: - **look**: 24 times - **go to [object]**: 34 times - **take [object] from [location]**: 2 times - **move [object] to [location]**: 2 times - **examine [object]**: 3 times - **open [object]**: 3 time

**Structured (1):** The agent has performed the following actions up to step 75: 1. **Movement actions ("go to ...")**: 32 times (e.g., steps 1, 3, 5, 7, 11, 12, 14, 15, 20, 21, 25, 28, 30, 31, 33, 36, 40, 41, 48, 49, 51, 52, 58, 59, 61, 62, 64, 67, 69, 70, 72, 73). 2. **Take act

**THE PLAN (memory-side artifact):**
```
Sub-question 1: How many times has the agent performed the "look" action?  
  Evidence: <step 0> "action=look", <step 8> "action=look", <step 10> "action=look", <step 13> "action=look", <step 16> "action=look", <step 17> "action=look", <step 19> "action=look", <step 22> "action=look", <step 24> "action=look", <step 29> "action=look", <step 32> "action=look", <step 37> "action=look", <step 39> "action=look", <step 42> "action=look", <step 44> "action=look", <step 47> "action=look", <step 50> "action=look", <step 53> "action=look", <step 55> "action=look", <step 57> "action=look", <step 63> "action=look",
```
**Plan+default answer (0):** The agent has performed the following actions up to step 75: - **look**: 24 times - **go to**: 34 times - **take**: 1 time - **move**: 2 times - **examine**: 3 times - **open**: 3 times - **close**: 3 times - **help**: 3 times - **inventory**: 1 time.
