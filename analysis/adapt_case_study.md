# ADAPT design case study — 10 questions, one instruction each

Goal: derive DETERMINISTIC question->instruction routing rules. For each case: what the fixed
structured instruction did (fix or break, via what mechanism), and what the optimal per-question
instruction would have been. Annotations added after manual reading.


==========================================================================================
## Case 1 [F+ / enumeration] [WEB | type A | ep 192]
**Q:** How many Edit clicks occurred before the first Next click at Step 14?

**GOLD:** Three Edit clicks (Steps 9, 11, 13).

**DEF (0):** 2

**STR (1):** 3 The first "Next" click occurs at **Step 14**. Before this step, the "Edit" link was clicked in the following steps: - **Step 9**: `action=click [14278] where [14278] is link 'Edit'` - **Step 11**: `action=click [16539] where [16539] is link 'Edit'` - **Step 13**: `action=click [18800] where [18800] is link 'Edit'` **Total Edit clicks before Step 14**: 3.

**ANNOTATION:** STR 干了什么:强制"先列举后计数"——列出 step 9/11/13 三次 Edit 再数,DEF 直接吐 "2"。
最优指令:"先把每个匹配事件连步号列出来,再数这个清单。"
表面特征(可正则):`how many ... (before|between|until|at step)`——有界的单事件计数。


==========================================================================================
## Case 2 [F+ / state-timeline] [EMBODIED_AI | type C | ep 49]
**Q:** How did the state of drawer 1 change throughout the trajectory, including what objects were placed in or removed from it?

**GOLD:** Drawer 1 evolution: step 3: opened (action: open drawer 1); step 4: closed (action: close drawer 1); step 20: opened (action: open drawer 1); step 21: knife 1 was put inside (action: move knife 1 to drawer 1).

**DEF (0):** Drawer 1 started closed. It was opened (step 3), found empty, and then closed (step 4). Later, it was reopened (step 20) and **knife 1** was placed inside (step 21). Throughout the trajectory, drawer 1 remained open except when explicitly closed. In step 63, **knife 2** was added to drawer 1. No objects were removed from drawer 1 after placement.

**STR (1):** The state of drawer 1 changed as follows: 1. **Step 2**: Drawer 1 is closed. 2. **Step 3**: Drawer 1 is opened and found to be empty ("In it, you see nothing"). 3. **Step 4**: Drawer 1 is closed again. 4. **Step 20**: Drawer 1 is reopened (still empty). 5. **Step 21**: Knife 1 is moved into drawer 1 ("You move the knife 1 to the drawer 1"). 6. **Step 63**: Knife 2 is moved into drawer 1 ("You move the knife 2 to the 

**ANNOTATION:** STR 干了什么:按步重建状态时间线;DEF 顺序大致对但**加了一句无依据的尾巴**(step 63 断言)被判 0。
最优指令:"按时间逐步重建:仅在状态变化的步各写一行并引步号;不得添加引用步之外的断言。"
表面特征:`how did the state|location histor|state change|throughout`。


==========================================================================================
## Case 3 [F+ / evidence-binding (concreteness)] [SOFTWARE | type A | ep 175]
**Q:** At step 74, what exact problematic matrix pattern did the agent identify through debugging?

**GOLD:** [[1, 12], [0, 8], [0, 5]]

**DEF (0):** The exact problematic matrix pattern identified was matrices of the form [[a, b], [0, c], [0, d]] where a, b, c, d are non-zero. Specifically, the algorithm incorrectly handled cases where: - The matrix had 3 rows and 2 columns (m=3, n=2) - The first column had a non-zero entry in the first row but zeros in subsequent rows - The second column had non-zero entries in all rows This pattern caused the HNF algorithm to e

**STR (1):** The exact problematic matrix pattern identified at step 74 was a **3×2 matrix** with the following structure: - First column: `[a, 0, 0]` (non-zero `a` in row 0, zeros in rows 1-2) - Second column: `[b, c, d]` (non-zero `c` and `d` in rows 1-2) This is explicitly checked in the code modification at step 74 with the condition: ```python if (i == 1 and k == 0 and m == 3 and n == 2 and A[0][0] != 0 and A[1][0] == 0 and 

**ANNOTATION:** STR 干了什么:强制引用证据→逼出具体形态(引了 step 74 的代码条件);DEF 答在错误的抽象层(泛化成 [[a,b],[0,c],[0,d]] 且细节错)。
最优指令:"题目要 exact 值:定位后逐字复制,引步号,禁止泛化。"
表面特征:`what exact|exactly what|the exact`。


==========================================================================================
## Case 4 [F+ / judge-style (content was right)] [OPENWORLD_QA | type B | ep 68]
**Q:** Before extracting the bug fix name at turn 10, what essential document did the agent need to have retrieved?

**GOLD:** The scikit-learn Version 0.19 changelog with the Bug fixes section

**DEF (0):** The agent needed to retrieve the official Scikit-Learn v0.19 (July 2017) changelog documents from https://scikit-learn.org/stable/whats_new/v0.19.html and https://scikit-learn.org/0.19/whats_new.html, which contain the bug fix entry for BaseLabelPropagation.

**STR (1):** The essential document the agent needed to retrieve before extracting the bug fix name at turn 10 was the **Scikit-Learn July 2017 changelog (Version 0.19)** from official sources. **Evidence**: - Step 8: The agent saved content titled *"Information from Scikit-Learn July 2017 changelog via scikit-learn.org"* into `search_results_summary_step1.md`, citing sources like `https://scikit-learn.org/stable/whats_new/v0.19.

**ANNOTATION:** STR 干了什么:内容与 DEF 几乎相同(v0.19 changelog),但"Evidence: 引步号"的风格赢了 judge——**judge-facing 类,题面上无法预判**。
设计含义:把"引证据"作为**所有分支的基础条款**(轻量,不含分解),吃下这类收益。


==========================================================================================
## Case 5 [F- / EMBODIED_AI] [EMBODIED_AI | type D | ep 46]
**Q:** Until step 75, what types of actions has the agent performed and how frequently?

**GOLD:** Action frequency: go (34 times), look (25 times), examine (3 times), help (3 times), open (3 times), close (3 times), take (2 times), move (2 times), inventory (1 times)

**DEF (1):** The agent has performed the following actions: - **go to**: 31 times (moving between locations) - **look**: 25 times (observing surroundings) - **take**: 2 times (taking credit cards) - **move**: 2 times (moving credit cards) - **examine**: 3 times (inspecting items) - **open**: 3 times (opening drawers) - **close**: 3 times (closing drawers) - **inventory**: 1 time (checking inventory) - **help**: 3 times (requestin

**STR (0):** [your answer here], using the context provided. The answer should be based on the steps up to 75, which includes step 75 where "move creditcard 2 to dresser 1" is performed. So the counts are as above. </think> Answer[1]: The agent has performed the following actions up to step 75: - **go to**: 32 times (e.g., steps 1, 3, 5, 11, 12, 14, 15, 20, 21, 25, 28, 30, 31, 33, 36, 40, 41, 45, 48, 49, 51, 52, 58, 59, 61, 62, 6

**ANNOTATION:** STR 干了什么(F−):对 75 步全量直方图强制枚举→列表爆炸+模板回声("[your answer here]...</think>"漏进答案)被判 0;DEF 直接给频次表,对。
教训:枚举规则要分箱——**有界单计数→枚举;长程全量直方图→禁止逐步列举,只给汇总**。
表面特征:`what types ... how frequent|frequency`(无窄界)。


==========================================================================================
## Case 6 [F- / Game] [Game | type A | ep 11]
**Q:** At step 94, the agent observes `coal` for the first time in this sequence, a resource that was not visible at step 90. What specific change to the agent's state, resulting from its five actions between steps 90 and 94, was directly responsible for revealing the `coal` deposit?

**GOLD:** The critical state change was the agent's physical position in the world. By executing the action sequence `Move North`, `Move North`, `Move East`, `Move East`, `Move East`, the agent relocated itself two units north and three units east of its starting position. In Crafter, the agent's observation is a limited window centered on its current location. The `coal` deposit was always present in the world but was outside

**DEF (1):** The agent moved North twice and East twice, repositioning themselves so that the coal deposit fell within their visible range to the south-east.

**STR (0):** The agent's movement from step 90 to step 94 directly caused the coal to become visible. Specifically, the agent executed **three consecutive "Move East" actions** (steps 92, 93, and 94), shifting their position eastward. At step 90, the agent faced "grass at your front" and observed no coal. By step 94, after moving east three times, the agent’s position changed such that "coal 7 steps to your south-east" became vis

**ANNOTATION:** STR 干了什么(F−):强制引证逼它对模糊记忆的动作序列**编造具体步骤**("steps 92,93,94 三次 Move East"——错),DEF 的适度抽象("重新定位→进入视野")就够了。
教训:忠实性条款——"只引你能逐字引用的步;不确定就不带步号地描述"。这条比路由更根本,进基础条款。


==========================================================================================
## Case 7 [F- / OPENWORLD_QA] [OPENWORLD_QA | type B | ep 63]
**Q:** Given turn 5's observation showing 13 articles in the November 2016 log, what action should the agent take to verify Giganotosaurus is about a dinosaur?

**GOLD:** Access the FAC archive page for Giganotosaurus to confirm its subject matter and nominator

**DEF (1):** The agent should review the "Giganotosaurus" section in the November 2016 FAC log (linked in step 4's results) to confirm the article's content. The log entry explicitly mentions it is about a dinosaur (e.g., "Giganotosaurus subsection" with context on its promotion and discussion of its status as a prehistoric theropod). This verifies the article's subject.

**STR (0):** The agent should verify the content of the Giganotosaurus article directly on Wikipedia or through reliable sources cited in the FAC log. The November 2016 log entry (step 5) shows the Giganotosaurus subsection includes support comments (e.g., "Support from Jim," "Support from Cas Liber") but no explicit discussion of the article's content. The FAC log archive does not provide the article's text or specific sources c

**ANNOTATION:** STR 干了什么(F−):建议/判断题("agent 该做什么")被分解带偏——怀疑前提、推荐去 Wikipedia,偏离 gold 的直接动作;DEF 直答对。
最优指令:"直接给出单个最直接的行动/答案,不要展开子问题。"
表面特征:`what should|should the agent|if ... had|would have|most reasonable` —— 反事实/建议类,保护性路由。


==========================================================================================
## Case 8 [F- / SOFTWARE] [SOFTWARE | type D | ep 159]
**Q:** What was the root cause of the pickling failure for draggable legends and how was it identified?

**GOLD:** The root cause was that draggable legends stored references to canvas objects that were not pickleable. This was identified by exploring the draggable implementation in offsetbox.py (steps 10-45), where DraggableBase stored a reference to self.canvas in its __init__ method. The issue was that when set_draggable(True) was called, it created a DraggableLegend instance that held onto the figure's canvas, making the enti

**DEF (1):** The root cause was that the `DraggableLegend` and its parent classes did not implement `__getstate__` to exclude unpicklable attributes like the `canvas` during serialization. This was identified by analyzing the error traceback, inspecting the Matplotlib source code for draggable classes, and confirming that the absence of custom pickling logic caused the canvas reference to be included, leading to a pickling failur

**STR (0):** The root cause of the pickling failure for draggable legends was the presence of unpicklable attributes (like `canvas` and event connection IDs) in the `DraggableBase` class (and its subclasses such as `DraggableLegend`). This was identified through: 1. **Step 73**: The test script `test_edge_cases.py` failed with an assertion error showing the canvas reference became invalid after unpickling, confirming the issue st

**ANNOTATION:** STR 干了什么(F−):root-cause 两段题,强制引证引错了"如何定位"的步(step 73 测试失败 vs gold 的 offsetbox.py 探索)。DEF 停在正确抽象层。
教训:同 Case 6 的忠实性条款;root-cause 类**不路由离开** structured(F+ 里该类净赢),靠基础条款兜底。


==========================================================================================
## Case 9 [F- / TEXT2SQL] [TEXT2SQL | type C | ep 129]
**Q:** Type C2 (Hidden state inference): Across steps 12, 13, 15, 17, and 22 the agent repeatedly hits HTTP read timeouts, yet several other queries succeed (steps 14, 16, 19, 20, 21, 23–26). What hidden system/performance state can be inferred from these indirect signals, and what evidence supports the inference?

**GOLD:** Hidden state inferred: the execution environment/tooling has a strict ~30-second API/network read timeout that is frequently exceeded by heavier multi-join + aggregation queries, while smaller or more selective queries complete under the limit. Evidence synthesized from multiple steps: 1) Multiple timeouts explicitly mention "Read timed out (read timeout=30)" at steps 12, 13, 15, 17, and 22. These are not SQL compila

**DEF (1):** The hidden state indicates that complex queries involving multiple joins (e.g., actor-film-category) or large datasets exceed the system's processing capacity within the 30s timeout, while simpler queries (e.g., direct ID lookups, single-table aggregations) succeed. Evidence includes: - Timeout-prone steps (12,17,22) involve multi-table joins with filters on `CATEGORY`, `LANGUAGE`, and `RATING`, suggesting high compu

**STR (0):** The hidden system/performance state inferred is **query complexity/resource contention causing timeouts for multi-table joins under specific filters**, supported by: 1. **Complex queries time out**: Steps 12, 17, and 22 involve multi-table joins (e.g., `ACTOR`, `FILM_ACTOR`, `FILM`, `FILM_CATEGORY`, `CATEGORY`) with filters on `category_id`, `language_id`, and `release_year`, while simpler queries (e.g., step 14’s `D

**ANNOTATION:** STR 干了什么(F−):综合改写弄丢了关键常量("30 秒超时"),DEF 明确点出。部分是 judge 噪声(两答案质量接近)。
教训:hidden-state 推断题加"把隐藏约束用原文精确串(如错误消息)锚定"——但信号弱,暂不单列,靠基础条款。


==========================================================================================
## Case 10 [F- / WEB] [WEB | type B | ep 192]
**Q:** Why did no customer email get identified?

**GOLD:** The agent never opened a specific customer record after finding the unhappy Zoe review; it only navigated lists and back links.

**DEF (1):** The customer email wasn't identified because the user was navigating the **Reviews** section (under Marketing), which displays product reviews, not customer accounts. Guest reviews (e.g., "Yan," "Valorie") do not store email addresses. Customer emails are accessible in the **Customers > All Customers** section, but the user did not interact with that area after step 27, where emails like "Gamepro456@gmail.com" and "a

**STR (0):** The customer email was not identified because the reviews in question were authored by "Guest" users, and the Magento Admin interface for editing reviews does not display email addresses for such entries. Evidence: - In step 17, the "Edit Review" page shows the author as "Guest" with no email field: `[22738] StaticText 'Guest'` No email-related elements are present in the "Review Details" section. - In step 29, the "

**ANNOTATION:** STR 干了什么(F−):"为什么没发生 X"类——引证偏置让它锚定在**能引用的东西**上(Guest 无邮箱字段),而 gold 是"缺失的动作"(从没打开客户记录);缺席解释无原文可引,被系统性带偏。DEF 直答对。
最优指令:直答模式。表面特征:`why did no|why was ... not|why didn't` —— 负因果题,保护性路由。


==========================================================================================
# 收敛:确定性指令路由规则表(v1,待用户签核)

**基础条款(所有分支共用,替换 default):**
"Answer using ONLY the context above. When you assert a value, step index, or quote, cite the
step it comes from; cite only steps whose content you can quote — if you cannot ground a
detail, state it without inventing specifics."
(来源:Case 4 的 judge-style 收益 + Case 6/8 的反编造忠实条款;不含分解,轻量)

**路由分支(正则可审计,按优先级):**

| # | 触发(regex 族) | 注入指令 | 来源 |
|---|---|---|---|
| R1 保护 | `if .* had / would (it|have) / what should / should the agent / most reasonable / why did no / why (was|were) .* not / why didn.t` | "Answer directly and concisely with the single most direct answer; do not enumerate sub-questions; prefer the simplest explanation consistent with the trajectory." | Case 7/10 + 早前 F− 反事实 |
| R2 单计数 | `how many .* (before|between|until|at step|first|last)` | "First list every matching occurrence with its step number, then give the count of that list." | Case 1 |
| R3 直方图 | `(what types|which actions) .* (how frequent|frequency|how often)` 且无窄界 | "Tally by scanning the whole range; report final counts only — do NOT list every step." | Case 5 |
| R4 状态线 | `how did the state / location histor / state change / throughout the trajectory` | "Reconstruct chronologically: one line per step where the state changes, citing each step; add nothing beyond cited steps." | Case 2 |
| R5 精确值 | `what exact / exactly what / the exact` | "Locate the exact item and copy it VERBATIM, citing its step; do not generalize." | Case 3 |
| R6 默认 | 其余(含 multi-hop/因果/root-cause) | 完整 structured(分解→引证→合成) | F+ 主力类保留 |

优先级 R1 > R2-R5 > R6(反事实里嵌着 how many 时保护条款赢)。
诚实披露:规则由 26 个人工案例归纳(12 F+ 案例研究 + 4 F− 早批 + 本文件 10 例),在 h0 全量(1248)上出样评估;
路由信号与 memory router 同族(确定性题面分析),但按 reader 轴报告——这是指令路由,不是记忆效应。
预注册预期:总分 ≥ RESTR2 0.6418;机制判据 = F− 类破坏显著减少而 F+ 类保留。
