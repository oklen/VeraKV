# VeraKV

**中文** | [English](README.en.md)

VeraKV 是供长时间运行的 agent 使用的记忆层（memory），保存原始执行轨迹。默认配置在构建和查询记忆时都不调用模型：先固定选入问题指定的步骤（step），其余步骤按词重合度排序，再加入最近几步。答题模型（reader）收到每个步骤的简短开头，以及选中步骤的原文，中位数约 10k token，上限 22k。选中的步骤既可以作为文本提供，也可以提取其 KV cache，保留原位置后提供给模型。

<img src="assets/demo-case-zh.gif" alt="同一道真实 AMA-Bench 题，两个系统并排演示，时钟显示实际秒数。为一条 150 步轨迹构建记忆时，AMA-Agent 并发发出 14 次模型调用，每次为约 8,192 字符生成状态摘要，分别耗时 35 到 73 秒，日志记录的总耗时为 76 秒。VeraKV 用每个步骤的开头几行建立索引，不调用模型，在笔记本 CPU 上耗时 3.5 ms。两边构建时间为 76 s 和 3.5 ms，相差约 20,000 倍。问题是 drawer 1 在 step 77 的状态：VeraKV 调用一次模型，82 秒后答出 open，并指出它在 step 17 变为 open，与参考答案一致。AMA-Agent 检索到 6 个步骤，其中没有打开抽屉的步骤，但它判断信息已足够，102 秒后答出 closed。关闭提前作答后，AMA-Agent 调用模型 3 次、耗时 311 秒，答案仍为 closed。" width="100%">

<sub>演示轨迹共 150 步。构建记忆时，AMA-Agent 耗时 76 秒、调用模型 14 次；VeraKV 耗时 3.5 ms、不调用模型，两者相差约 20,000 倍。两项测量使用的机器不同：AMA-Agent 的数据来自运行日志，环境为 Qwen3-32B、vLLM 和一台 8×A100，14 次调用并发执行，每次处理约 8,192 字符、耗时 35 到 73 秒，每条轨迹只构建一次记忆；VeraKV 的数据是本仓库代码在 Apple M3 Pro 笔记本 CPU 上运行 9 次的中位数（`figures/time_index_build.py`）。本题 VeraKV 调用一次模型，82 秒答对；AMA-Agent 判断信息足够后，102 秒答错；关闭提前作答后，调用 3 次、耗时 311 秒，仍然答错。在 2,136 道配对题上，每题耗时中位数分别为 VeraKV 54 秒、AMA-Agent 71 秒、关闭提前作答的 AMA-Agent 207 秒。实际耗时受各次运行负载影响，这里只比较数量级。VeraKV 用每个步骤的开头几行作为索引，按词重合度和步骤编号选择内容。演示选自 AMA-Bench 的 alfworld 第 34 集，属于 VeraKV 答对、AMA-Agent 答错的 333 道题之一；反过来的情况有 252 道。除 3.5 ms 外，调用次数、耗时和答案均来自[配对重跑](docs/AMA_AGENT_PAIRED.md)日志，答案仅显示前两句。下图按段展示同一日志中两个答题模型实际收到的提示词。两张图由 `figures/make_demo.py` 和 `figures/make_context.py` 根据 `figures/demo_case.json` 生成。</sub>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/context-case-zh-dark.png">
  <img src="assets/context-case-zh.png" alt="本题两个答题模型收到的提示词，按段摘自实际日志。AMA-Agent 收到路由角色说明、问题、检索到的 6 个轮次（4、6、9、12、16、77，不含打开 drawer 1 的 step 5 和 17）、关于这些轮次只是按相似度选出的一小部分的说明、信息足够就作答否则继续索取的决策流程，以及输出格式。VeraKV 收到 step 0 到 145 的逐步概览，每步一行，包含 step 5 和 17 的 open drawer 1，表明抽屉已打开；还收到最后 4 步的全文、选中的 step 21、67、77、136、141 的全文、问题，以及直接简洁作答的指令。" width="100%">
</picture>

AMA-Agent 会让模型判断已检索的内容是否足够作答。在 84% 的题目中，模型读过检索到的 5 个步骤后就停止检索。VeraKV 不依赖模型判断该舍弃哪些内容，在配对测试中，准确率比 AMA-Agent 高 4.2 个百分点，状态更新题高 13.7 个百分点。

关闭 AMA-Agent 的提前作答（early answer）后，VeraKV 的准确率低约 2 个百分点，每题耗时约为它的三分之一，且构建记忆不调用模型。使用自行部署的模型时，还可以直接提供选中步骤的 KV cache。在窗口内的实验中，准确率与文本方式持平；证据长度增至 4k token 时，首个 token 延迟（TTFT）的加速比达到 10.9 倍。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/how-it-works-zh-dark.png">
  <img src="assets/how-it-works-zh.png" alt="多数 agent 记忆系统在写入时用模型抽取事实，作答时再用模型判断信息是否足够。VeraKV 原样保存每个步骤，按步骤编号、词重合度和最近步骤选择内容，不调用模型；答题模型收到各步骤的简短开头和选中步骤的全文。选中的步骤也可以通过 KV cache 提供。" width="100%">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/results-zh-dark.png">
  <img src="assets/results-zh.png" alt="左：2026-07-15 的 AMA-Bench 准确率，VeraKV 为 0.6478，高于 HybridFocus 的 0.6246、AMA-Agent 的 0.5722 和另外五个系统。右：相同推理服务、相同评分模型下，VeraKV 相对 AMA-Agent 的总体准确率差值为 +4.2 个百分点，状态更新题为 +13.7 个百分点；相对关闭提前作答的 AMA-Agent 为 −2.0 个百分点，每题耗时为 63 秒和 207 秒。" width="100%">
</picture>

- `0.6478`：AMA-Bench 得分，提交时（2026-07-15）排名第 1。
- `+4.2`：在相同推理服务和评分模型下，相对 AMA-Agent 的准确率提升，单位为百分点。
- `10.9×`：使用 KV cache 后，TTFT 的加速比。
- `76 s vs 3.5 ms`：为一条 150 步轨迹构建记忆的耗时；AMA-Agent 调用模型 14 次，VeraKV 不调用模型。

[论文（PDF）](paper/kvmemory.pdf) · [复现](#复现) · [与 AMA-Agent 的对比说明](docs/AMA_AGENT_PAIRED.md)<br>Agent memory 系列：[VeraKV](https://github.com/oklen/VeraKV) 讨论该把什么交给模型；[Compute Globally, Materialize Locally](https://github.com/oklen/Compute-Globally-Materialize-Locally) 讨论数据被驱逐后还保留什么；[TracePack](https://github.com/oklen/TracePack) 讨论会话压缩后还保留什么。

论文 [`paper/kvmemory.pdf`](paper/kvmemory.pdf)（*When Does KV Reuse Work for Agent Memory? A Controlled Study of Payload Fidelity, Store Conditioning, and Cache Harvesting — with a Reproducible State-of-the-Art System*，源文件 [`paper/kvmemory.tex`](paper/kvmemory.tex)）中各项数字对应的代码、配置、提示词、原始预测和评分模型输出，均保存在本仓库。

## 主要发现

- 在这组实验中，词重合匹配加问题指定的步骤编号已经足够。让模型选择步骤没有带来明确提升，修复原先未正常工作的调用后也是如此：+0.2 [−1.5, +2.1] 和 +1.4 [−0.2, +3.2]，见[详情](docs/MODEL_PICK_FIX.md)。
- 提供给答题模型的内容量影响较大。步骤选择方法不变时，将拼接后的上下文从 6k 增加到 22k token，准确率从 0.550 提高到 0.644，这项测试覆盖一半题目。若直接提供原始轨迹最近的 22k token，准确率则下降 8.4 个百分点。
- AMA-Agent 在 84% 的题目中，第一次检查后就判断信息足够并作答。关闭提前作答后，它的准确率比 VeraKV 高约 2 个百分点，但每题耗时约为三倍。
- 上下文窗口内，按原位置提取选中步骤的 KV cache，可以复现「完整 prefill 后加 mask」的 oracle，答题准确率与重新读取这些步骤的文本持平（17.3% vs 17.4%），收益主要体现在延迟。超出窗口后，各事件独立编码会使准确率下降 13.2 个百分点。

## 测试结果

| AMA-Bench 官方测试框架，全部 2,496 题 | 准确率 |
|---|---|
| **VeraKV**：经公开榜单核验，2026-07-15 时排第 1 | **0.6478** |
| HybridFocus（榜单条目） | 0.6246 |
| VeraKV，仅使用记忆层（测试框架默认的作答配置） | 0.5954 |
| AMA-Agent（为该基准测试设计） | 0.5722 |
| 长上下文 Qwen3-32B，不使用记忆层 | 0.52 |
| MemoRAG | 0.4606 |
| HippoRAG2 | 0.4480 |
| MemGPT | 0.3304 |
| Mem0 | 0.2104 |

VeraKV 的结果由我们运行得到，其他系统使用已发表结果或公开榜单分数。

- 与 AMA-Agent 公布的分数相比，VeraKV 在 6 个领域中的 5 个准确率更高。
- 保持测试框架默认的作答配置、不改动 reader 侧，仅替换记忆层，得分为 0.5954，高于 AMA-Agent 公布的 0.5722。
- 在相同推理服务和评分模型下，对 2,136 道题进行配对测试，VeraKV 比原版 AMA-Agent 高 4.2 个百分点 [+1.8, +6.6]，状态更新题高 13.7 个百分点 [+8.7, +18.6]。主要差异来自 AMA-Agent 在首次检查后提前作答。关闭提前作答后，AMA-Agent 的准确率高约 2 个百分点：当前对比配置的 VeraKV 相对它为 −2.0 [−4.4, +0.4]；发布的默认配置在记忆构建和检索中均不调用模型，相对差值为 −2.3 [−4.6, −0.1]。代价是每题耗时约为 VeraKV 的 3 倍，生成 token 约为 7 倍。见[详情](docs/AMA_AGENT_PAIRED.md)。
- 按 LOCOMO 公开的 gpt-4o-mini 协议，VeraKV 的 J = 0.704，高于对话记忆系统 Mem0 的 0.671 和 Zep 的 0.660。
- 直接提取选中片段的 KV cache 时，TTFT 稳定在约 40 ms。随着证据长度增加到 4k token，相对重新读取文本的加速比从 1.6 倍增至 10.9 倍；在 32k 时测得 32 倍。生成的首个 token 与完整 prefill 的 oracle 完全相同。

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/ttft-zh-dark.png">
  <img src="assets/ttft-zh.png" alt="Qwen3-8B 上，TTFT 随所选证据长度的变化。将选中步骤作为文本重新 prefill 时，TTFT 从 575 token 时的 66 ms 增至 4,387 token 时的 448 ms；直接使用 KV cache 则保持在约 41 ms。896 道题上的准确率为 KV 17.3%、文本 17.4%。" width="80%">
</picture>

2026-09-09，榜单上得分为 0.6975 的条目超过 VeraKV，VeraKV 随后排第 2。在 gpt-4o-mini 评分协议下，直接将 LOCOMO 完整历史放入提示词可得 0.756。这些结果的比较条件及其他限制，见论文和[测量上的说明](#测量上的说明)。

## 实验分析

### 原文、摘要与抽取事实

多数记忆系统都保存原始历史，区别在于最终提供给答题模型的内容。Mem0 的搜索返回抽取的事实；AMA-Agent 发布的代码会把按嵌入相似度检索的 top-5 原始轮次与模型生成的状态摘要一起提供，但在 84% 的题目中，它在读取摘要前就已根据那几个轮次作答，见[配对重跑](docs/AMA_AGENT_PAIRED.md)。

在引用的证据、答题模型和评分模型相同时，用简短的模型摘要或抽取事实替换原始步骤，准确率下降 8 到 10 个百分点，精确召回题下降 14 个百分点。这些替代内容也短得多，因此实验测到的是信息遗漏造成的损失，无法单独归因于改写。

在部署流程中，按确定性规则替换为与问题相关的原始行，准确率下降 2.6 个百分点；替换为摘要和事实，分别下降 5.1 和 5.9 个百分点。AMA-Bench 的 needle 对照也显示了类似损失：将含答案的轮次换成 MemoryBank 构建的记忆，BabyAI 准确率从 0.46 降至 0.27，相对下降 41%。

### 单独调整作答配置

同批次的 memory × reader 因子实验显示，无论 reader 使用哪种作答配置，步骤选择都带来 3.1 至 3.5 个百分点的提升；无论采用哪种步骤选择方法，一条结构化作答指令都带来 5.1 至 5.5 个百分点的提升。这条指令可以随记忆返回值一起提供；编码为 rotated skill-KV block 后，效果与文本指令持平。

### 窗口内的 KV 提取

在原位置提取选中片段的 KV cache，可以复现「完整 prefill 后加 mask」的 oracle。896 道题上的准确率与将相同片段作为文本重新 prefill 持平，分别为 17.3% 和 17.4%。在这组测量中，TTFT 不再随证据长度增长。

### 超出窗口后的存储方式

在官方测试框架中，各事件独立编码的 event store 准确率明显下降，配对差值为 −13.2 个百分点，n=2431；anchored store 加上两条新近的 tail 事件，可以弥补其中 72% 的损失。实验支持两条有适用范围的观察：证据需要联合编码才能改善准确率；写入时的条件信息（conditioning），必须与读取时提供的信息在内容和位置上对应。

六种在新预算内选择内容的方法，包括一种 KEEP 风格的方法，均未超过按最近时间选择的 recency 方法。复用 agent 自身执行过程中产生的缓存（harvest），可以在效果持平时将 store 写入开销降低约 5 倍；实验覆盖 8B/32B、Qwen/Llama、AMA-Bench/LOCOMO。一种混合部署策略在完整官方测试框架中，与 routed-text carrier baseline 持平（.397 vs .397）。

### 系统配置与 SOFTWARE 题目

仅使用记忆层、保持测试框架默认作答配置时，得分为 0.5954，AMA-Agent 公布的分数为 0.5722。加入公开说明的结构化作答指令后，系统达到经核验的 0.6478。

唯一没有领先的领域是 SOFTWARE，主要差距来自 temporal-scan 题，例如「哪一步第一次或最后一次……」，而非检索或推理。所有配置在这类题上的得分都较低，且不少参考答案与轨迹不符。

## 目录结构

```
paper/      kvmemory.tex + .bib + 样式文件 + 编译好的 PDF（30 页；tectonic 或 pdflatex）
kvmemory/   memory 包（core、components、routers、带 KV sub-selection 的 HF backend），以及每个实验一个
            模块，都能用 `python -m kvmemory.<name>` 运行（用法见各自的 docstring）
ama/        AMA-Bench 集成：harness/（2 个打过 patch 的文件）、configs/、agentic_reader.py（reader 模式
            ablation）、作答 instruction、maxutil_run.sh（vLLM 启动 + 分片 + 合并）、
            br_judge.py / bg_judge.py（用官方 judge 给 KV-bridge 的答案打分）、
            paired/（和 AMA-Agent 的 paired re-run；运行方法见 paired/README.md）
docs/       AMA_AGENT_PAIRED.md：和 AMA-Agent 的 paired re-run；MODEL_PICK_FIX.md：修好 model-pick 调用后的重跑
analysis/   把运行输出变成论文数字的脚本（见下表）
results/    逐题原始预测 + judge 分数（mu_merged_<TAG>.json，运行清单 mu_<TAG>_done）、
            results/oow/（KV serving 和 out-of-window 运行）、results/probes/（合成 probe）、
            results/ama_paired/（和 AMA-Agent 的 paired re-run）、results/modelpick_fix/（model-pick 修好后的
            重跑）、以及较小的输出；索引在 results/MANIFEST.md
scripts/    run_ama.sh：在一台 8×A100 上端到端跑官方 harness
figures/    make_figures.py：根据上面这些文件画 assets/ 里的 README 图（浅色/深色，英文/中文各一套）；
            make_demo.py 和 make_context.py：单题 demo 动图和「两边 reader 各看到什么」的图，数据来自
            demo_case.json（由 build_demo_case.py 生成）；time_index_build.py：测 VeraKV 为一集轨迹构建 memory 的耗时
submissions/ leaderboard 提交文件
data/       放 benchmark 数据的位置（不随仓库发布；见 data/README.md）
```

## 环境

实验使用两套环境，具体版本见 [ENVIRONMENT.md](ENVIRONMENT.md)：

- AMA-Bench 官方测试框架：文本流程，用于所有榜单口径的结果；通过 vLLM 部署 Qwen3-32B，同时作为答题模型和评分模型。
- KV 实验（`kvmemory/kv_*.py`）：使用 HuggingFace transformers，每张 GPU 一份模型；模型包括 Qwen3-8B、Qwen3-32B（用 `SPRAG_DEVICE_MAP=auto` 切分）和 Llama-3.1-8B-Instruct。

各模块使用以下环境变量：`SPRAG_MODEL_PATH` 指定本地模型目录，必填；`SPRAG_EMBED_PATH` 指向 Qwen3-Embedding-0.6B，供 embedding / hybrid router 使用；`SPRAG_ATTN_IMPL=sdpa` 用于 store 实验所需的 block-diagonal mask，attention probe 使用 `eager`；`SPRAG_ROPE_FACTOR=4.0` + `SPRAG_MAX_CTX=131072` 用于启用 YaRN 的窗口外实验。输出默认写入 `./out/`。

## 复现

### AMA-Bench 官方测试框架

使用 Qwen3-32B 作为答题模型和评分模型，需要一台 8×A100-80GB、[AMA-Bench](https://github.com/AMA-Bench/AMA-Bench)、vLLM 和 Qwen3-32B 权重。

```bash
git clone https://github.com/AMA-Bench/AMA-Bench
cp ama/harness/method_kvmemory.py  AMA-Bench/src/method/kvmemory.py      # 并在 src/method_register.py 里注册 "kvmemory"
cp ama/harness/memory_interface.py AMA-Bench/src/memory_interface.py     # 由环境变量控制的作答 instruction 和 reader hook
export AMA_BENCH=$PWD/AMA-Bench MODEL=/path/to/Qwen3-32B VENV=/path/to/vllm_env
bash scripts/run_ama.sh cfg_kvmem_causal.json MYTAG structured   # lexical + pin 的 memory，structured reader -> results/mu_merged_MYTAG.json
bash scripts/run_ama.sh cfg_kvmem_causal.json MYTAG_DEF plain     # 默认 reader（只用 memory）那一格
python analysis/cluster_ci.py                                     # 题目级和按 episode 聚类的 CI
```

`ama/maxutil_run.sh` 中说明了答题模式开关（`AMA_AGENTIC_READER`、`AMA_AGENTIC_MODE`），以及打乱固定选入步骤的消融实验（`SPRAG_PIN_SHUFFLE`）。提交分数 0.6478 使用 `cfg_flagship.json`，配置为 lexical + model-pick fusion + pin。论文附录「The deployed router」显示，这种融合未影响分数；发布的默认配置为 `cfg_kvmem_causal.json`，使用 lexical + step-pin。

官方测试框架中的 model-pick 调用启用了 Qwen3 thinking 模式，只返回了推理开头。`cfg_flagship_nothink.json`（`"pick_thinking": false`）修复了这个调用，修复后的信号仍未带来明确的分数变化，见 [docs/MODEL_PICK_FIX.md](docs/MODEL_PICK_FIX.md)。

### KV 与窗口外实验

使用 HF transformers。将 AMA-Bench 测试文件放入 `data/ama_test.jsonl`，见 [data/README.md](data/README.md)。各模块的 docstring 中都有运行命令，例如：

```bash
SPRAG_MODEL_PATH=/path/to/Qwen3-8B SPRAG_ATTN_IMPL=sdpa PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
    python -m kvmemory.kv_globhot --shard 0 --queue_dir ./out/gh_q
```

机制子集包含 103 集窗口内轨迹，长度 ≤24k token，按领域轮流抽取，共 n=824 题。运行时使用 8 路工作队列（`--queue_dir`），每张 GPU 一个进程。官方测试框架的 bridge 先通过 `kvmemory.kv_ama_bridge` 生成答案，使用 Qwen3-32B 和 YaRN 4×，再交给 AMA-Bench 自带的评分模型：`ama/br_judge.py` 用于 tx / iso / b_hot，`ama/bg_judge.py` 用于 deployment map。

### 用已发布输出计算配对统计

```bash
python analysis/oow_paired.py results/oow/official_k12.jsonl.gz iso-tx b_hot-tx     # 表 "oow"，K=12
python analysis/oow_paired.py results/oow/gh.jsonl.gz glob_hotR-bh glob_R-glob     # harvest 那一格
python analysis/oow_paired.py results/oow/rv.jsonl.gz --arms                        # 列出所有 arm
```

`oow_paired.py` 可以逐位复现准确率、不一致配对数（discordant pairs）和精确 McNemar p 值。按 episode 聚类的 bootstrap 使用 10,000 次采样、seed 0，所得置信区间可能与论文相差最后一位。

## 实验与复现脚本（本次发布新增结果）

| 论文位置 | 内容 | 脚本 | 输出 / 分析 |
|---|---|---|---|
| §3.4, Tab. "family" | 将附录内容替换为原始行 / 事实；完整轨迹按 recency 截断 | `ama/agentic_reader.py`（`extlines`、`facts`）、`ama/configs/cfg_full22.json` | `results/mu_merged_{EXTLW,FACTSW,FULLTR}.json` |
| §3.4, Fig. "payload" | oracle evidence 下的内容消融（verbatim / summary / facts，910 题），并用 Llama-3.1-8B 重新评分 | `kvmemory/kv_payload.py`（vLLM client）、`kvmemory/kv_payload_hf.py`、`kvmemory/cross_judge.py` | — |
| App. D | 记忆拼装 2×2（selection × gist overview），按领域统计 | `ama/configs/cfg_arm_{recent,gist,retrieval}.json`、`cfg_kvmem_lex.json` | `results/mu_merged_ASM_*.json`、`analysis/assembly_2x2.py` |
| §3.1, App. E | SOFTWARE temporal-scan 题：所有配置得分均较低，参考答案存在噪声（8B 和 32B 答题模型） | `kvmemory/sw_mechanism.py` | `results/sw_mechanism/` |
| App. F | 部署路由审计：同批复现，lexical+pin 0.6466 vs model-pick hybrid+pin 0.6538，配对差值 −0.8 个百分点 [−2.3, +0.8]；升级推理服务栈后重跑 model-pick，得分 0.6410（未配对） | `ama/configs/cfg_kvmem_causal.json`、`cfg_flagship.json` | `results/mu_merged_RESTRLEXF.json`（lexical+pin）、`mu_merged_RESTRF2.json`（model-pick+pin，同批）、`mu_merged_RESTRF4.json`（model-pick+pin，后续运行）；flagship 的几次复现 `RESTRF`、`RESTRF2`、`RESTRF3` |
| App. G | 删除步骤编号 / 仅固定选入步骤时的稳健性，并用 Llama-3.1-8B 重新评分 | `kvmemory/kv_stepablate.py`、`kvmemory/cross_judge.py` | — |
| §3.5, App. H | 合成 mask-oracle 检验，logit gap 为 0.7 vs 13.5；TTFT 随证据长度的变化 | `kvmemory/kv_select_smoke.py` | — |
| §3.5, App. H | 序列级保真度（faithfulness）、logit-KL、eager/SDPA、YaRN 4× | `kvmemory/kv_faithful.py` | — |
| §3.5, App. H | KV 与文本端到端效果持平：144 题和 896 题（17.3% vs 17.4%） | `kvmemory/kv_equiv.py` | `results/kv_equiv_144/`、`results/kv_equiv_896/` |
| §3.5 | 扰动未选中的上游内容（提取的 KV 中有 111/168 发生变化，文本为 0） | `kvmemory/kv_sidechannel.py` | `results/kv_sidechannel/` |
| §3.7 | 存储开销、所选 token 增至 64k 时的 TTFT，以及一次运行节省的 prefill | `kvmemory/kv_efficiency.py`、`kvmemory/bench_kv.py` | — |
| §3.7 | 完整轨迹常驻的 prefix-cache 基线 | `kvmemory/kv_pfx.py` | `results/kv_cost/pfx_8b.json`、`analysis/pfx_ttft.py` |
| §3.6 | 将作答规程编码为 rotated skill-KV block | `kvmemory/kv_skill.py` | `results/oow/sk.jsonl.gz` |
| §3.6 | 按问题路由的 skill 库（替换 vs 追加；文本 vs KV） | `kvmemory/kv_skillib.py` | `results/oow/sl.jsonl.gz` |
| §4, Tab. "oow" | 官方测试框架：tx / iso / anchored+tail，K=12 和 K=5 | `kvmemory/kv_ama_bridge.py`、`ama/br_judge.py` | `results/oow/official_k{12,5}.jsonl.gz` |
| §4 | 超窗口实验（30 至 100k，YaRN），8B 和 32B | `kvmemory/kv_owfloor.py` | `results/oow/owf{8,32}.jsonl.gz` |
| §4 | 分阶段的查询开销与写入开销 | `kvmemory/kv_ttft.py` | `results/kv_cost/ttft_{8b,32b}.jsonl`、`analysis/ttft_stages.py` |
| §4 | 证据匹配（tx_plus）、conditioner 内容、完整阅读的上限、tail dial（8B、32B） | `kvmemory/kv_review.py` | `results/oow/rv{,32}.jsonl.gz` |
| §4 | 生成 digest 作为 conditioner，以及其他 floor 实验组 | `kvmemory/kv_floor.py` | `results/oow/fl.jsonl.gz` |
| §4 | harvest 2×2（提取行、开头行、新近 tail）；Llama-3.1-8B 复现 | `kvmemory/kv_globhot.py` | `results/oow/gh{,_llama}.jsonl.gz` |
| §4 | agent 风格的 cache header；32B 上对比 harvest、store 和完整阅读 | `kvmemory/kv_hxfer.py` | `results/oow/hx{,32}.jsonl.gz` |
| §4 | LOCOMO，将会话视为事件（8B、32B；harvest 和 skill 实验组） | `kvmemory/kv_locomo.py` | `results/oow/loc{8,32}.jsonl.gz` |
| §4 | 分布层面的分析（first-token KL） | `kvmemory/kv_klprobe.py` | `results/oow/kl.jsonl.gz`、`analysis/kl_probe.py` |
| §4 | 回退触发条件（mode gate、self-verification、attention landing、voting） | `kvmemory/kv_gate.py`、`kv_verify.py`、`kv_attn.py`、`kv_matrix.py` | `results/oow/mx.jsonl.gz`、`analysis/{vote_analysis,cascade_ceiling}.py` |
| §4 Reg. I | 各载体的 K=5 vs K=12；两条路径增加相同数量的事件 | `kvmemory/kv_ama_bridge.py`、`kvmemory/kv_evext.py` | `results/oow/official_k*.jsonl.gz`、`results/oow/ev.jsonl.gz` |
| §4 Reg. II | note-transplant 测试（delimiter vs 随机 anchor 行） | `kvmemory/kv_notesel.py` | `results/oow/nt.jsonl.gz` |
| §4 Reg. II | 冻结的全局 conditioner / conditioner 的几何结构 | `kvmemory/kv_fixanchor.py`、`kvmemory/kv_distanchor.py` | `results/oow/{fx,dx}.jsonl.gz` |
| §4 Reg. II | EPIC 风格的边界重算；dependency-graph conditioner | `kvmemory/kv_epic.py`、`kvmemory/kv_depanchor.py` | `results/oow/{ep,da}.jsonl.gz` |
| §4 | 六种 fresh-budget 选择方法（recency、graph、attention、query、random；KEEP 风格） | `kvmemory/kv_frontier.py`、`kvmemory/kv_keepsel.py` | `results/oow/{fr,kp}.jsonl.gz` |
| §4 | 官方测试框架中的 deployment map（gsk vs anchored store vs 文本） | `kvmemory/kv_ama_bridge.py --arms gsk,g_txt`、`ama/bg_judge.py` | `results/oow/official_deploymap.jsonl.gz` |
| §4 probe 1 至 3 | phantom conclusion、erasure 边界、sink vs dependency | `kvmemory/kv_phantom.py`（+ `kv_vartrack.py`） | `results/probes/phvt/`、`analysis/probes/phvt_analyze.py` |
| §4 probe 4 | 四实验组分解；不完整的路由 | `kvmemory/kv_vardecomp.py`、`kv_predigest.py`、`kv_predigest_sweep.py` | `results/probes/{vd,pd,pds}/`、`analysis/probes/` |
| §4 probe 4 | 独立 decoy、menu readout、entropy | `kvmemory/kv_decoyctl.py`、`kv_decoyctl2.py`、`kv_sweepmenu.py` | `results/probes/{dc,d2,sm}/` |
| §4 probe 4 | computed verdict、predicate binding、写入时的笔记 | `kvmemory/kv_mater.py`、`kv_refcarrier.py`、`kv_noteknob2.py` | `results/probes/{mt,rc,nk2}/`、`analysis/probes/nk2_analyze.py` |
| §3.2 | LOCOMO，Qwen3-32B 答题与评分模型；公开的 gpt-4o-mini 协议 | `kvmemory/locomo_eval.py`、`kvmemory/locomo_gpt4omini.py` | — |
| §3.3 | LongMemEval-S（128k YaRN 读取全文 vs 28k 路由内容） | `kvmemory/longmemeval_eval.py` | — |
| App. I | 用 Llama-3.1-70B 对一次完整官方运行重新评分 | `analysis/rejudge_llama70b.py`、`ama/configs/judge_llama70b.yaml` | `results/judge_llama70b/ama_rejudged.json` |
| 论文之外 | 与 AMA-Agent 配对重跑：原版（A1）、按 token 计的上下文上限（A2）、关闭提前作答（A3）；VeraKV V1 / V2（[说明](docs/AMA_AGENT_PAIRED.md)） | `ama/paired/run.py`、`make_tokcap.py`、`make_nofast.py` | `results/ama_paired/`、`ama/paired/analyze.py`、`a3_analyze.py` |
| 论文之外 | 修复 model-pick 调用后重跑：两种作答模式下的 V1 / V1f / V2，2,496 题（[说明](docs/MODEL_PICK_FIX.md)） | `ama/paired/run.py`（实验组 V1f）、`ama/configs/cfg_flagship_nothink.json` | `results/modelpick_fix/`、`ama/paired/modelpick_analyze.py` |

共享模块 `kv_scope`、`kv_replay`、`kv_write`、`kv_matrix`、`kv_ow`、`kv_floor`、`kv_globhot`、`kv_distanchor` 和 `kv_orphan` 提供窗口外实验使用的 store 编码、拼装和 KV 提取原语。每个模块也可以独立运行其最初对应的实验。四个合成 probe 是后续论文 [Compute Globally, Materialize Locally](https://github.com/oklen/Compute-Globally-Materialize-Locally) 的起点，扩展版本见该仓库。`gpt-4o-mini` 实验需要兼容 OpenAI 接口的服务端点（`LOCOMO_API_BASE_URL`、`LOCOMO_API_KEY`）；使用本地答题模型运行 LOCOMO / LongMemEval 需要 vLLM 服务。

## 测量上的说明

- AMA-Bench 只公开测试集，榜单配置在该测试集上开发，因此论文将 0.6478 标为探索性的工程结果。窗口外方案先在 103 集的机制子集上开发，再在完整官方测试集上评估一次。
- 同日重跑的波动约为 0.5 个百分点，跨日约为 2 个百分点，SOFTWARE 的批次波动为 ±4 个百分点。报告中的配对差值均来自同一批次；唯一的跨批次读数已注明，为 App. F 中的 0.6466 vs 0.6410。
- §4 的 serving 实验采用固定 K 检索、不提供概览、使用官方默认作答提示词，与部署流程的协议不同。其中的文本实验组（.397）是 routed-text carrier baseline，比主系统低约 25 个百分点。两者不能跨协议直接比较，见论文附录「Protocol ledger」。
- KV 提取只在位置窗口内具有精确性，所有主要准确率均来自文本流程。缓存路径的结论只涉及 TTFT，不能据此推断整题作答更快。
- 结果表和其后前两条要点使用 AMA-Agent 公布的分数。我们重跑其发布版本时，在其覆盖的 5 个领域上低于公布分数 4.9 个百分点，SOFTWARE 上低 26 个百分点，原因尚未找到。配对比较使用两个系统各自的重跑结果，见[复现核对](docs/AMA_AGENT_PAIRED.md#reproduction-checks)。
- 部署路由中的 model-pick 调用限制为 32 个输出 token。官方测试框架未关闭 Qwen3-32B 的 thinking 模式，导致我们两次重跑中的每次调用都在推理尚未结束时停止，未按设计工作。论文的路由审计测到的也是这些截断调用，未发现该信号对分数有影响。修复调用后，在全部 2,496 题上同批重跑，结果仍未显示明确提升：默认作答模式 +0.2 [−1.5, +2.1]，结构化作答模式 +1.4 [−0.2, +3.2]，见[详情](docs/MODEL_PICK_FIX.md)。

## 许可

代码以 MIT 许可发布，见 `LICENSE`。基准测试和数据集保留各自许可，数据文件不随仓库发布，见 `data/README.md`。逐题结果文件中引用的 AMA-Bench 题目和参考答案用于核查分数。

## 联系

Zefeng Cai，见论文首页。
