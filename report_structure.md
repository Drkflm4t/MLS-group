# 报告重写大纲（基于原始模板 report_local.tex）

## 0. 适用范围

- 本文档用于从零重写报告。
- 以 report_local.tex 的原始模板要求为准。
- 忽略旧版 report.tex 的现有结构与表述。

## 1. 总体写作原则

- 严格使用模板规定的 7 个主章节标题。
- 每节只承担模板定义的职责，避免跨节重复。
- 结果口径必须明确：
  - benchmark_detailed：组件级 + 50 token 估算。
  - benchmark.sh：真实端到端（本次样本为 13 tokens）。

## 2. 章节分工（最终执行版）

### Section 1: Introduction
- Motivation：解释为什么做 GPU kernel 优化、GLM-ASR 的系统挑战。
- System Overview：给出完整 workflow 图，并简述最终系统和主要优化路径。
- 这里可放一句总结果，但不展开所有数据表。

### Section 2: Implementation（重点）
- 本节只写“最终实现版本”是什么，不写完整试错过程。
- 使用 workflow 的 five phase 组织正文。
- 你已确认：five phase 对应 workflow 第 2-6 层。

建议 five phase 如下：
1. Mel Spectrogram (128 bins)
2. Conv Subsampler (4x downsample)
3. Audio Encoder (32 layers)
4. Projector (pool 4 frames, MLP)
5. Text Decoder (28 layers)

每个 phase 固定写法：
- 该 phase 做什么计算。
- 关键 kernel/算子是什么。
- 主要访存/算力特征（compute-leaning 或 memory-leaning）。

Design Choices 子节只放最终配置：
- Linear kernel: BLOCK_M/BLOCK_N/BLOCK_K
- num_warps, num_stages
- 数据布局处理
- GQA 映射策略

### Section 3: Performance Profiling
- 写“怎么测”和“测到了什么”。
- 必须包含：
  - Setup（硬件、软件、命令、输入、runs/warmup）
  - End-to-end Evaluation（建议放 stage 结果：example/TD_1/TD_2/TD_3）
  - Component/Operator-Level Breakdown（TD 分解）
- 该节是证据展示，不做太多根因推理。

### Section 4: Bottleneck Analysis
- 只做归因：compute-bound vs memory-bound + overall bottleneck。
- 基于 Section 3 数据解释“为什么是这个瓶颈”。
- 可讨论非 kernel 开销（launch、同步、框架调度）。
- 不再重复贴完整比较大表。

### Section 5: Optimization Attempts
- 严格按 Hypothesis -> Change -> Result 写三项优化。
- 建议顺序：
  1. Tile/Block Size Tuning
  2. Kernel Fusion
  3. FlashAttention-Style Attention
- 这里可以写“优化2组件级收益但端到端近乎持平”的口径解释。

### Section 6: Comparison
- 这是“最终比较结论节”。
- 必须回答：最终系统相对 example baseline 提升多少。
- 推荐结构：
  - Data Collection（数据与可比性）
  - End-to-End Comparison（最终对比主表）
  - Per-Operator Comparison（关键差异）
  - Root Cause Analysis（根因总结）

### Section 7: Conclusion
- 总结最有效优化、经验与局限、未来工作。
- 保持简洁，和前文结论一致。


## 3. 组内协作建议（按板块拆分）

- 成员 A：Section 2（Implementation）
- 成员 B：Section 3 + Section 4（Profiling + Bottleneck）
- 成员 C：Section 5 + Section 6（Optimization + Comparison）
- 统一负责人：Section 1 + Section 7 收口，统一口径与术语

## 4. 提交前检查清单

- [ ] 7 个主章节与模板完全一致。
- [ ] Section 2 只描述最终实现版本。
- [ ] five phase 明确对应 workflow 第 2-6 层。
- [ ] benchmark_detailed 与 benchmark.sh 口径在文中明确区分。
- [ ] Section 3 与 Section 6 无明显重复叙事。
- [ ] 所有关键数字在全文前后一致。
- [ ] 图表标题、引用、编号无冲突。
