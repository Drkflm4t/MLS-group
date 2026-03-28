## 2 IMPLEMENTATION
2.1 
是否每个阶段都兼顾到了（a）（b）
【核对结果】已处理。
- report\_local 第2节已对五个 phase 分别补充了 memory-access pattern 与 compute/memory 倾向说明。
- 对应位置：Implementation Summary 段落（约 L101-L125）。
2.2 
Element-wise kernels.部分有解释“What BLOCK SIZE did you choose for element-wise kernels?”吗
【核对结果】已处理。
- 已改为与实现一致的表述：BLOCK\_SIZE 不是统一常数，而是按算子选择。
- 例如：gelu/silu/embedding 使用 BLOCK\_SIZE=256；norm/softmax 在 Triton 路径下使用 next\_power\_of\_two(feature or seq len)。
- 对应位置：Design Choices / Element-wise kernels（约 L147-L152）。

## 3
3.1
Two scripts are used with distinct roles部分有问题，写出测试时是runs5 取均值即可
【核对结果】已处理。
- Setup 里已改成：stage-level 分析统一使用 benchmark\_detailed runs=5 均值。
- benchmark.sh 改为仅用于最终端到端对比与正确性（PASS/accuracy）。
- 对应位置：Performance Profiling / Setup（约 L191-L198）。

3.3
结尾“ Net effect is still positive at stage level because large gains and small regressions occur in
different components”意义不明，意向删除（解释）
【核对结果】已处理。
- 该句已删除，3.3 的 Key observations 仅保留前两条。
- 对应位置：Component/Operator-Level Breakdown（约 L267-L271）。

## 4
4.2
1. decoder是否满足模板要求的Which kernels
【核对结果】已处理。
- 已明确 decoder-side repeated computation 是主要 wall-clock bottleneck，并给出 Prefill + 50*DecodeStep 定量计算。
- 对应位置：Overall Bottleneck（约 L297-L306）。

2. “Consider non-kernel overheads: launch latency, Python overhead, host-device synchronization, memory allocation vs. in-place reuse.”有分析到吗，检查一下
【核对结果】已处理。
- 已从“列项”升级为“具体分析”：分别解释了 launch latency、Python/host orchestration、synchronization、allocation vs in-place reuse 在解码迭代与注意力路径中的出现方式及其对端到端的影响。
- 同时标注证据口径为 system-level evidence（代码路径+benchmark行为），不是独立 profiler counter 的直接测量。
- 对应位置：Overall Bottleneck 结尾（约 L312-L315）。

3. 检查逻辑连贯，我感觉是不是根据第三部分看出第四部分的bottleneck，然后第五部分针对优化。这里是不是没提到第五部分优化的需求来源（不照应）
【核对结果】已处理。
- 已在第4节末增加第5节优化目标映射语句，形成 3->4->5 的因果衔接。
- 对应位置：Overall Bottleneck 末尾（约 L317-L319）。

## 5
5.1
Hypothesis我觉得应该把每个实验的期望都作为，这样能显示出我们做每组实验的意义。
【核对结果】已处理。
- 已在 5.1 增加 C1-C5 的逐项 Expected effects（parallelism/locality、occupancy/register pressure、pipeline/shared-memory pressure 等）。
- 对应位置：Tile/Block Size Tuning（约 L338-L350）。

Result: Which config won? Why? (occupancy, register pressure, shared memory)，这里面的指标我们的实验好像没有，是少测东西了吗还是你没写上（需要我再次补传一次detail结果吗）
【核对结果】已处理（按工程解释口径）。
- 已在 Result 明确：缺少 profiler counter 时，occupancy/register/shared-memory 的解释属于 engineering explanations，不做强因果断言。
- 对应位置：Tile/Block Size Tuning / Result（约 L374-L378）。

5.2 & 5.3
各自的If implemented, describe in detail要求好像都没写进去
【核对结果】已处理。
- 5.2 已补 before vs after（分离 kernel 与融合 kernel 的对照，强调 intermediate writes/round-trips 变化）。
- 5.3 现已显式回答模板三问：
	1) 与 naive 3-kernel 的差异（QK^T -> softmax -> AV 的全量中间矩阵物化 vs 流式分块融合）；
	2) 数值稳定性（online softmax / log-sum-exp 风格的 running max/sum 更新）；
	3) Q/K/V 的序列维分块策略（BLOCK_M 查询块 + BLOCK_N 键值块迭代）。
- 对应位置：Kernel Fusion Change（约 L398-L404）；FlashAttention Change（约 L433-L436）。

5.4
可以删除这一节因为没有OTHER OPTIMIZATIONS
【核对结果】已处理。
- 5.4 小节已删除；FlashAttention 小节后直接进入 Section 6 Comparison。
- 对应位置：FlashAttention 结束后转入 Comparison（约 L448）。

---

## 同步结论

本文件中列出的修改点，现已全部同步到 report\_local。


## 仍需check
4.2