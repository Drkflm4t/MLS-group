# 后续工作 TODO（针对第 3 项优化后的验证与优化）

## A. 先做稳定性验证（必须）

- [ ] 在同一 `srun` 会话内，固定环境变量后，连续运行 3 次：
	- `./benchmark_detailed.sh glm_asr_triton_template`
- [ ] 丢弃第 1 次（冷启动/JIT 影响），统计后 2 次均值，记录：
	- `Audio Encoder`
	- `Decoder Prefill`
	- `Decode Step (avg)`
	- `Total (estimated for 50 tokens)`
- [ ] 同一会话内再跑 2 次 `ED`（example）作为参照，确认 `TD_3` 是否持续快于 `ED`。
- [ ] 跑正确性回归：
	- `./benchmark.sh glm_asr_triton_template`
	- 确认 `Accuracy` 和 `Status: PASS`。

## B. Prefill 回退专项优化（重点）

- [ ] 调整 FlashAttention kernel 配置并做小网格搜索：
	- `BLOCK_M`：`32 / 64`
	- `BLOCK_N`：`32 / 64 / 128`
	- `num_warps`：`4 / 8`
	- `num_stages`：`2 / 3 / 4`
- [ ] 分离评估 prefill 和 decode：优先选择“prefill 不回退且总时延最低”的配置。
- [ ] 检查是否因为回退路径触发导致 prefill 变慢（例如 mask 场景未走 Flash 路径）。
- [ ] 若 prefill 仍明显回退，增加策略开关：
	- prefill 场景按阈值选择原 Triton 路径
	- decode 场景优先走 Flash 路径

## C. 验收标准（完成条件）

- [ ] `TD_3` 在稳定复跑（后 2 次均值）下，总时延优于 `TD_2`。
- [ ] `Decoder Prefill` 相比当前结果不再显著回退（目标：接近或优于 `TD_2`）。
- [ ] 端到端正确性保持 `PASS`，转写结果与基线一致。

## D. 文档同步

- [ ] 在 `RECORD.md` 新增“稳定复跑均值表（TD_2 vs TD_3）”。
- [ ] 在 `RECORD.md` 说明最终采用的 Flash 参数组合和选择理由。
- [ ] 在最终报告中注明：
	- 第 3 项已实现 FlashAttention-style
	- 若存在 prefill/decode trade-off，说明工程取舍依据。
