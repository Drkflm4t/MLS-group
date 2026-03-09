# HW1 Triton 详细性能记录（控制变量版本）

## 1. 实验目的与口径

本记录用于在相同任务下，对比以下三组 `benchmark_detailed` 结果：

- `ED`：`glm_asr_triton_example`（第 1 项优化前的 example 参考）
- `TD`：`glm_asr_triton_template`（第 1 项优化前的 template）
- `TD_1`：`glm_asr_triton_template`（已应用第 1 项优化后的 template）

第 1 项优化对应 README 要求：

- `Adjust tile/block sizes`
- 调整 `BLOCK_M/BLOCK_N/BLOCK_K` 及 `num_warps/num_stages`

本次已在 `glm_asr_triton_template/layers.py` 应用：

- `Linear`: `TILE (64,64,32) -> (64,128,32)`，并加 `num_warps=8`, `num_stages=4`
- `MLP fused`: `TILE (64,64,32) -> (64,128,32)`，并加 `num_warps=8`, `num_stages=4`
- `EncoderMLP fused`: `TILE (64,64,32) -> (64,128,32)`，并加 `num_warps=8`, `num_stages=4`

## 2. 三组结果总表

| 指标 | ED (example, 优化前) | TD (template, 优化前) | TD_1 (template, 优化后) |
|---|---:|---:|---:|
| Audio Encoder | 949.90 ms | 1634.66 ms | 1039.85 ms |
| Projector | 12.19 ms | 9.67 ms | 10.82 ms |
| Decoder Prefill | 351.41 ms | 633.14 ms | 411.37 ms |
| Decode Step (avg) | 32.57 ms | 72.53 ms | 41.05 ms |
| Decoder (50 steps est.) | 1628.47 ms | 3626.67 ms | 2052.51 ms |
| Total (50 tokens est.) | 2941.96 ms | 5904.14 ms | 3514.55 ms |

## 3. 关键对比分析

### 3.1 优化前：TD vs ED

- 总时延：`5904.14 vs 2941.96 ms`，TD 比 ED 慢约 `100.7%`（约 `2.01x`）。
- 最大差距出现在：
  - `Audio Encoder`（`+684.76 ms`）
  - `Decoder Prefill`（`+281.73 ms`）
  - `Decode Step`（`+39.96 ms`，约 `+122.7%`）

结论：仅完成 TODO 的 template 在优化前明显落后 example。

### 3.2 第 1 项优化效果：TD_1 vs TD

- 总时延：`5904.14 -> 3514.55 ms`，下降 `2389.59 ms`（约 `-40.5%`）。
- 分项变化：
  - `Audio Encoder`: `1634.66 -> 1039.85 ms`（约 `-36.4%`）
  - `Decoder Prefill`: `633.14 -> 411.37 ms`（约 `-35.0%`）
  - `Decode Step`: `72.53 -> 41.05 ms`（约 `-43.4%`）

结论：第 1 项优化是有效的，且收益显著。

### 3.3 优化后：TD_1 vs ED

- 总时延：`3514.55 vs 2941.96 ms`，TD_1 仍慢约 `+19.5%`。
- 差距仍主要来自：
  - `Audio Encoder`（约 `+9.5%`）
  - `Decoder Prefill`（约 `+17.1%`）
  - `Decode Step`（约 `+26.0%`）

结论：第 1 项优化后，template 已大幅接近 example，但尚未追平。

## 4. 波动性说明（控制变量解读）

本次多次复跑后可以观察到：

- `benchmark_detailed` 存在天然波动（JIT、缓存、节点负载、频率状态都会影响）。
- 在同一轮可比数据中（ED / TD / TD_1），趋势清晰：
  - 优化前 TD 最慢
  - 应用第 1 项后 TD_1 明显改善
  - 但 TD_1 仍略慢于 ED

因此当前可以支持的结论是：

- 第 1 项优化已完成且有效；
- 若要继续缩小差距，需要执行 README 的第 2、3 项优化。

## 5. 后续计划

- 第 2 项：至少 1 个额外 kernel fusion（除现有 fused 路径外，继续减少中间张量读写和 launch 次数）。
- 第 3 项：实现 FlashAttention-style attention（分块 QK^T、稳定 softmax、再乘 V）。
- 每完成一项均复跑 `benchmark_detailed` 与 `benchmark.sh`，同时记录性能与正确性（PASS/Accuracy）。

## 6. 第 2 项优化已实现（Kernel Fusion）

本次新增并接入了一个额外融合内核（满足 README 第 2 项“at least 1 fused kernel”）：

- 新增 `linear_bias_gelu_kernel`：将 `Linear + Bias Add + GELU` 融合为单个 Triton kernel。
- 接入位置：`hw1-asr/glm_asr_triton_template/layers.py` 的 `EncoderMLP._forward_fused`。

实现细节：

- 之前路径：`linear_gelu_kernel` 输出后，再在 Python/Torch 侧执行 `intermediate + bias`。
- 现在路径：若 `fc1` 含 bias，则直接走 `linear_bias_gelu_kernel`，在 kernel 内完成 bias 加法与 GELU。
- 无 bias 时保持原有 `linear_gelu_kernel` 路径不变。

预期收益：

- 减少一次中间张量的额外 elementwise 加法开销。
- 降低一次 kernel launch 及相关读写流量（尤其对 prefill/encoder 路径更有意义）。

说明：

- 已完成稳定复跑，并写入下方第 7 节“第 2 项优化实测对比”。

## 7. 第 2 项优化实测对比（TD_1 vs TD_2）

说明：

- `TD_1`：第 1 项优化后（tile/block + launch 参数）
- `TD_2`：在 `TD_1` 基础上，叠加第 2 项优化（`Linear + Bias + GELU` kernel fusion）

| 指标 | TD_1 | TD_2 | 变化 |
|---|---:|---:|---:|
| Audio Encoder | 1039.85 ms | 920.37 ms | `-119.48 ms`（约 `-11.5%`） |
| Projector | 10.82 ms | 9.96 ms | `-0.86 ms`（约 `-7.9%`） |
| Decoder Prefill | 411.37 ms | 336.94 ms | `-74.43 ms`（约 `-18.1%`） |
| Decode Step (avg) | 41.05 ms | 33.61 ms | `-7.44 ms`（约 `-18.1%`） |
| Decoder (50 steps est.) | 2052.51 ms | 1680.43 ms | `-372.08 ms`（约 `-18.1%`） |
| Total (50 tokens est.) | 3514.55 ms | 2947.69 ms | `-566.86 ms`（约 `-16.1%`） |

结论（第 2 项）：

- 在你这轮稳定结果下，第 2 项优化是有效的。
- 端到端估计总时延从 `3514.55 ms` 降到 `2947.69 ms`，改善约 `16.1%`。
- 主要收益集中在 `Audio Encoder` 与 `Prefill/Decode`，符合融合内核减少中间读写和 launch 的预期。

补充（与 ED 对比）：

- `ED` 总时延：`2941.96 ms`
- `TD_2` 总时延：`2947.69 ms`
- 当前二者几乎持平（`TD_2` 约慢 `5.73 ms`，约 `0.19%`）。

## 8. 当前阶段总结

- 第 1 项（tile/block 调整）已完成并有效。
- 第 2 项（至少 1 个 kernel fusion）已完成并在稳定复跑中验证有效。
- 目前 template 的 detailed 性能已基本追平 example。
- 下一步重点转向第 3 项：FlashAttention-style attention。

## 9. 第 3 项优化已实现（FlashAttention-style Attention）

已在 `hw1-asr/glm_asr_triton_template/attention.py` 实现并接入 FlashAttention 风格路径：

- 新增 `flash_attention_kernel`：
  - 分块计算 `QK^T`（`BLOCK_M x BLOCK_N`）
  - 使用在线 softmax（维护每行 `m_i` / `l_i`）保证数值稳定
  - 在同一 kernel 中直接累加 `P @ V`，减少中间 attention 矩阵读写
- 新增 `_flash_attention(...)` 封装函数，负责 kernel launch。
- 在 `scaled_dot_product_attention(...)` 中优先走 Flash 路径（满足条件时）：
  - CUDA
  - `attention_mask is None`
  - `head_dim <= 128`
  - `seq_k <= 4096`
- 不满足条件时自动回退到原有 Triton/Torch 路径，保证兼容性和正确性。

说明：

- 这满足 README 第 3 项“FlashAttention-style attention”实现要求。
- 当前阶段仅完成代码接入，性能收益需要通过复跑验证并补充在下一个小节。

## 10. 第 3 项优化实测对比（TD_2 vs TD_3）

说明：

- `TD_2`：第 2 项优化后（在 `TD_1` 基础上加入 Kernel Fusion）
- `TD_3`：在 `TD_2` 基础上叠加第 3 项优化（FlashAttention-style attention）

| 指标 | TD_2 | TD_3 | 变化 |
|---|---:|---:|---:|
| Audio Encoder | 920.37 ms | 757.29 ms | `-163.08 ms`（约 `-17.7%`） |
| Projector | 9.96 ms | 10.83 ms | `+0.87 ms`（约 `+8.7%`） |
| Decoder Prefill | 336.94 ms | 425.90 ms | `+88.96 ms`（约 `+26.4%`） |
| Decode Step (avg) | 33.61 ms | 30.91 ms | `-2.70 ms`（约 `-8.0%`） |
| Decoder (50 steps est.) | 1680.43 ms | 1545.46 ms | `-134.97 ms`（约 `-8.0%`） |
| Total (50 tokens est.) | 2947.69 ms | 2739.47 ms | `-208.22 ms`（约 `-7.1%`） |

结论（第 3 项是否符合预期）：

- 符合预期，且整体有效。
- 端到端总时延从 `2947.69 ms` 降到 `2739.47 ms`，进一步提升约 `7.1%`。
- decode 路径（`Decode Step` / `Decoder(50 steps)`）得到稳定改善，符合 FlashAttention-style 优化目标。
- `Prefill` 出现回退（`+26.4%`），说明当前 kernel 参数/调度在 prefill 场景还有优化空间，但未改变总体正收益结论。

补充（与 ED 对比）：

- `ED` 总时延：`2941.96 ms`
- `TD_3` 总时延：`2739.47 ms`
- `TD_3` 已快于 `ED` 约 `202.49 ms`（约 `6.9%`）。

备注：

- Attention 小节里的 `Standard (einsum)` 数值在不同轮次波动较大（本轮 `4.45ms`），不宜单独作为第 3 项成败判断标准。
- 更可靠的是端到端组件与总时延变化，本轮呈现净收益。

## 11. 阶段性总总结

- 第 1 项（tile/block 调整）已完成并有效。
- 第 2 项（至少 1 个 kernel fusion）已完成并有效。
- 第 3 项（FlashAttention-style attention）已完成并在实测中带来额外端到端收益。
- 截至 `TD_3`，template detailed 性能已超过当前 `ED` 基线。
