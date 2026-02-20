# ICT-OpenCity 开发 Todo List (v2)

基于三份文档（OpenCity 代码分析、ICT 代码分析、ICT-OpenCity 集成方案）及一致性分析 / 验证机制审查后的修订版。

**核心改进**：
1. 每个 Phase 末尾内嵌**验证门（Gate）**，写完即测，阻止错误积累
2. 修复 `num_prefix_selections` 设计缺陷：dataloader 提供 `[B, S, K, T, N, F]`，S 组独立 demo
3. `forward_ict` 内置 runtime assertions
4. TP 缓存优化直接内嵌，不再作为 TODO
5. 回归测试 (K=0) 从 Phase 7 前移至 Phase 3

---

## Phase 0: 环境准备与 Baseline 验证

- [ ] **0.1** 确认预训练权重存在：`model_weights/OpenCity/OpenCity-plus.pth`
- [ ] **0.2** 运行 zero-shot baseline，得到参考指标
  ```bash
  # 编辑 conf/general_conf/pretrain.conf:
  #   dataset_use = ['CAD3']
  #   val_ratio = 0.1, test_ratio = 0.4   ← baseline 模式仍需手动配置
  python main.py -mode test -model OpenCity -load_pretrain_path OpenCity-plus.pth \
      -batch_size 2 --embed_dim 512 --skip_dim 512 --enc_depth 6
  ```
- [ ] **0.3** 运行 Fast Adaptation baseline（CD_DIDI），记录对比指标
- [ ] **0.4** 确认所有 8 个评估数据集可正常加载（逐一 `load_st_dataset` 无报错）

**Phase 0 Gate**: 所有命令正常运行，baseline MAE/RMSE/MAPE 已记录。

---

## Phase 1: 配置系统扩展 (~20 min)

- [ ] **1.1** 修改 `lib/Params_pretrain.py`：新增 ICT 参数
  ```python
  parser.add_argument('-num_demonstrations', default=1, type=int,
                      help='Number of demonstration pairs (K) for ICT')
  parser.add_argument('-num_prefix_selections', default=1, type=int,
                      help='Number of independent demo sets (S) to average at test time')
  parser.add_argument('-demo_selection', default='random', type=str,
                      help='Demo selection strategy: random, recent, similar')
  ```
- [ ] **1.2** 创建 `conf/ICT/ICT.conf`
  ```ini
  [ict]
  num_demonstrations = 1
  num_prefix_selections = 1
  demo_selection = random
  ict_batch_size = 32
  ```

**Phase 1 Gate**:
```bash
cd model && python -c "
import sys; sys.path.append('../lib')
from Params_pretrain import get_config
args = get_config()
assert hasattr(args, 'num_demonstrations'), 'missing num_demonstrations'
assert hasattr(args, 'num_prefix_selections'), 'missing num_prefix_selections'
assert hasattr(args, 'demo_selection'), 'missing demo_selection'
print('Phase 1 Gate: PASS')
"
```

---

## Phase 2: ICT 数据加载器 (~2 h)

- [ ] **2.1** 创建 `lib/ict_data_process.py`，添加必要 import
- [ ] **2.2** 实现 `ICTTrafficDataset(Dataset)` 类
  - `__init__` 参数：`data, batch_size, input_window, output_window, demo_pool, num_demonstrations, num_prefix_selections, eval_only`
  - `__getitem__` 返回 `(batch_x, batch_y, demos_x, demos_y)`
  - **Shape 契约**：
    - `batch_x`: `[B, T, N, F]`
    - `demos_x`: `[B, S, K, T, N, F]`（S = num_prefix_selections）
  - Demo 采样：对 batch 中每个 query，独立采样 S 组 × K 个 demo（均来自 training split）
  - 遵循原始 `TrafficDataset` 的内部 batch 模式
- [ ] **2.3** 创建 `conf/general_conf/dataset_splits.conf`
  - 统一入口：每个数据集一个 INI section，包含 `val_ratio` 和 `test_ratio`
  - `[default]` section 提供 fallback（0.1, 0.4）
  - 需要手动调整分割比例时只改这一个文件
- [ ] **2.4** 实现 `load_dataset_splits()` + `get_dataset_split()` 辅助函数
  - 从 `conf/general_conf/dataset_splits.conf` 读取分割比例
  - 未知数据集 fallback 到 `[default]` section
- [ ] **2.5** 实现 `define_ict_dataloader(args)` 函数
  - 复用 `load_st_dataset()` + `split_data_by_ratio()` 流程
  - **调用 `get_dataset_split()` 自动解析每个数据集的分割比例**
  - 训练集 demo_pool → Val/Test 共享（防止泄漏）
  - Train/Val 的 `num_prefix_selections=1`，Test 使用 `args.num_prefix_selections`
  - 返回 `(train_loader, val_loader, test_loader, scaler_dict)`
- [ ] **2.6** 修改 `lib/data_process.py` — 现有 `define_dataloder()` 也读取 `dataset_splits.conf`
  - `from lib.ict_data_process import load_dataset_splits, get_dataset_split`
  - 替换 `args.val_ratio` / `args.test_ratio` 为 `get_dataset_split(dataset_name, splits, default_split)`
  - **覆盖所有模式**（test / eval / ori / pretrain），统一分割入口

**Phase 2 Gate**:
```python
# test_phase2.py — 独立运行，不依赖后续 Phase
from lib.ict_data_process import ICTTrafficDataset
import numpy as np, torch

# Mock data: T=1000, N=10, F=3
data = np.random.randn(1000, 10, 3).astype(np.float32)
iw, ow = 288, 288
demo_pool = [(data[i:i+iw], data[i+iw:i+iw+ow]) for i in range(len(data)-iw-ow+1)]
ds = ICTTrafficDataset(data, batch_size=4, input_window=iw, output_window=ow,
                        demo_pool=demo_pool, num_demonstrations=2,
                        num_prefix_selections=3, eval_only=True)
bx, by, dx, dy = ds[0]
assert bx.shape == (4, 288, 10, 3), f"batch_x shape: {bx.shape}"
assert dx.shape == (4, 3, 2, 288, 10, 3), f"demos_x shape: {dx.shape}"  # [B, S, K, T, N, F]
assert dy.shape == dx.shape, f"demos_y shape mismatch"
print(f"Phase 2 Gate: PASS — batch_x {bx.shape}, demos_x {dx.shape}")
```

---

## Phase 3: OpenCity 模型 — `forward_ict` 方法 (~2 h)

- [ ] **3.1** 在 `model/OpenCity/OpenCity.py` 的 `OpenCity` 类中新增 `forward_ict()` 方法
- [ ] **3.2** Query 处理路径（复用 `forward()` 逻辑）
  - Temporal context encoding → Spatial PE → Instance Normalization → Patch embedding → `[B, 24, N, D]`
- [ ] **3.3** Demo 处理循环（K 个 demo）
  - 各自独立：temporal encoding、instance norm（基于自身 history）、patch embedding
  - Demo hist `[B, 24, N, D]` + Demo future `[B, 24, N, D]` → concat `[B, 48, N, D]`
  - **缓存 `dk_TP`** 到 `all_demo_TP` 列表（避免重复计算）
- [ ] **3.4** 序列拼接 + Runtime Assertions
  ```python
  enc_all = cat([demo1_enc(48), ..., demoK_enc(48), query_enc(24)], dim=1)
  assert enc_all.shape[1] == K * 48 + 24, f"enc_all T-dim: expected {K*48+24}, got {enc_all.shape[1]}"
  # TP_all 使用缓存的 all_demo_TP（无冗余 patch_embedding_time 调用）
  ```
- [ ] **3.5** Encoder blocks → 提取 query patches → 预测头 → De-IN
  ```python
  query_out = enc_all[:, -24:, :, :]
  assert query_out.shape[1] == 24
  ```
- [ ] **3.6** K=0 回归测试 (**前移，不等到 Phase 7**)
  - `forward_ict` 中处理 K=0 fallback：当 `K == 0` 时退化为 `forward` 逻辑
- [ ] **3.7** 确保 `forward()` 原方法完全不被修改

**Phase 3 Gate**:
```python
# test_phase3.py — mock tensor 直接测 forward_ict，不依赖 dataloader
import torch, sys
sys.path.append('../lib')
# ... load model with pretrained weights ...

B, T, N, F, K = 2, 288, 10, 3, 1
D = 512  # OpenCity-plus embed_dim
mock_input = torch.randn(B, T, N, F).cuda()
mock_lbls = torch.randn(B, T, N, F).cuda()
mock_dx = torch.randn(B, K, T, N, F).cuda()
mock_dy = torch.randn(B, K, T, N, F).cuda()

# Test forward_ict
out_ict = model.predictor.forward_ict(mock_input, mock_lbls, mock_dx, mock_dy, dataset_name)
assert out_ict.shape == (B, T, N, 1), f"ICT output shape: {out_ict.shape}"

# Test K=0 regression: forward_ict with 0 demos ≡ forward
mock_dx0 = torch.randn(B, 0, T, N, F).cuda()
mock_dy0 = torch.randn(B, 0, T, N, F).cuda()
out_ict0 = model.predictor.forward_ict(mock_input, mock_lbls, mock_dx0, mock_dy0, dataset_name)
out_orig = model.predictor(mock_input, mock_lbls, dataset_name)
diff = (out_ict0 - out_orig).abs().max().item()
assert diff < 1e-5, f"K=0 regression failed: max diff = {diff}"

print(f"Phase 3 Gate: PASS — ICT output {out_ict.shape}, K=0 diff={diff:.2e}")
```

---

## Phase 4: 模型封装层修改 (~15 min)

- [ ] **4.1** 修改 `model/Model.py` — `Traffic_model.forward()`
  - 新增 `demos_x=None, demos_y=None` 可选参数
  - `demos_x is not None and model == 'OpenCity'` → `self.predictor.forward_ict(...)`
  - 否则 → 原逻辑不变

**Phase 4 Gate**:
```python
# 无 demo 时输出与修改前完全一致
out_no_demo = model(mock_input, mock_lbls, dataset_name)
out_with_none = model(mock_input, mock_lbls, dataset_name, demos_x=None, demos_y=None)
diff = (out_no_demo - out_with_none).abs().max().item()
assert diff == 0.0, f"Backward compat broken: diff = {diff}"
print("Phase 4 Gate: PASS — backward compatible")
```

---

## Phase 5: ICT 测试方法 (~1 h)

- [ ] **5.1** 在 `model/BasicTrainer.py` 新增 `test_ict()` 静态方法
  - 加载权重 → 冻结参数 → `model.eval()` + `torch.no_grad()`
- [ ] **5.2** 推理循环：正确处理 `[B, S, K, T, N, F]` 的 demo 维度
  ```python
  demos_x = demos_x.squeeze(0)  # [B, S, K, T, N, F]
  S = demos_x.shape[1]
  outputs = []
  for s in range(S):
      output_s = model(inputs, targets, select_dataset,
                       demos_x=demos_x[:, s],    # [B, K, T, N, F]
                       demos_y=demos_y[:, s])
      outputs.append(output_s)
  output = torch.stack(outputs).mean(dim=0)
  ```
  **关键修复**：每个 `s` 使用**不同的 K 个 demo**（在 dataloader 采样时已确定），确保多次 forward 产生不同输出。
- [ ] **5.3** 指标计算：复用 `All_Metrics()`，逆标准化 `scaler.inverse_transform`

**Phase 5 Gate**:
```python
# 用 1 个 batch mock data 跑 test_ict，确认指标计算无异常
# 构造 minimal dataloader with 2 batches
# 调用 Trainer.test_ict(model, args, scaler_dict, mock_loader, logger)
# 检查 logger 输出包含 "ICT Test — MAE:" 字段
print("Phase 5 Gate: PASS — test_ict runs without error")
```

---

## Phase 6: 模式调度集成 (~30 min)

- [ ] **6.1** 修改 `model/Run.py`：新增 `mode='ict'` 分支
  - `from lib.ict_data_process import define_ict_dataloader`
  - 加载预训练权重，冻结参数
  - `define_ict_dataloader(args)` → `test_dataloader_ict`
  - `trainer.test_ict(...)` 执行推理
- [ ] **6.2** 确保不影响现有 mode（pretrain / ori / eval / test）

**Phase 6 Gate — 端到端 Smoke Test**:
```bash
# 编辑 pretrain.conf: dataset_use = ['CAD3']
# （所有模式自动从 dataset_splits.conf 解析 val_ratio=0.1, test_ratio=0.4）
python main.py -mode ict -model OpenCity \
    -load_pretrain_path OpenCity-plus.pth -num_demonstrations 1 \
    -batch_size 2 --embed_dim 512 --skip_dim 512 --enc_depth 6
# 应输出 "ICT Test — MAE: x.xxxx, RMSE: x.xxxx, MAPE: x.xxxx"
```

**同时验证原有模式也自动解析分割**:
```bash
# 无需手动设置 val_ratio/test_ratio，同样从 dataset_splits.conf 自动解析
python main.py -mode test -model OpenCity \
    -load_pretrain_path OpenCity-plus.pth -batch_size 2 \
    --embed_dim 512 --skip_dim 512 --enc_depth 6
# 指标应与 Phase 0.2 记录的 baseline 完全一致
```

---

## Phase 7: 全面集成测试 (~2 h)

- [ ] **7.1** 多 interval 兼容性测试
  - 5min: `CAD3` (N=480) — 已在 smoke test 验证
  - 10min: `CD_DIDI` (N=524) — 验证 gap=2 patch padding
  - 30min: `CHI_TAXI` (N=77) — 验证 gap=6 patch padding
- [ ] **7.2** GCN 兼容性验证：`einsum('bdkt,nk->bdnt')` 在 T=K×48+24 时无报错（已在 Phase 3 Gate 隐式覆盖，此处显式确认跨数据集）
- [ ] **7.3** `geo_mask` 兼容性验证：确认 geo_mask 仅用于 spatial 维度，不受 T 维度扩展影响
- [ ] **7.4** K=3 测试：`-num_demonstrations 3 -batch_size 2`，确认无 OOM
- [ ] **7.5** S=3 方差降低验证：`-num_prefix_selections 3`
  - 确认 S 次 forward 结果**不同**（因使用不同 demo 组）
  - 确认最终 output 是 S 次预测的平均
- [ ] **7.6** GPU 显存估算
  - K=1, batch_size=32 vs baseline batch_size=64
  - K=3, 找到可用 batch_size

---

## Phase 8: 实验运行与对比 (~4 h)

- [ ] **8.1** Zero-shot baseline（`mode=test`）全 8 数据集
- [ ] **8.2** Fast Adaptation baseline（`mode=eval`, 3 epochs）全 8 数据集
- [ ] **8.3** ICT K=1（`mode=ict, -num_demonstrations 1`）全 8 数据集
- [ ] **8.4** ICT K=3（`mode=ict, -num_demonstrations 3`）全 8 数据集
- [ ] **8.5** ICT K=1 × S=10（`-num_demonstrations 1 -num_prefix_selections 10`）全 8 数据集
- [ ] **8.6** 整理结果对比表（MAE / RMSE / MAPE）

---

## Phase 9: 高级特性与分析 (~4 h)

- [ ] **9.1** Demo 选择策略：`recent`（时间最近）、`similar`（DTW/余弦相似度）、`same_time`（同 time-of-day）
- [ ] **9.2** Attention 可视化：提取 TC Cross-Attention 权重，分析 query → demo future 注意力
- [ ] **9.3** 消融实验
  - Demo future flow → zeros（验证 future 信息价值）
  - Demo 来自不同数据集（cross-domain 鲁棒性）
  - K = 1, 2, 3, 5 系统对比

---

## 文件变更总览

| 文件 | 变更类型 | Phase |
|------|----------|-------|
| `lib/Params_pretrain.py` | 修改 | 1 |
| `conf/ICT/ICT.conf` | **新建** | 1 |
| `conf/general_conf/dataset_splits.conf` | **新建** | 2 |
| `lib/ict_data_process.py` | **新建** | 2 |
| `lib/data_process.py` | 修改（`define_dataloder` 读取 `dataset_splits.conf`） | 2 |
| `model/OpenCity/OpenCity.py` | 修改（新增 `forward_ict`） | 3 |
| `model/Model.py` | 修改 | 4 |
| `model/BasicTrainer.py` | 修改（新增 `test_ict`） | 5 |
| `model/Run.py` | 修改 | 6 |

---

## 接口契约

| 接口 | Provider → Consumer | Shape | 验证点 |
|------|---------------------|-------|--------|
| Dataloader → test_ict | Phase 2 → Phase 5 | `demos_x: [B, S, K, T, N, F]` | Phase 2 Gate |
| test_ict → Model.forward | Phase 5 → Phase 4 | `demos_x: [B, K, T, N, F]` (per-selection) | Phase 5 内循环 |
| Model.forward → forward_ict | Phase 4 → Phase 3 | `demos_x: [B, K, T, N, F]` | Phase 3 内 assertions |
| forward_ict 内部 | Phase 3 | `enc_all: [B, K*48+24, N, D]` | Phase 3 内 `assert` |
| forward_ict → output | Phase 3 | `[B, T, N, 1]` | Phase 3 Gate |

---

## 依赖关系（含验证门）

```
Phase 0 (环境验证 + Baseline) ──── Gate ✓
    │
    ├──→ Phase 1 (配置扩展) ──── Gate ✓
    │
    ├──→ Phase 2 (ICT DataLoader) ──── Gate ✓  ←── mock data shape test
    │         │
    └──→ Phase 3 (forward_ict) ──── Gate ✓  ←── mock tensor + K=0 regression
              │
              ▼
         Phase 4 (Model.py) ──── Gate ✓  ←── backward compat diff < 1e-6
              │
              ▼
         Phase 5 (test_ict) ──── Gate ✓  ←── 1-batch smoke test
              │
              ▼
         Phase 6 (Run.py) ──── Gate ✓  ←── 端到端 + 原 mode 不变
              │
              ▼
         Phase 7 (全面集成测试)
              │
              ▼
         Phase 8 (实验运行)
              │
              ▼
         Phase 9 (高级分析)
```

**MVP（Phase 1-6 + Gates）：~6.5 小时**（含验证时间）
**完整交付（Phase 0-9）：~16 小时**
