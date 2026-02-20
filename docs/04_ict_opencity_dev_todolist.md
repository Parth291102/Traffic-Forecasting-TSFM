# ICT-OpenCity 开发 Todo List

基于三份文档（OpenCity 代码分析、ICT 代码分析、ICT-OpenCity 集成方案）整理的开发任务清单。

---

## Phase 0: 环境准备与代码熟悉

- [ ] **0.1** 确认 OpenCity 基线可运行：用现有 `mode=test` 跑一次 zero-shot 推理，得到 baseline 指标
  - `python main.py -mode test -model OpenCity -dataset_use "['CAD3']" -load_pretrain_path OpenCity-plus.pth`
- [ ] **0.2** 确认 `mode=eval`（Fast Adaptation）可运行，记录 baseline 指标作为对比
- [ ] **0.3** 确认所有 8 个评估数据集可正常加载（CAD3, CAD5, PEMS07M, TrafficSH, CHI_TAXI, NYC_BIKE-3, CD_DIDI, SZ_DIDI）
- [ ] **0.4** 确认预训练权重文件 `model_weights/OpenCity/OpenCity-plus.pth` 存在且可加载

---

## Phase 1: 配置系统扩展 (P1, ~20 min)

- [ ] **1.1** 修改 `lib/Params_pretrain.py`：新增 ICT 相关命令行参数
  - `-num_demonstrations`（默认 1）
  - `-num_prefix_selections`（默认 1）
  - `-demo_selection`（默认 `random`）
- [ ] **1.2** 创建 `conf/ICT/ICT.conf`：ICT 默认配置文件
  ```ini
  [ict]
  num_demonstrations = 1
  num_prefix_selections = 1
  demo_selection = random
  ict_batch_size = 32
  ```

---

## Phase 2: ICT 数据加载器 (P0, ~2 h)

- [ ] **2.1** 创建 `lib/ict_data_process.py`
- [ ] **2.2** 实现 `ICTTrafficDataset` 类
  - 继承 `Dataset`，在 `__getitem__` 中返回 `(batch_x, batch_y, demos_x, demos_y)`
  - `demos_x/demos_y` shape: `[B, K, T, N, F]`
  - 内部维护 `demo_pool`（来自 training split 的滑窗列表）
  - 每次 `__getitem__` 随机从 `demo_pool` 采样 K 个 demo
  - 遵循原始 `TrafficDataset` 的 batch 内聚模式（内部 batch → 外层 DataLoader `batch_size=1`）
- [ ] **2.3** 实现 `define_ict_dataloader()` 函数
  - 复用 `load_st_dataset()` 与 `split_data_by_ratio()` 等原有工具函数
  - 为每个数据集构建 demo_pool（从 training split 的滑窗中采样）
  - Val/Test 的 demo_pool 必须来自 training split（防止数据泄露）
  - 返回 `(train_loader, val_loader, test_loader, scaler_dict)`
- [ ] **2.4** 添加 `ict_data_process.py` 必要的 import（`load_st_dataset`, `StandardScaler`, `ConcatDataset` 等）
- [ ] **2.5** 单元测试：验证 dataloader 输出 shape 正确
  - `batch_x`: `[B, T, N, F]`, `demos_x`: `[B, K, T, N, F]`

---

## Phase 3: OpenCity 模型 — `forward_ict` 方法 (P0, ~2 h)

- [ ] **3.1** 在 `model/OpenCity/OpenCity.py` 的 `OpenCity` 类中新增 `forward_ict()` 方法
- [ ] **3.2** Query 处理路径（复用 `forward()` 逻辑）
  - Temporal context encoding（`patch_embedding_time`）
  - Spatial PE（`spatial_embedding` + LaplacianPE）
  - Instance Normalization（per-query, per-node）
  - Patch embedding（`patch_embedding_flow`）→ `[B, 24, N, D]`
- [ ] **3.3** Demo 处理循环（K 个 demo）
  - 对每个 demo 分别执行：temporal encoding、instance norm（基于 demo 自身的 history）、patch embedding
  - Demo history → `[B, 24, N, D]`，Demo future → `[B, 24, N, D]`，拼接 → `[B, 48, N, D]`
  - 缓存 `dk_TP` 避免重复计算（优化点）
- [ ] **3.4** 序列拼接
  - `enc_all = cat([demo1_enc(48), ..., demoK_enc(48), query_enc(24)], dim=1)` → `[B, K*48+24, N, D]`
  - `TH_all = cat([demo1_TH(48), ..., demoK_TH(48), query_TH(24)], dim=1)`
  - `TP_all = cat([demo1_TP(48), ..., demoK_TP(48), query_TP(24)], dim=1)`
- [ ] **3.5** Encoder blocks 处理 + Query 提取 + 预测头
  - 遍历 `encoder_blocks`，传入 `enc_all, TH_all, TP_all, adj, geo_mask`
  - 从 `enc_all[:, -24:, :, :]` 提取 query 的 24 个 patches
  - 通过 `flatten → linear → reshape` 产生预测 → `[B, T, N, 1]`
  - De-Instance-Normalization 使用 query 自身的 mean/stdev
- [ ] **3.6** 确保 `forward()` 方法完全不被修改（向后兼容）

---

## Phase 4: 模型封装层修改 (P0, ~15 min)

- [ ] **4.1** 修改 `model/Model.py` 中 `Traffic_model.forward()`
  - 新增可选参数 `demos_x=None, demos_y=None`
  - 当 `demos_x is not None` 且 `model == 'OpenCity'` 时调用 `self.predictor.forward_ict()`
  - 当 `demos_x is None` 时保持原有逻辑不变
- [ ] **4.2** 验证向后兼容性：不传 demo 参数时行为完全一致

---

## Phase 5: ICT 测试方法 (P0, ~1 h)

- [ ] **5.1** 在 `model/BasicTrainer.py` 中新增 `test_ict()` 静态方法
  - 参数：`model, args, scaler_dict, test_dataloader, logger, path, num_prefix_selections`
  - 加载预训练权重（若 `path` 非空）
  - 冻结所有参数（`requires_grad = False`）
  - `model.eval()` + `torch.no_grad()` 上下文
- [ ] **5.2** 实现推理循环
  - 从 dataloader 获取 `(inputs, targets, demos_x, demos_y)`
  - `squeeze(0)` 移除外层 batch 维度
  - 通过 `get_key_from_value(num_nodes_dict, ...)` 识别数据集
  - 调用 `model(inputs, targets, select_dataset, demos_x=demos_x, demos_y=demos_y)`
- [ ] **5.3** 实现 `num_prefix_selections > 1` 的多次采样平均逻辑
  - 多次前向传播，输出取均值（降低 demo 选择方差）
- [ ] **5.4** 指标计算与日志输出
  - 复用 `All_Metrics()` 计算 MAE / RMSE / MAPE（`mask_value=0.001`）
  - 逆标准化处理（`scaler.inverse_transform`）

---

## Phase 6: 模式调度集成 (P0, ~30 min)

- [ ] **6.1** 修改 `model/Run.py`：新增 `mode='ict'` 分支
  - 导入 `from lib.ict_data_process import define_ict_dataloader`
  - 加载预训练权重到模型
  - 冻结所有参数（zero-training）
  - 调用 `define_ict_dataloader(args)` 创建 ICT dataloaders
  - 调用 `trainer.test_ict()` 执行推理
- [ ] **6.2** 确保 `mode='ict'` 不影响现有 mode 的代码路径（pretrain / ori / eval / test）

---

## Phase 7: 集成测试与调试 (P2, ~2 h)

- [ ] **7.1** Shape 一致性验证
  - 构造小 batch mock 数据，验证 `forward_ict()` 各步骤 shape 正确
  - 特别检查：`enc_all` shape `[B, K*48+24, N, D]`，输出 shape `[B, T, N, 1]`
- [ ] **7.2** 回归测试：K=0 等价性
  - 当 K=0（无 demo）时，`forward_ict` 的结果应等价于 `forward`（需要在代码中处理 K=0 的 fallback）
- [ ] **7.3** 单数据集端到端测试
  - `python main.py -mode ict -model OpenCity -dataset_use "['CAD3']" -load_pretrain_path OpenCity-plus.pth -num_demonstrations 1`
  - 确认无报错，输出 MAE/RMSE/MAPE 指标
- [ ] **7.4** 多数据集兼容性测试
  - 在不同 temporal interval 数据集上测试（5min: CAD3, 10min: CD_DIDI, 30min: CHI_TAXI）
  - 验证 variable-resolution patch embedding 与 ICT 兼容
- [ ] **7.5** GPU 显存估算
  - K=1, batch_size=32 → 预计 ~等同 baseline batch_size=64
  - K=3, 需降低 batch_size → 测试找到可用配置

---

## Phase 8: 实验运行与 Baseline 对比 (P2, ~4 h)

- [ ] **8.1** 运行 Zero-shot baseline（`mode=test`）在所有 8 个数据集，记录 MAE/RMSE/MAPE
- [ ] **8.2** 运行 Fast Adaptation baseline（`mode=eval`, 3 epochs）在所有 8 个数据集
- [ ] **8.3** 运行 ICT K=1（`mode=ict, num_demonstrations=1`）在所有 8 个数据集
- [ ] **8.4** 运行 ICT K=3（`mode=ict, num_demonstrations=3`）在所有 8 个数据集
- [ ] **8.5** 运行 ICT K=1 × 10 avg（`mode=ict, num_demonstrations=1, num_prefix_selections=10`）
- [ ] **8.6** 整理结果对比表格

---

## Phase 9: 高级特性与分析 (P3, ~4 h)

- [ ] **9.1** Demo 选择策略扩展
  - `recent`：选择时间上最接近 query 的 demo
  - `similar`：基于 DTW 或余弦相似度选择 demo
  - `same_time`：选择同一 time-of-day / day-of-week 的 demo
- [ ] **9.2** Attention 可视化
  - 提取 TC Cross-Attention 权重，分析 query 对 demo history vs. demo future 的注意力分布
  - 验证 query 确实在利用 demo 的 future flow 信息
- [ ] **9.3** 消融实验设计
  - Demo future flow 被替换为 zeros → 验证 future 信息的价值
  - Demo 来自不同数据集 → 测试 cross-domain 鲁棒性
  - 不同 K 值的系统对比 (K=1,2,3,5)

---

## 文件变更总览

| 文件 | 变更类型 | 对应 Phase |
|------|----------|-----------|
| `lib/Params_pretrain.py` | 修改 | Phase 1 |
| `conf/ICT/ICT.conf` | **新建** | Phase 1 |
| `lib/ict_data_process.py` | **新建** | Phase 2 |
| `model/OpenCity/OpenCity.py` | 修改（新增方法） | Phase 3 |
| `model/Model.py` | 修改 | Phase 4 |
| `model/BasicTrainer.py` | 修改（新增方法） | Phase 5 |
| `model/Run.py` | 修改 | Phase 6 |

---

## 关键依赖关系

```
Phase 0 (环境验证)
    │
    ├──→ Phase 1 (配置扩展)
    │
    ├──→ Phase 2 (ICT DataLoader)  ──┐
    │                                │
    └──→ Phase 3 (forward_ict)   ──┐ │
                                   │ │
         Phase 4 (Model.py)  ←────┘ │
              │                      │
              ▼                      │
         Phase 5 (test_ict)  ←──────┘
              │
              ▼
         Phase 6 (Run.py ict mode)
              │
              ▼
         Phase 7 (集成测试)
              │
              ▼
         Phase 8 (实验运行)
              │
              ▼
         Phase 9 (高级分析)
```

**MVP 最小交付（Phase 1-6）：预估 ~6 小时**
**完整交付（Phase 0-9）：预估 ~16 小时**
