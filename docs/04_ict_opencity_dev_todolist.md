# ICT-OpenCity 开发 Todo List (v3 — Residual Correction)

基于三份文档（OpenCity 代码分析、ICT 代码分析、ICT-OpenCity 集成方案）及实验反馈后的修订版。

**v3 核心变更**：
- **Phase 0-6 全部完成** — 代码实现已就绪
- **架构方案从 v1（序列拼接）切换为 v2（残差修正）** — 见 `05_experiment_log.md § Approach Evolution`
- v1 序列拼接方案经实验验证后废弃（Exp 2-4），原因：预训练模型只见过 24-patch 序列，扩展为 OOD
- v2 残差修正：每次 forward 调用均为标准 24-patch，用 demo 残差修正 query 预测
- fp32 推理，不再需要 bfloat16

---

## Phase 0: 环境准备与 Baseline 验证 ✅ Done

- [x] **0.1** 确认预训练权重存在：`model_weights/OpenCity/OpenCity-plus.pth`
- [x] **0.2** 运行 zero-shot baseline (PEMS07M)：MAE 4.50, RMSE 8.21, MAPE 12.20%
- [ ] **0.3** 运行 Fast Adaptation baseline（CD_DIDI），记录对比指标
- [ ] **0.4** 确认所有 8 个评估数据集可正常加载

**Phase 0 Gate**: ✅ PASS — zero-shot baseline 已记录（Exp 1）。

---

## Phase 1: 配置系统扩展 ✅ Done

- [x] **1.1** 修改 `lib/Params_pretrain.py`：新增 ICT 参数（`-num_demonstrations`, `-num_prefix_selections`, `-demo_selection`）
- [x] **1.2** 创建 `conf/ICT/ICT.conf`
- [x] **1.3** 创建 `conf/general_conf/dataset_splits.conf`
- [x] **1.4** 实现 `load_dataset_splits()` + `get_dataset_split()` in `lib/data_process.py`
- [x] **1.5** 修改 `lib/data_process.py` — `define_dataloder()` 读取 `dataset_splits.conf`

**Phase 1 Gate**: ✅ PASS

---

## Phase 2: ICT 数据加载器 ✅ Done

- [x] **2.1** 创建 `lib/ict_data_process.py`，添加必要 import
- [x] **2.2** 实现 `ICTTrafficDataset(Dataset)` 类
  - Shape 契约：`batch_x [B,T,N,F]`, `demos_x [B,S,K,T,N,F]`
  - K=0 时返回空 tensor（已修复 np.stack 崩溃问题）
- [x] **2.5** 实现 `define_ict_dataloader(args)` 函数

**Phase 2 Gate**: ✅ PASS — 端到端运行无报错。

---

## Phase 3: OpenCity 模型 — `forward_ict` 方法 ✅ Done

> **重要**：v1（序列拼接）已实现并测试（Exp 2-4），因性能问题废弃。当前代码为 v2（残差修正）。

- [x] **3.1** ~~v1: 序列拼接 forward_ict~~ → 已废弃
- [x] **3.2** ~~v1: Block-diagonal attention mask~~ → 已废弃
- [x] **3.3** v2: 残差修正 `forward_ict()` — 当前实现
  ```python
  # 核心逻辑（model/OpenCity/OpenCity.py）
  pred_query = self.forward(input, lbls, select_dataset)  # 标准 24-patch
  for k in range(K):
      pred_dk = self.forward(dk_x, dk_y, select_dataset)  # 标准 24-patch
      residuals.append(dk_gt_flow - pred_dk)               # 系统误差
  return pred_query + mean(residuals)                       # 修正预测
  ```
- [x] **3.4** K=0 回归：正确退化为 `forward()` 逻辑
- [x] **3.5** `forward()` 原方法完全未被修改

**Phase 3 Gate**: ✅ PASS — 端到端运行，K=0 退化正确。

> **遗留代码**：`TemporalSelfAttention.forward()` 和 `STEncoderBlock.forward()` 中的 `t_attn_mask=None` 参数为 v1 遗留，默认 None，无影响。

---

## Phase 4: 模型封装层修改 ✅ Done

- [x] **4.1** 修改 `model/Model.py` — `Traffic_model.forward()`
  - 新增 `demos_x=None, demos_y=None` 可选参数
  - 有 demo → `self.predictor.forward_ict(...)`；无 demo → 原逻辑
- [x] **4.2** 向后兼容：无 demo 时输出与修改前完全一致

**Phase 4 Gate**: ✅ PASS

---

## Phase 5: ICT 测试方法 ✅ Done

- [x] **5.1** 在 `model/BasicTrainer.py` 新增 `test_ict()` 静态方法
- [x] **5.2** 推理循环：正确处理 `[B, S, K, T, N, F]` 的 demo 维度，循环 S 次取平均
- [x] **5.3** 指标计算：复用 `All_Metrics()`，逆标准化 `scaler.inverse_transform`

**Phase 5 Gate**: ✅ PASS

> **待清理**：`test_ict` 中仍有 v1 遗留的 bfloat16 autocast 代码，功能正常但不再需要。

---

## Phase 6: 模式调度集成 ✅ Done

- [x] **6.1** 修改 `model/Run.py`：新增 `mode='ict'` 分支（fp32 推理，无 bfloat16）
- [x] **6.2** 确保不影响现有 mode（pretrain / ori / eval / test）

**Phase 6 Gate**: ✅ PASS — 端到端 smoke test 通过。

---

## Phase 7: v2 残差修正实验 🔄 In Progress

> 替代原 Phase 7（v1 全面集成测试）。v2 无序列扩展，无 OOM 风险，无兼容性问题。

- [ ] **7.1** **Exp 5**: Residual K=1, S=1, fp32, PEMS07M（基础验证）
  ```bash
  cd model && uv run python Run.py -mode ict -model OpenCity \
      -load_pretrain_path OpenCity-plus.pth -num_demonstrations 1 \
      -batch_size 1 --embed_dim 512 --skip_dim 512 --enc_depth 6
  ```
- [ ] **7.2** **Exp 6**: Residual K=3, S=1, fp32, PEMS07M（更多 demo → 更好误差估计）
- [ ] **7.3** **Exp 7**: Residual K=1, S=10, fp32, PEMS07M（方差降低验证）
- [ ] **7.4** 分析结果：residual correction 是否优于 zero-shot？确认方向

**Phase 7 Gate**: Exp 5 在 PEMS07M 上 MAE ≤ 4.50（达到或优于 zero-shot）。

---

## Phase 8: 全数据集实验运行 (~4 h)

- [ ] **8.1** Zero-shot baseline（`mode=test`）全 8 数据集
- [ ] **8.2** Fast Adaptation baseline（`mode=eval`, 3 epochs）全 8 数据集
- [ ] **8.3** ICT Residual K=1（`mode=ict, -num_demonstrations 1`）全 8 数据集
- [ ] **8.4** ICT Residual K=3（`mode=ict, -num_demonstrations 3`）全 8 数据集
- [ ] **8.5** ICT Residual K=1 × S=10（`-num_demonstrations 1 -num_prefix_selections 10`）全 8 数据集
- [ ] **8.6** 整理结果对比表（MAE / RMSE / MAPE）

---

## Phase 9: 高级特性与分析 (~4 h)

- [ ] **9.1** Demo 选择策略：`recent`（时间最近）、`similar`（DTW/余弦相似度）、`same_time`（同 time-of-day）
- [ ] **9.2** 加权残差修正：用 query 与 demo 的 history 嵌入相似度加权残差（替代均匀平均）
- [ ] **9.3** 消融实验
  - K = 1, 2, 3, 5, 10 系统对比
  - S = 1, 5, 10, 20 方差降低曲线
  - Demo 来自不同数据集（cross-domain 残差迁移）

---

## 文件变更总览

| 文件 | 变更类型 | Phase | 状态 |
|------|----------|-------|------|
| `lib/Params_pretrain.py` | 修改 | 1 | ✅ Done |
| `conf/ICT/ICT.conf` | **新建** | 1 | ✅ Done |
| `conf/general_conf/dataset_splits.conf` | **新建** | 1 | ✅ Done |
| `lib/data_process.py` | 修改 | 1 | ✅ Done |
| `lib/ict_data_process.py` | **新建** | 2 | ✅ Done |
| `model/OpenCity/OpenCity.py` | 修改（`forward_ict` v2 残差修正） | 3 | ✅ Done |
| `model/Model.py` | 修改 | 4 | ✅ Done |
| `model/BasicTrainer.py` | 修改（`test_ict`） | 5 | ✅ Done |
| `model/Run.py` | 修改（`mode='ict'` fp32） | 6 | ✅ Done |

---

## 接口契约

| 接口 | Provider → Consumer | Shape | 状态 |
|------|---------------------|-------|------|
| Dataloader → test_ict | Phase 2 → Phase 5 | `demos_x: [B, S, K, T, N, F]` | ✅ 验证 |
| test_ict → Model.forward | Phase 5 → Phase 4 | `demos_x: [B, K, T, N, F]` (per-selection) | ✅ 验证 |
| Model.forward → forward_ict | Phase 4 → Phase 3 | `demos_x: [B, K, T, N, F]` | ✅ 验证 |
| forward_ict → self.forward | Phase 3 内部 | 标准 `[B, T, N, F]` 输入，24-patch | ✅ 验证 |
| forward_ict → output | Phase 3 | `[B, T, N, 1]` | ✅ 验证 |

---

## 依赖关系

```
Phase 0 (环境验证 + Baseline) ──── ✅ DONE
    │
    ├──→ Phase 1 (配置扩展) ──── ✅ DONE
    │
    ├──→ Phase 2 (ICT DataLoader) ──── ✅ DONE
    │         │
    └──→ Phase 3 (forward_ict v2) ──── ✅ DONE
              │
              ▼
         Phase 4 (Model.py) ──── ✅ DONE
              │
              ▼
         Phase 5 (test_ict) ──── ✅ DONE
              │
              ▼
         Phase 6 (Run.py) ──── ✅ DONE
              │
              ▼
         Phase 7 (v2 实验验证) ──── 🔄 IN PROGRESS — 下一步
              │
              ▼
         Phase 8 (全数据集实验)
              │
              ▼
         Phase 9 (高级分析)
```

**代码实现（Phase 0-6）：✅ 完成**
**下一步：Phase 7 — 运行 Exp 5（Residual K=1 fp32 PEMS07M）验证残差修正方案**
