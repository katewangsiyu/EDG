# Phase 1: 数据探索总结

## 数据结构发现

### 1. Teacher Features (QM9)
**文件**: `GEOM3D/examples_3D/dataset/QM9/processed/teacher_feats/ED_Evaluator_from_10_epoch_best_epoch=18_loss=0.30.npz`

- **Keys**: `['feats', 'drug_id']`
- **Shape**:
  - `feats`: (130831, 512) - ED特征向量
  - `drug_id`: (130831,) - 分子ID (gdb_1, gdb_2, ...)
- **统计**:
  - Min: 0.0, Max: 4.73, Mean: 0.27, Std: 0.24

**关键发现**: 可靠性分数**不是预存储的**，而是在训练时通过EDEvaluator网络动态计算。

### 2. 可靠性分数计算流程
```python
# 从代码 finetune_QM9_distillation.py:735
ED_image_confidence = -1 * EDEvaluator(batch.img_feat)
```

- **输入**: img_feat (512维，来自teacher_feats)
- **模型**: EDEvaluator (两层MLP: 512→256→1)
- **输出**: confidence score (标量，乘以-1后值越大越好)
- **预训练模型**: `pretrained_IEMv2_models/.../best_epoch=18_loss=0.30.pth` (90MB)

### 3. 实验结果文件
**文件**: `experiments/run_QM9_distillation/.../evaluation_best.pth.npz`

- **Keys**: `['val_target', 'val_pred', 'test_target', 'test_pred']`
- **Shape**:
  - val: (10000,) - 验证集
  - test: (10831,) - 测试集
- **内容**: 每个样本的真实值和预测值

### 4. 可用实验数量

#### QM9 (3个模型 × 12个任务)
- **SchNet**: 540个实验 (EDG=180, EDG++=180)
- **Equiformer**: 256个实验 (EDG=86, EDG++=84)
- **SphereNet**: 539个实验 (EDG=179, EDG++=180)

#### rMD17 (SphereNet × 10个分子)
- **SphereNet**: 68个实验

**关键发现**: 有大量EDG和EDG++的配对实验，可以进行对比分析。

## 数据可行性评估

### ✅ 可以做的分析

1. **计算可靠性分数**
   - 加载预训练的EDEvaluator模型
   - 对所有QM9样本计算可靠性分数
   - 保存为新的npz文件供后续分析

2. **EDG vs EDG++的样本级对比**
   - 对于同一任务，加载EDG和EDG++的预测结果
   - 计算每个样本的误差差异
   - 分析哪些样本在EDG++中改进了

3. **可靠性分数与预测误差的相关性**
   - 在EDG实验中，分析低可靠性样本是否误差更大
   - 验证可靠性评估器的有效性

4. **分层分析**
   - 按可靠性分数分组（高/中/低）
   - 比较各组在EDG/EDG++中的表现

### ❌ 无法做的分析

1. **Baseline vs EDG的改进幅度**
   - 原因：没有Baseline (weight_ED=0) 的详细预测结果
   - 用户正在跑Naphthalene + Uracil的baseline

## 下一步建议

### 方案A: 先计算可靠性分数（推荐）
创建脚本计算所有QM9样本的可靠性分数，保存为：
```
GEOM3D/examples_3D/dataset/QM9/processed/reliability_scores.npz
```

### 方案B: 直接进行EDG vs EDG++分析
使用现有的实验结果，分析样本级差异。

**建议**: 先执行方案A，为后续所有分析提供基础数据。
