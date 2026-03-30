# EDG++ 中文说明

## 简介

EDG++ 是 IJCAI 2025 论文 EDG 的扩展版本，引入了  选择性知识蒸馏  机制，使用电子密度作为特权信息增强分子表示学习。

## 核心创新

- **预训练可靠性评估器**：在预训练阶段学习评估ED预测质量
- **自适应阈值机制**：全局+局部混合过滤策略
- **缓解负迁移**：过滤不可靠的ED预测，防止性能下降

## 快速开始

详见 [README.md](README.md)

## 文档

- [安装指南](docs/INSTALL.md)
- [数据集准备](docs/DATASETS.md)
- [预训练指南](docs/PRETRAIN.md)
- [蒸馏训练](docs/DISTILLATION.md)
- [常见问题](docs/FAQ.md)

## 引用

```bibtex
@inproceedings{edg2025ijcai,
  title={EDG: Cross-modal Knowledge Distillation with Electron Density},
  booktitle={IJCAI},
  year={2025}
}
```
