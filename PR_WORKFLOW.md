# EDG++ Pull Request 工作流程文档

## 项目背景

这是向师兄仓库 [HongxinXiang/EDG](https://github.com/HongxinXiang/EDG) 提交EDG++代码的工作目录。

## 当前状态

- **本地仓库**: `/home/lzeng/workspace/EDG_for_PR`
- **你的GitHub fork**: https://github.com/katewangsiyu/EDG
- **师兄的原始仓库**: https://github.com/HongxinXiang/EDG
- **PR分支**: `edgpp-for-pr`
- **PR状态**: 已发起，等待师兄审核

## Git仓库配置

```bash
# 查看远程仓库配置
git remote -v

# 输出：
# origin    https://github.com/katewangsiyu/EDG.git (你的fork)
# upstream  https://github.com/HongxinXiang/EDG.git (师兄的仓库)
```

## 目录结构

```
/home/lzeng/workspace/
├── EDG_for_PR/              # PR工作目录（保留）
│   ├── EDG/                 # EDG++核心代码
│   │   ├── finetune_QM9_distillation.py
│   │   ├── finetune_MD17_distillation.py
│   │   ├── config_distillation.py
│   │   ├── distillation_utils.py
│   │   └── pretrain/        # 预训练代码
│   ├── pretrained_IEMv2_models/  # 预训练模型目录
│   └── README.md
├── GEOM3D/                  # 实验代码（原始开发目录）
└── EDG_IEEE_Trans_extracted/ # 论文LaTeX
```

## 常用操作

### 1. 查看当前状态

```bash
cd /home/lzeng/workspace/EDG_for_PR
git status
git branch  # 确认在 edgpp-for-pr 分支
```

### 2. 修改代码并提交

```bash
# 修改文件后...
git add 文件名              # 添加特定文件
# 或
git add -A                 # 添加所有改动

git commit -m "修改：描述你的改动"
```

### 3. 推送到GitHub（更新PR）

```bash
git push origin edgpp-for-pr
```

推送后，GitHub上的PR会自动更新，师兄会看到新的提交。

### 4. 同步师兄仓库的最新代码（如果需要）

```bash
# 获取师兄仓库的最新代码
git fetch upstream

# 将师兄的最新代码合并到你的分支
git merge upstream/master

# 如果有冲突，解决后提交
git push origin edgpp-for-pr
```

## 提交信息规范

使用中文，格式：`动作：具体内容`

示例：
- `修改：更新README安装说明`
- `修复：修正config_distillation.py中的参数错误`
- `添加：新增实验结果分析脚本`

## PR相关链接

- **你的fork仓库**: https://github.com/katewangsiyu/EDG
- **PR页面**: https://github.com/HongxinXiang/EDG/pulls (查看你发起的PR)
- **创建新PR**: https://github.com/katewangsiyu/EDG/pull/new/edgpp-for-pr

## 注意事项

1. **不要删除这个目录**：PR合并前需要用它来更新代码
2. **始终在 edgpp-for-pr 分支工作**：不要切换到master分支
3. **推送前先commit**：确保改动已提交到本地仓库
4. **网络问题**：如果push失败，可能是GitHub连接问题，稍后重试

## 常见问题

### Q: 如何查看PR状态？
访问 https://github.com/HongxinXiang/EDG/pulls 查看你的PR

### Q: 师兄要求修改代码怎么办？
在本目录修改代码，然后 `git add` → `git commit` → `git push origin edgpp-for-pr`

### Q: PR合并后这个目录还有用吗？
合并后可以保留用于后续开发，或者删除（代码已经在师兄仓库里了）

### Q: 如何查看提交历史？
```bash
git log --oneline -10  # 查看最近10条提交
```

## EDG++核心改动说明

本PR向师兄的EDG仓库添加了以下内容：

1. **蒸馏训练脚本**
   - `EDG/finetune_QM9_distillation.py` - QM9数据集蒸馏训练
   - `EDG/finetune_MD17_distillation.py` - rMD17数据集蒸馏训练

2. **配置文件**
   - `EDG/config_distillation.py` - 支持自适应阈值参数

3. **蒸馏工具**
   - `EDG/distillation_utils.py` - 选择性蒸馏机制实现

4. **预训练代码**
   - `EDG/pretrain/` - ED-aware Teacher和可靠性评估器训练代码

5. **预训练模型目录**
   - `pretrained_IEMv2_models/` - 预训练模型存放位置

## 创建时间

2026-03-31

---

如有问题，参考本文档或查看git历史记录。
