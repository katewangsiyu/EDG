# EDG_for_PR — Claude Code 项目指南

## ⚠️ 这是 monorepo，不是单项目

本目录混合了 **多条研究线** 和 **多个交付物**。Claude 进来**先问清楚"哪一条"**，再动手。禁止在目录间跨项目擅自修改。

---

## 🗺 子目录导航（最重要）

| 子目录 | 是什么 | 上下游 |
|---|---|---|
| `EDG/` | EDG++ 核心代码（QM9/MD17 蒸馏、finetune、EDG baseline） | 要 PR 到师兄 repo `HongxinXiang/EDG`（见 `PR_WORKFLOW.md`） |
| `EDG_IEEE_Trans_extracted/` | **EDG++ IEEE Trans 期刊论文** LaTeX 源文件（`main_edgpp.tex/pdf`） | 当前期刊投稿主战场 |
| `EDG_IEEE_Trans_extracted/IJCAI-2025-EDG-camera/` | **IJCAI 2025 EDG camera-ready** 源 | 已投/已接收的前作 |
| `ED_teacher/` | ED teacher 模型预训练（`pretrain_ED_teachers.py`） | 为 EDG/ImageED 提供 teacher checkpoint |
| `ImageED/` | 图像版 ED（MAE 多视图 RGBD 预训练） | 与 EDG/ 平行的另一条 modality |
| `EDG-for-VisNet/` | 独立项目，环境与 VisNet 一致 | 和 `/home/lzeng/workspace/VISNET` 有关联 |
| `GEOM3D/` | 外部 baseline 代码（Liu et al., ArXiv 2306.09375） | 只读依赖，不改 |
| `pretrain/pretrained_models/` | 预训练权重存储位置 | 产物，不是代码 |
| `downstream/` `analysis/` `docs/` | 下游任务 / 分析脚本 / 文档 | 共享工具 |

**识别规则**：用户说"EDG++" 默认指 `EDG_IEEE_Trans_extracted/`（期刊）和 `EDG/`（代码）的组合；说"IJCAI 版本"指 `IJCAI-2025-EDG-camera/`；说"图像版"指 `ImageED/`。拿不准就问。

---

## 🎯 当前主要交付物

1. **EDG++ IEEE Trans 期刊投稿**（进行中）
   - 源：`EDG_IEEE_Trans_extracted/main_edgpp.tex`
   - 主图：`EDG_IEEE_Trans_extracted/imgs_edgpp/our_method/EDG++框架图_en.png`
   - Bib：`edgpp_journal.bib`（有 `.bak` 备份）
   - 最近改动：作者顺序（Jun Xia 置于 Xin Jin 之前，commit `4234749`）

2. **向师兄 repo 提 PR**（`EDG/` 子目录）
   - 流程详情见 `PR_WORKFLOW.md`
   - Fork: `katewangsiyu/EDG`；Upstream: `HongxinXiang/EDG`；PR 分支 `edgpp-for-pr`

---

## 📦 仓库配置

- 远程：`https://github.com/katewangsiyu/EDG.git`（你的 fork）
- 分支：`main`
- 提交：中文，格式 `动作：具体内容`（例 `修改：…` / `添加：…` / `删除：…`）
- 完成工作主动 `git add <文件>` → `commit` → `push`（同根 CLAUDE.md 规则）
- 大文件：`.gitignore` 已排除常见权重 + 数据；新增大文件先更 `.gitignore`
- **注意**：多次出现 `.bak` 备份文件（`.bib.bak`、`.xlsx.bak`），说明你习惯手工备份——Claude 也照做

---

## 🧭 新会话续跑

1. 先确认本次对话针对哪个子目录 / 哪个交付物
2. 若是论文写作 → 进 `EDG_IEEE_Trans_extracted/`，读 `sections_edgpp/*.tex`
3. 若是代码/实验 → 进 `EDG/` 或 `ImageED/` 或 `ED_teacher/`，读对应目录里的脚本
4. `git log --oneline -10` 看最近状态
5. 不清楚就问，不要跨目录猜测

---

## 💬 交互

- 中文优先，简洁
- 目标/路径不清先问；路径非最短明说并给备选（承袭根 CLAUDE.md 第一性原理）
- **不要在未确认子目录归属时执行 Write/Edit**
