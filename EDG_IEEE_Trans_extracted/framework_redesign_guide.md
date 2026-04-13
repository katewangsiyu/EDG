# EDG++ 框架图重设计指南

> 以论文正文 `sections_edgpp/method.tex` 为准。下面给出修改后的四个面板的完整框架图和逐项说明。

---

## 一、修改后的完整框架图

### Panel (a) — Generation of Multi-view RGB-D Electron Density Images

```
┌─────────────────────────────────────────────────────────────────────────┐
│ (a) Generation of Multi-view RGB-D Electron Density Images              │
│                                                                         │
│                                                                         │
│   ⚛ ⚛ ⚛           DFT with            Cubing and         Multi-view   │
│   2 million  ──→   B3LYP        ──→    rendering    ──→  RGB-D ED     │
│   conformers       6-31G(d,p)           ED                images       │
│                    ─────────            ~~~~               ┌─┬─┬─┬─┬─┐ │
│                    [修正：原图写                             │ │ │ │ │ │ │
│                     6-31G**/+G**                           └─┴─┴─┴─┴─┘ │
│                     改为 6-31G(d,p)]                        6-view      │
│                                                            RGB-D(4ch)  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**修改点**：`6-31G**/+G**` → `6-31G(d,p)`（与论文 Section 3.3 一致）

---

### Panel (b) — ImageED (Multi-view Masked Autoencoder)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ (b) ImageED (Multi-view Masked Autoencoder)                                  │
│                                                                              │
│  ┌─┬─┬─┬─┬─┐                                                                │
│  │ │ │ │ │ │  6-view     patchify     ┌───────────┐     ┌───────────┐       │
│  └─┴─┴─┴─┴─┘  ED imgs  ──────────→   │ED Encoder │ ──→ │ED Decoder │       │
│   ...                    75% mask     │  f_EDE    │     │  f_EDD    │       │
│                                       └─────┬─────┘     └──┬────┬──┘       │
│                                             │               │    │          │
│                                             │          ┌────┘    └────┐     │
│                                             │          ▼              ▼     │
│                                             │     ┌─────────┐  ┌─────────┐ │
│                                             │     │ t^m_hat  │  │ t^u_hat  │ │
│                                             │     │(masked   │  │(visible  │ │
│                                             │     │ tokens)  │  │ tokens)  │ │
│                                             │     └────┬─────┘  └────┬─────┘ │
│                                             │          │              │      │
│                                             │          ▼              ▼      │
│                                             │      L_MR            L_VR     │
│                                             │   (masked reconst) (visible reconst) │
│                                             │          │              │      │
│                                             │          └──────┬───────┘      │
│                                             │                 ▼              │
│                                             │          L_ImageED =           │
│                                             │          λ_VR·L_VR + λ_MR·L_MR│
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

**修改点**：
- $\mathcal{L}_{MP}$ → $\mathcal{L}_{MR}$（masked reconstruction）
- $\mathcal{L}_{RP}$ → $\mathcal{L}_{VR}$（visible reconstruction）
- 与论文 Section 3.4 公式和图注 caption 保持一致

---

### Panel (c) — Pretraining of ED-aware Teacher

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ (c) Pretraining of ED-aware Teacher                                              │
│                                                                                  │
│                                                    ┌──────────────┐              │
│                                                    │ ED Encoder   │ ❄ frozen     │
│                                                    │ f_EDE        │              │
│                                                    │ (ViT-Large)  │              │
│                                                    └──────┬───────┘              │
│                                                     Token-wise                   │
│  Structural                                         avg pooling                  │
│  images S                                                │                       │
│  ┌─┬─┬─┐                                                ▼                       │
│  │ │ │ │ 4-view                                     ┌────────┐                   │
│  └─┴─┴─┘                                           │ F^U    │                   │
│      │                                              │(target)│                   │
│      ▼                                              └───┬────┘                   │
│  ┌──────────────┐      ┌──────────────┐                 │                        │
│  │ ED-aware     │ 🔥   │ ED Predictor │ 🔥              │                        │
│  │ Teacher f_S  │      │ f_EDP        │                 │                        │
│  │ (ResNet18)   │      │ (2-layer MLP)│                 │                        │
│  └──────┬───────┘      └──────┬───────┘                 │                        │
│         │ F^S                 │ F^{S→U}                 │                        │
│         │                     │                         │                        │
│         ├────────→ f_EDP ─────┤                         │                        │
│         │                     │                         │                        │
│         │                     ▼                         ▼                        │
│         │                  ┌─────────────────────────────────┐                   │
│         │                  │         L_Align (L1 loss)       │                   │
│         │                  │     F^{S→U}    vs    F^U        │                   │
│         │                  └──────────────┬──────────────────┘                   │
│         │                                 │                                      │
│         │                          ‖F^{S→U} − F^U‖₂                             │
│         │                          (reconstruction error)                        │
│         │                                 │                                      │
│         │ ⊘ sg                            │ ⊘ sg                                │
│         │ (input detach)                  │ (target detach)                      │
│         │                                 │                                      │
│         ▼                                 ▼                                      │
│     ┌───────────────────────────────────────────────┐                            │
│     │          Reliability Estimator                │ 🔥 trainable               │
│     │          φ_eval (MLP: 512→256→1)              │                            │
│     │                                               │                            │
│     │   input: sg(F^S)    target: sg(‖F^{S→U}−F^U‖₂)│                           │
│     └────────────────────────┬──────────────────────┘                            │
│                              │                                                   │
│                           L_eval                                                 │
│                              │                                                   │
│                              ▼                                                   │
│                        c_i = −φ_eval(·)                                          │
│                        (confidence score → Stage d)                              │
│                                                                                  │
│  Total loss = L_Align + L_eval （联合训练 f_S, f_EDP, φ_eval）                    │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

**修改点**：
1. 输入标注 "Multi-view images" → `Structural images S`（4-view, RGB），与 ED images 区分
2. 显式画出 φ_eval 的 target 来源：从 L_Align 处引出 ‖F^{S→U}−F^U‖₂ 支路
3. 两处 ⊘ sg (stop-gradient) 标记：input detach + target detach
4. φ_eval 横跨底部，双输入结构一目了然
5. 各模块加架构名小字：ResNet18 / ViT-Large / 2-layer MLP / MLP:512→256→1

---

### Panel (d) — Quantum chemistry-enhanced knowledge distillation

```
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│ (d) Quantum chemistry-enhanced knowledge distillation                                    │
│                                                                                          │
│                          ┌──────────────┐                                                │
│                          │Task Predictor│ 🔥                                             │
│                          │ f_T          │                                                │
│                          └──────┬───────┘                                                │
│                                 │ y_hat                                                  │
│                                 ▼                                                        │
│   Molecular    ┌───────────┐   L_Task                                                   │
│   graphs  ──→  │ Geometry  │ 🔥  │                                                      │
│    ⚛           │ Student   │     │                                                      │
│                │ f_G       │     │                                                      │
│                └─────┬─────┘     │                                                      │
│                      │ F^G       │                                                      │
│                      │           │                                                      │
│          ┌───────────┤           │                                                      │
│          │           │           │                                                      │
│          │    f_T(F^G)=y_hat     │                                                      │
│          │                       │                                                      │
│          ▼                       │                                                      │
│   ┌────────────┐                 │                                                      │
│   │  Mapper    │ 🔥              │                                                      │
│   │  f_M       │                 │                                                      │
│   │ (Linear)   │                 │                                                      │
│   └─────┬──────┘                 │                                                      │
│         │ f_M(F^G)               │                                                      │
│         │                        │                                                      │
│         ▼                        │                                                      │
│   ┌──────────────┐ ❄            │                    ┌──────────┐                       │
│   │ ED Predictor │ frozen        │                    │          │                       │
│   │ f_EDP        │───────────────┼────────────────────│  ⊕ sum  │──→  L_EDG             │
│   │ (2-layer MLP)│               │                    │          │                       │
│   └──┬───────┬───┘            L_Task ────────────→    └──────────┘                       │
│      │       │                                            ↑                              │
│      │       │                                            │ ×λ                           │
│  F^{G→U}  F^{S→U}                                        │                              │
│  (from     (from                                          │                              │
│  student)  teacher)                                       │                              │
│      │       │                                            │                              │
│      │       │         ┌──────────────────────┐           │                              │
│      └───┬───┘         │ Adaptive Thresholding│           │                              │
│          │             │                      │           │                              │
│          ▼             │  τ = β·τ_local       │           │                              │
│   ┌─────────────┐      │    + (1−β)·τ_global  │           │                              │
│   │   L_ED      │      │                      │           │                              │
│   │ SL1 loss    │←─────│  m_i = 1[c_i ≥ τ]   │───────────┘                              │
│   │ (selective) │  m_i └──────────┬───────────┘                                          │
│   └─────────────┘           ↑          ↑                                                 │
│                             │          │                                                 │
│                          c_i (batch) μ_all, σ_all                                        │
│                             │        (precomputed)                                       │
│                             │                                                            │
│  Structural   ┌───────────────┐ ❄                                                       │
│  images S ──→ │ ED-aware      │ frozen                                                   │
│  ┌─┬─┬─┐     │ Teacher f_S   │                                                          │
│  │ │ │ │     │ (ResNet18)    │                                                          │
│  └─┴─┴─┘     └──────┬────────┘                                                          │
│  4-view              │ F^S                                                               │
│                      │                                                                   │
│              ┌───────┴────────┐                                                          │
│              │                │                                                          │
│              ▼                ▼                                                           │
│   ┌──────────────┐ ❄   ┌──────────────┐ ❄                                              │
│   │ Reliability  │      │ ED Predictor │                                                │
│   │ Estimator    │      │ f_EDP        │──→ F^{S→U} (上方汇入 L_ED)                     │
│   │ φ_eval       │      └──────────────┘                                                │
│   └──────┬───────┘                                                                       │
│          │                                                                               │
│       c_i = −φ_eval(F^S)                                                                │
│       (→ Adaptive Thresholding)                                                          │
│                                                                                          │
│  ── 实线 = trainable path ──  --- 虚线 = frozen/precomputed path ---                     │
│                                                                                          │
│  Inference: 只需 f_G + f_T，无额外计算开销                                                │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

**修改点**：
1. 补上 $\mathcal{L}_{ED} \xrightarrow{\times\lambda} \oplus \leftarrow \mathcal{L}_{Task} \to \mathcal{L}_{EDG}$ 的完整连接
2. 输入标注 "Multi-view images" → `Structural images S`（4-view）
3. Adaptive Thresholding 增加 $\mu_{all}, \sigma_{all}$（precomputed）作为第二输入
4. 显式标注 $\tau = \beta \cdot \tau_{local} + (1-\beta) \cdot \tau_{global}$ 和 $m_i = \mathbb{1}[c_i \ge \tau]$
5. 标注 Inference 时只需 $f_G + f_T$

---

## 二、修改清单汇总

| # | 面板 | 优先级 | 问题 | 改法 |
|---|------|--------|------|------|
| 1 | (a) | P0 必须 | 基组 `6-31G**/+G**` 错误 | → `6-31G(d,p)` |
| 2 | (b) | P0 必须 | $\mathcal{L}_{MP}$/$\mathcal{L}_{RP}$ 与论文不一致 | → $\mathcal{L}_{MR}$/$\mathcal{L}_{VR}$ |
| 3 | (d) | P0 必须 | $\mathcal{L}_{ED} \to \mathcal{L}_{EDG}$ 连接缺失 | 加箭头 + `×λ` 标注 |
| 4 | (c)(d) | P1 建议 | "Multi-view images" 不精确 | → `Structural images S`（4-view） |
| 5 | (c) | P1 建议 | $\phi_{eval}$ 的 target 来源不可见 | 画出 ‖F^{S→U}−F^U‖₂ 支路 + 两处 sg |
| 6 | (d) | P1 建议 | Adaptive Thresholding 缺全局统计输入 | 加 $\mu_{all}, \sigma_{all}$ 输入 |
| 7 | (c) | P2 可选 | 模块缺架构名 | 加 ResNet18/ViT-Large/MLP 小字 |

---

## 三、图注 Caption（建议更新）

原 caption 中 `$\mathcal{L}_{VR}$ and $\mathcal{L}_{MR}$` 已正确，无需改动。建议在 (c) 描述中补充：

> **(c)** ... The reliability estimator $\phi_{eval}$ is jointly trained to predict the reconstruction error $\|\mathcal{F}^{\mathcal{S}\rightarrow\mathcal{U}} - \mathcal{F}^{\mathcal{U}}\|_2$ with stop-gradient on both input and target.

在 (d) 描述中补充：

> **(d)** ... The adaptive threshold combines batch-level and dataset-level statistics. The total loss is $\mathcal{L}_{EDG} = \mathcal{L}_{Task} + \lambda \mathcal{L}_{ED}$.
