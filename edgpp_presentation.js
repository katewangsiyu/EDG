const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "EDG++ Team";
pres.title = "EDG++: Reliability-Aware Electron Density-Enhanced Molecular Geometry Learning";

// === Color Palette (Ocean Gradient) ===
const C = {
  darkBg: "0A2540",
  primary: "065A82",
  secondary: "1C7293",
  accent: "0891B2",
  midnight: "21295C",
  lightBg: "F0F5F9",
  white: "FFFFFF",
  textDark: "1E293B",
  textMid: "475569",
  textLight: "94A3B8",
  border: "E2E8F0",
  success: "059669",
  warning: "D97706",
  error: "DC2626",
  purple: "7C3AED",
};

// === Figure Paths ===
const IMG = "/home/lzeng/workspace/EDG_IEEE_Trans_extracted/imgs_edgpp";
const fig = (name) => path.join(IMG, name);

// === Reusable factory functions (avoid object reuse pitfall) ===
const makeShadow = () => ({ type: "outer", blur: 6, offset: 2, angle: 135, color: "000000", opacity: 0.12 });
const makeCardShadow = () => ({ type: "outer", blur: 8, offset: 3, angle: 135, color: "000000", opacity: 0.10 });

const TOTAL = 26;

// === Slide helpers ===
function addBottomBar(slide, pageNum) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 5.225, w: 10, h: 0.4, fill: { color: C.primary }
  });
  slide.addText(`${pageNum} / ${TOTAL}`, {
    x: 8.5, y: 5.225, w: 1.2, h: 0.4,
    fontSize: 10, color: C.white, align: "center", valign: "middle", fontFace: "Calibri"
  });
}

function addSectionTag(slide, section) {
  slide.addText(section, {
    x: 0.3, y: 5.225, w: 3, h: 0.4,
    fontSize: 9, color: "CADCFC", align: "left", valign: "middle", fontFace: "Calibri", italic: true
  });
}

function addSlideTitle(slide, title) {
  slide.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 0.3, w: 0.08, h: 0.5, fill: { color: C.accent }
  });
  slide.addText(title, {
    x: 0.75, y: 0.3, w: 8.75, h: 0.5,
    fontSize: 24, fontFace: "Georgia", color: C.textDark,
    bold: true, align: "left", valign: "middle", margin: 0
  });
}

function contentSlide(title, pageNum, section) {
  const slide = pres.addSlide();
  slide.background = { color: C.lightBg };
  addSlideTitle(slide, title);
  addBottomBar(slide, pageNum);
  addSectionTag(slide, section);
  return slide;
}

function sectionCover(num, title, subtitle, pageNum) {
  const slide = pres.addSlide();
  slide.background = { color: C.darkBg };
  // Number circle
  slide.addShape(pres.shapes.OVAL, { x: 4.25, y: 1.2, w: 1.5, h: 1.5, fill: { color: C.accent } });
  slide.addText(num, {
    x: 4.25, y: 1.2, w: 1.5, h: 1.5,
    fontSize: 36, fontFace: "Georgia", color: C.white, bold: true, align: "center", valign: "middle"
  });
  slide.addText(title, {
    x: 1, y: 3.0, w: 8, h: 0.8,
    fontSize: 32, fontFace: "Georgia", color: C.white, bold: true, align: "center", valign: "middle"
  });
  if (subtitle) {
    slide.addText(subtitle, {
      x: 1, y: 3.8, w: 8, h: 0.5,
      fontSize: 14, fontFace: "Calibri", color: "94A3B8", align: "center", valign: "middle"
    });
  }
  addBottomBar(slide, pageNum);
  addSectionTag(slide, title);
  return slide;
}

// ================================================================
// SLIDE 1 — Title
// ================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.darkBg };
  // Top accent line
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });
  // Title
  s.addText("EDG++", {
    x: 1, y: 1.0, w: 8, h: 0.9,
    fontSize: 52, fontFace: "Georgia", color: C.white, bold: true, align: "center", valign: "middle", charSpacing: 6
  });
  s.addText("Reliability-Aware Electron Density-Enhanced\nMolecular Geometry Learning", {
    x: 1, y: 1.95, w: 8, h: 0.85,
    fontSize: 18, fontFace: "Calibri", color: "CADCFC", align: "center", valign: "middle"
  });
  // Divider
  s.addShape(pres.shapes.RECTANGLE, { x: 3.5, y: 3.05, w: 3, h: 0.02, fill: { color: C.accent } });
  // Authors
  s.addText("Hongxin Xiang, Xin Jin, Jun Xia, Wenjie Du, Haowen Chen,\nStan Z. Li (Fellow, IEEE), Xiangxiang Zeng (Senior Member, IEEE)", {
    x: 1, y: 3.25, w: 8, h: 0.7,
    fontSize: 12, fontFace: "Calibri", color: "94A3B8", align: "center", valign: "middle"
  });
  s.addText("Hunan University  ·  Westlake University  ·  USTC", {
    x: 1, y: 3.95, w: 8, h: 0.35,
    fontSize: 11, fontFace: "Calibri", color: "64748B", align: "center", valign: "middle"
  });
  s.addText("IEEE Transactions on Knowledge and Data Engineering", {
    x: 1, y: 4.4, w: 8, h: 0.35,
    fontSize: 11, fontFace: "Calibri", color: C.accent, italic: true, align: "center", valign: "middle"
  });
  addBottomBar(s, 1);
}

// ================================================================
// SLIDE 2 — Table of Contents
// ================================================================
{
  const s = contentSlide("目录", 2, "Overview");
  const sections = [
    { num: "01", title: "研究背景与动机", desc: "分子性质预测 · 电子密度 · 三大挑战", c: C.primary },
    { num: "02", title: "方法设计", desc: "ImageED · ED教师 · 可靠性评估 · 选择性蒸馏", c: C.secondary },
    { num: "03", title: "实验结果与分析", desc: "QM9 · rMD17 · 消融实验 · 误差分析", c: C.accent },
    { num: "04", title: "总结与展望", desc: "主要贡献 · 局限性 · 未来方向", c: "0E7490" },
  ];
  sections.forEach((sec, i) => {
    const y = 1.15 + i * 1.0;
    s.addShape(pres.shapes.OVAL, { x: 1.2, y: y + 0.08, w: 0.6, h: 0.6, fill: { color: sec.c } });
    s.addText(sec.num, {
      x: 1.2, y: y + 0.08, w: 0.6, h: 0.6,
      fontSize: 17, fontFace: "Georgia", color: C.white, bold: true, align: "center", valign: "middle"
    });
    s.addText(sec.title, {
      x: 2.15, y: y, w: 5, h: 0.38,
      fontSize: 19, fontFace: "Georgia", color: C.textDark, bold: true, align: "left", valign: "middle", margin: 0
    });
    s.addText(sec.desc, {
      x: 2.15, y: y + 0.4, w: 6, h: 0.3,
      fontSize: 11, fontFace: "Calibri", color: C.textMid, align: "left", valign: "middle", margin: 0
    });
  });
}

// ================================================================
// SLIDE 3 — Section 1 Cover
// ================================================================
sectionCover("01", "研究背景与动机", "Background & Motivation", 3);

// ================================================================
// SLIDE 4 — Background
// ================================================================
{
  const s = contentSlide("分子性质预测的重要性与局限", 4, "研究背景与动机");

  // Left card – importance
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.1, w: 4.3, h: 2.0, fill: { color: C.white }, shadow: makeShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.1, w: 4.3, h: 0.06, fill: { color: C.accent } });
  s.addText("分子性质预测的重要性", {
    x: 0.7, y: 1.25, w: 4, h: 0.35,
    fontSize: 14, fontFace: "Calibri", color: C.primary, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "药物发现与材料设计的核心任务", options: { bullet: true, breakLine: true } },
    { text: "传统 DFT 方法：O(N\u00B3)~O(N\u2074) 复杂度", options: { bullet: true, breakLine: true } },
    { text: "机器学习方法：速度快但精度受限", options: { bullet: true } },
  ], {
    x: 0.7, y: 1.7, w: 3.9, h: 1.2,
    fontSize: 12, fontFace: "Calibri", color: C.textDark, align: "left", valign: "top", paraSpaceAfter: 6
  });

  // Right card – limitation
  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.1, w: 4.3, h: 2.0, fill: { color: C.white }, shadow: makeShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.1, w: 4.3, h: 0.06, fill: { color: C.warning } });
  s.addText("现有方法的局限", {
    x: 5.4, y: 1.25, w: 4, h: 0.35,
    fontSize: 14, fontFace: "Calibri", color: C.warning, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "仅利用分子几何结构（原子坐标）", options: { bullet: true, breakLine: true } },
    { text: "忽视电子密度这一关键物理量", options: { bullet: true, breakLine: true } },
    { text: "无法捕获电子级别的化学信息", options: { bullet: true } },
  ], {
    x: 5.4, y: 1.7, w: 3.9, h: 1.2,
    fontSize: 12, fontFace: "Calibri", color: C.textDark, align: "left", valign: "top", paraSpaceAfter: 6
  });

  // Bottom figure
  s.addImage({
    path: fig("introduction/Fig1_introduction\u66F4\u65B0\u540E.png"),
    x: 0.5, y: 3.35, w: 9.0, h: 1.65, sizing: { type: "contain", w: 9.0, h: 1.65 }
  });
}

// ================================================================
// SLIDE 5 — ED Bridge
// ================================================================
{
  const s = contentSlide("电子密度：连接结构与性质的桥梁", 5, "研究背景与动机");

  // HK theorem box
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.1, w: 9.0, h: 0.85, fill: { color: "EFF6FF" }, line: { color: C.primary, width: 1.5 } });
  s.addText([
    { text: "Hohenberg-Kohn 定理：", options: { bold: true, color: C.primary } },
    { text: "基态电子密度 \u03C1(r) 唯一决定体系的所有基态性质", options: { color: C.textDark } },
  ], {
    x: 0.7, y: 1.1, w: 8.6, h: 0.85,
    fontSize: 15, fontFace: "Calibri", align: "center", valign: "middle"
  });

  // Flow: structure → ED → properties
  const fy = 2.25;
  const boxes = [
    { x: 1.0, label: "分子几何结构", col: C.secondary },
    { x: 4.1, label: "电子密度 \u03C1(r)", col: C.accent },
    { x: 7.2, label: "分子性质", col: C.primary },
  ];
  boxes.forEach((b) => {
    s.addShape(pres.shapes.RECTANGLE, { x: b.x, y: fy, w: 2.2, h: 0.75, fill: { color: b.col }, shadow: makeShadow() });
    s.addText(b.label, {
      x: b.x, y: fy, w: 2.2, h: 0.75,
      fontSize: 14, fontFace: "Calibri", color: C.white, bold: true, align: "center", valign: "middle"
    });
  });
  // Arrows
  s.addText("\u2192", { x: 3.3, y: fy, w: 0.7, h: 0.75, fontSize: 28, color: C.accent, align: "center", valign: "middle" });
  s.addText("\u2192", { x: 6.4, y: fy, w: 0.7, h: 0.75, fontSize: 28, color: C.accent, align: "center", valign: "middle" });

  // Core idea card
  s.addShape(pres.shapes.RECTANGLE, { x: 1.5, y: 3.4, w: 7, h: 1.5, fill: { color: C.white }, shadow: makeShadow() });
  s.addText("核心思想：特权信息学习 (LUPI)", {
    x: 1.7, y: 3.45, w: 6.6, h: 0.4,
    fontSize: 15, fontFace: "Calibri", color: C.primary, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "训练阶段", options: { bold: true, color: C.success } },
    { text: "：利用电子密度作为特权信息增强几何模型学习", options: { breakLine: true } },
    { text: "推理阶段", options: { bold: true, color: C.accent } },
    { text: "：仅需几何结构，零额外计算开销", options: {} },
  ], {
    x: 1.7, y: 3.9, w: 6.6, h: 0.9,
    fontSize: 13, fontFace: "Calibri", color: C.textDark, align: "left", valign: "top", paraSpaceAfter: 6
  });
}

// ================================================================
// SLIDE 6 — Three Challenges
// ================================================================
{
  const s = contentSlide("三大核心挑战", 6, "研究背景与动机");

  const chs = [
    { n: "1", t: "计算成本高", d: "电子密度依赖昂贵的\nDFT 计算，点云/体素表示\n随网格指数增长", sol: "多视角 RGB-D\n图像表示", col: C.error },
    { n: "2", t: "模态鸿沟", d: "连续三维标量场 vs\n离散原子坐标图\n表示空间差异巨大", sol: "跨模态\n知识蒸馏", col: C.warning },
    { n: "3", t: "负迁移风险", d: "教师模型的跨模态\n预测并非总是可靠\n少数高误差样本污染梯度", sol: "可靠性感知\n选择性蒸馏", col: C.purple },
  ];
  chs.forEach((ch, i) => {
    const x = 0.6 + i * 3.1;
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.05, w: 2.7, h: 3.9, fill: { color: C.white }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x, y: 1.05, w: 2.7, h: 0.06, fill: { color: ch.col } });
    // Number circle
    s.addShape(pres.shapes.OVAL, { x: x + 0.93, y: 1.3, w: 0.7, h: 0.7, fill: { color: ch.col } });
    s.addText(ch.n, {
      x: x + 0.93, y: 1.3, w: 0.7, h: 0.7,
      fontSize: 22, fontFace: "Georgia", color: C.white, bold: true, align: "center", valign: "middle"
    });
    // Title
    s.addText("挑战：" + ch.t, {
      x: x + 0.1, y: 2.15, w: 2.5, h: 0.35,
      fontSize: 14, fontFace: "Calibri", color: ch.col, bold: true, align: "center", valign: "middle", margin: 0
    });
    // Description
    s.addText(ch.d, {
      x: x + 0.1, y: 2.6, w: 2.5, h: 1.0,
      fontSize: 11, fontFace: "Calibri", color: C.textMid, align: "center", valign: "top"
    });
    // Divider
    s.addShape(pres.shapes.RECTANGLE, { x: x + 0.35, y: 3.7, w: 2.0, h: 0.015, fill: { color: C.border } });
    // Solution label
    s.addText("解决方案", {
      x: x + 0.1, y: 3.8, w: 2.5, h: 0.25,
      fontSize: 10, fontFace: "Calibri", color: C.textLight, align: "center", valign: "middle"
    });
    s.addText(ch.sol, {
      x: x + 0.1, y: 4.1, w: 2.5, h: 0.6,
      fontSize: 12, fontFace: "Calibri", color: C.success, bold: true, align: "center", valign: "top", margin: 0
    });
  });
}

// ================================================================
// SLIDE 7 — Section 2 Cover
// ================================================================
sectionCover("02", "方法设计", "Methodology", 7);

// ================================================================
// SLIDE 8 — EDG++ Overall Framework
// ================================================================
{
  const s = contentSlide("EDG++ 整体框架", 8, "方法设计");
  s.addImage({
    path: fig("our_method/EDG++\u4E3B\u56FE\uFF08\u91CD\u753B\uFF09.png"),
    x: 0.3, y: 0.95, w: 9.4, h: 4.1, sizing: { type: "contain", w: 9.4, h: 4.1 }
  });
}

// ================================================================
// SLIDE 9 — Multi-view RGB-D
// ================================================================
{
  const s = contentSlide("Stage I: 多视角 RGB-D 电子密度图像", 9, "方法设计");

  s.addImage({
    path: fig("introduction/Fig1b_ed_representation.png"),
    x: 0.3, y: 1.05, w: 4.4, h: 3.2, sizing: { type: "contain", w: 4.4, h: 3.2 }
  });

  // Right card
  s.addShape(pres.shapes.RECTANGLE, { x: 5.0, y: 1.05, w: 4.5, h: 4.0, fill: { color: C.white }, shadow: makeShadow() });
  s.addText("设计要点", {
    x: 5.2, y: 1.12, w: 4.1, h: 0.35,
    fontSize: 15, fontFace: "Calibri", color: C.primary, bold: true, align: "left", valign: "middle", margin: 0
  });
  const points = [
    ["6 个标准视角", "消除视角选择偏差"],
    ["RGB 通道 = 静电势映射", "红=正电 / 白=中性 / 蓝=负电"],
    ["D 通道 = 空间深度信息", "保留三维空间布局"],
    ["固定 224\u00D7224 分辨率", "与 ED 网格大小无关"],
    ["GPU 效率提升 42.1%", "相比点云/体素表示"],
  ];
  points.forEach((p, i) => {
    const py = 1.6 + i * 0.62;
    s.addShape(pres.shapes.RECTANGLE, { x: 5.15, y: py, w: 0.06, h: 0.45, fill: { color: i === 4 ? C.success : C.accent } });
    s.addText(p[0], {
      x: 5.35, y: py, w: 3.9, h: 0.25,
      fontSize: 12, fontFace: "Calibri", color: C.textDark, bold: true, align: "left", valign: "middle", margin: 0
    });
    s.addText(p[1], {
      x: 5.35, y: py + 0.23, w: 3.9, h: 0.22,
      fontSize: 10, fontFace: "Calibri", color: C.textMid, align: "left", valign: "middle", margin: 0
    });
  });
}

// ================================================================
// SLIDE 10 — ImageED Feature Learning
// ================================================================
{
  const s = contentSlide("Stage I: ImageED 电子密度特征学习", 10, "方法设计");

  s.addImage({
    path: fig("experiments/ImageED_output.png"),
    x: 0.3, y: 1.0, w: 5.2, h: 3.8, sizing: { type: "contain", w: 5.2, h: 3.8 }
  });

  // Right architecture card
  s.addShape(pres.shapes.RECTANGLE, { x: 5.7, y: 1.0, w: 3.9, h: 3.8, fill: { color: C.white }, shadow: makeShadow() });
  s.addText("架构与训练", {
    x: 5.9, y: 1.08, w: 3.5, h: 0.35,
    fontSize: 15, fontFace: "Calibri", color: C.primary, bold: true, align: "left", valign: "middle", margin: 0
  });

  const items = [
    ["骨干网络", "ViT-Large/16 Masked Autoencoder"],
    ["预训练数据", "200 万分子 (EDBench, DFT B3LYP)"],
    ["自监督任务", "可见重建 (VR) + 掩码重建 (MR)"],
    ["掩码比例", "75% (MAE 标准设定)"],
    ["损失函数", "L = \u03BB_VR\u00B7L_VR + \u03BB_MR\u00B7L_MR"],
    ["输出", "ED 特征表示 F^U"],
  ];
  items.forEach((it, i) => {
    const iy = 1.55 + i * 0.52;
    s.addText(it[0], {
      x: 5.9, y: iy, w: 3.5, h: 0.25,
      fontSize: 11, fontFace: "Calibri", color: C.primary, bold: true, align: "left", valign: "middle", margin: 0
    });
    s.addText(it[1], {
      x: 5.9, y: iy + 0.22, w: 3.5, h: 0.22,
      fontSize: 10, fontFace: "Calibri", color: C.textMid, align: "left", valign: "middle", margin: 0
    });
  });
}

// ================================================================
// SLIDE 11 — ED-aware Teacher
// ================================================================
{
  const s = contentSlide("Stage II: ED-aware 教师预训练", 11, "方法设计");

  // Pipeline boxes
  const y0 = 1.2;
  const pipe = [
    { x: 0.5, w: 2.5, label: "结构图像编码器 f_S\nResNet18 + 视角平均池化", col: C.secondary },
    { x: 3.7, w: 2.5, label: "ED 预测器 f_EDP\n2层MLP: Linear\u2192Softplus\u2192Linear", col: C.accent },
    { x: 6.9, w: 2.6, label: "ED 特征预测\nL_align = L1(F^(S\u2192U), F^U)", col: C.primary },
  ];
  pipe.forEach((p) => {
    s.addShape(pres.shapes.RECTANGLE, { x: p.x, y: y0, w: p.w, h: 1.3, fill: { color: p.col }, shadow: makeShadow() });
    s.addText(p.label, {
      x: p.x, y: y0, w: p.w, h: 1.3,
      fontSize: 12, fontFace: "Calibri", color: C.white, bold: true, align: "center", valign: "middle"
    });
  });
  s.addText("\u2192", { x: 3.1, y: y0, w: 0.5, h: 1.3, fontSize: 24, color: C.accent, align: "center", valign: "middle" });
  s.addText("\u2192", { x: 6.3, y: y0, w: 0.5, h: 1.3, fontSize: 24, color: C.accent, align: "center", valign: "middle" });

  // Left card – physics
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 2.8, w: 4.3, h: 2.15, fill: { color: C.white }, shadow: makeShadow() });
  s.addText("物理依据", {
    x: 0.7, y: 2.85, w: 4, h: 0.3,
    fontSize: 14, fontFace: "Calibri", color: C.primary, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "Born-Oppenheimer 近似", options: { bullet: true, breakLine: true } },
    { text: "  \u2192 电子密度仅依赖核坐标", options: { breakLine: true, color: C.textMid } },
    { text: "Hohenberg-Kohn 定理", options: { bullet: true, breakLine: true } },
    { text: "  \u2192 Markov 性质: S \u2192 ED \u2192 F^U", options: { color: C.textMid } },
  ], {
    x: 0.7, y: 3.2, w: 3.9, h: 1.5,
    fontSize: 11, fontFace: "Calibri", color: C.textDark, align: "left", valign: "top", paraSpaceAfter: 4
  });

  // Right card – advantage
  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 2.8, w: 4.3, h: 2.15, fill: { color: C.white }, shadow: makeShadow() });
  s.addText("关键优势", {
    x: 5.4, y: 2.85, w: 4, h: 0.3,
    fontSize: 14, fontFace: "Calibri", color: C.success, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "推理时无需 DFT 计算", options: { bullet: true, breakLine: true } },
    { text: "仅从结构图像预测 ED 特征", options: { bullet: true, breakLine: true } },
    { text: "消除推理阶段的 ED 依赖", options: { bullet: true, breakLine: true } },
    { text: "与任意几何模型兼容", options: { bullet: true } },
  ], {
    x: 5.4, y: 3.2, w: 3.9, h: 1.5,
    fontSize: 11, fontFace: "Calibri", color: C.textDark, align: "left", valign: "top", paraSpaceAfter: 4
  });
}

// ================================================================
// SLIDE 12 — Why EDG++ (Negative Transfer)
// ================================================================
{
  const s = contentSlide("从 EDG 到 EDG++：负迁移问题", 12, "方法设计");

  // Problem box
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.05, w: 9.0, h: 1.2, fill: { color: "FEF2F2" }, line: { color: C.error, width: 1 } });
  s.addText("问题：朴素蒸馏的假设与现实", {
    x: 0.7, y: 1.1, w: 8.6, h: 0.35,
    fontSize: 14, fontFace: "Calibri", color: C.error, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "朴素蒸馏假设教师预测均匀可靠，但跨模态预测误差 \u03B5_i 是非零均值、重尾分布", options: { breakLine: true } },
    { text: "\u2192 少数高误差样本严重污染梯度信号，导致负迁移", options: { bold: true, color: C.error } },
  ], {
    x: 0.7, y: 1.5, w: 8.6, h: 0.7,
    fontSize: 12, fontFace: "Calibri", color: C.textDark, align: "left", valign: "top"
  });

  // Gradient analysis
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 2.45, w: 4.3, h: 2.5, fill: { color: C.white }, shadow: makeShadow() });
  s.addText("梯度分解分析", {
    x: 0.7, y: 2.5, w: 4, h: 0.3,
    fontSize: 14, fontFace: "Calibri", color: C.primary, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "教师预测：", options: { bold: true, breakLine: true } },
    { text: "F^(S\u2192U)_i = F^U_i + \u03B5_i", options: { breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 4 } },
    { text: "朴素梯度分解：", options: { bold: true, breakLine: true } },
    { text: "\u2207L = g_signal \u2212 g_noise", options: { breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 4 } },
    { text: "噪声项使优化偏离真实 ED 方向", options: { bold: true, color: C.error, breakLine: true } },
    { text: "少数高误差样本贡献大量噪声", options: { bold: true, color: C.error } },
  ], {
    x: 0.7, y: 2.85, w: 3.9, h: 2.0,
    fontSize: 12, fontFace: "Calibri", color: C.textDark, align: "left", valign: "top", paraSpaceAfter: 2
  });

  // Solution
  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 2.45, w: 4.3, h: 2.5, fill: { color: "F0FDF4" }, line: { color: C.success, width: 1 } });
  s.addText("EDG++ 解决思路", {
    x: 5.4, y: 2.5, w: 4, h: 0.3,
    fontSize: 14, fontFace: "Calibri", color: C.success, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "\u2460 预训练可靠性评估器", options: { bold: true, breakLine: true } },
    { text: "   预测教师 ED 特征的重建误差", options: { breakLine: true, color: C.textMid } },
    { text: "", options: { breakLine: true, fontSize: 4 } },
    { text: "\u2461 自适应阈值筛选", options: { bold: true, breakLine: true } },
    { text: "   全局 + 局部混合阈值机制", options: { breakLine: true, color: C.textMid } },
    { text: "", options: { breakLine: true, fontSize: 4 } },
    { text: "\u2462 选择性蒸馏", options: { bold: true, breakLine: true } },
    { text: "   仅从高置信样本学习", options: { breakLine: true, color: C.textMid } },
    { text: "   L^selective = \u03A3 m_i\u00B7SL1(...) / \u03A3 m_i", options: { color: C.textMid } },
  ], {
    x: 5.4, y: 2.85, w: 3.9, h: 2.0,
    fontSize: 12, fontFace: "Calibri", color: C.textDark, align: "left", valign: "top", paraSpaceAfter: 2
  });
}

// ================================================================
// SLIDE 13 — Reliability Estimator + Adaptive Threshold
// ================================================================
{
  const s = contentSlide("可靠性评估器与自适应阈值", 13, "方法设计");

  // Estimator card
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.05, w: 4.4, h: 2.2, fill: { color: C.white }, shadow: makeShadow() });
  s.addText("可靠性评估器", {
    x: 0.7, y: 1.1, w: 4, h: 0.3,
    fontSize: 14, fontFace: "Calibri", color: C.primary, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "结构：2层 MLP (~131K 参数)", options: { bullet: true, breakLine: true } },
    { text: "  Linear(512,256)\u2192Softplus\u2192Linear(256,1)", options: { breakLine: true, color: C.textMid, fontSize: 10 } },
    { text: "输出：置信度 c_i = \u2212\u03C6_eval(F^S)", options: { bullet: true, breakLine: true } },
    { text: "联合训练于 Stage II", options: { bullet: true } },
  ], {
    x: 0.7, y: 1.45, w: 4.0, h: 1.6,
    fontSize: 12, fontFace: "Calibri", color: C.textDark, align: "left", valign: "top", paraSpaceAfter: 4
  });

  // Stop-gradient card
  s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y: 1.05, w: 4.4, h: 2.2, fill: { color: "FFFBEB" }, line: { color: C.warning, width: 1 } });
  s.addText("关键设计：Stop-Gradient", {
    x: 5.3, y: 1.1, w: 4, h: 0.3,
    fontSize: 14, fontFace: "Calibri", color: C.warning, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "L_eval = E[|\u03C6(sg(F^S)) \u2212 sg(||\u03B5_i||)]", options: { bold: true, breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 4 } },
    { text: "输入和目标均 detach", options: { bold: true, color: C.error, breakLine: true } },
    { text: "\u2192 防止教师变得 \u201C可预测误差\u201D", options: { breakLine: true, color: C.textMid } },
    { text: "   而非 \u201C准确预测\u201D", options: { color: C.textMid } },
  ], {
    x: 5.3, y: 1.5, w: 4.0, h: 1.5,
    fontSize: 12, fontFace: "Calibri", color: C.textDark, align: "left", valign: "top", paraSpaceAfter: 2
  });

  // Threshold mechanism
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.45, w: 9.0, h: 1.55, fill: { color: C.white }, shadow: makeShadow() });
  s.addText("自适应阈值机制 (分布无关设计)", {
    x: 0.7, y: 3.5, w: 8.6, h: 0.3,
    fontSize: 14, fontFace: "Calibri", color: C.primary, bold: true, align: "left", valign: "middle", margin: 0
  });

  const ths = [
    { l: "局部阈值", f: "\u03C4_local = \u03BC_B + \u03BA\u00B7\u03C3_B", d: "batch 级别" },
    { l: "全局阈值", f: "\u03C4_global = \u03BC_all + \u03BA\u00B7\u03C3_all", d: "dataset 级别" },
    { l: "混合阈值", f: "\u03C4 = \u03B2\u00B7\u03C4_local + (1\u2212\u03B2)\u00B7\u03C4_global", d: "\u03B2=0.5 最优" },
  ];
  ths.forEach((t, i) => {
    const tx = 0.75 + i * 2.95;
    s.addShape(pres.shapes.RECTANGLE, { x: tx, y: 3.85, w: 2.65, h: 0.95, fill: { color: "F0F9FF" }, line: { color: C.border, width: 0.5 } });
    s.addText(t.l, {
      x: tx, y: 3.88, w: 2.65, h: 0.28,
      fontSize: 11, fontFace: "Calibri", color: C.primary, bold: true, align: "center", valign: "middle"
    });
    s.addText(t.f, {
      x: tx, y: 4.16, w: 2.65, h: 0.28,
      fontSize: 11, fontFace: "Calibri", color: C.textDark, align: "center", valign: "middle"
    });
    s.addText(t.d, {
      x: tx, y: 4.44, w: 2.65, h: 0.25,
      fontSize: 9, fontFace: "Calibri", color: C.textLight, align: "center", valign: "middle"
    });
  });
}

// ================================================================
// SLIDE 14 — Theoretical Analysis
// ================================================================
{
  const s = contentSlide("理论分析：梯度噪声缩减条件", 14, "方法设计");

  // Proposition box
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.05, w: 9.0, h: 2.0, fill: { color: "EFF6FF" }, line: { color: C.primary, width: 1.5 } });
  s.addText("Proposition 1 (Gradient Noise Reduction)", {
    x: 0.7, y: 1.1, w: 8.6, h: 0.35,
    fontSize: 15, fontFace: "Georgia", color: C.primary, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "对 batch B 中排除最大教师误差后的子集 S：", options: { breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 4 } },
    { text: "上界:   E[||g^S_noise||\u00B2] / E[||g^B_noise||\u00B2] \u2264 (|B|/|S|) \u00B7 (r\u0304_S / r\u0304_B)", options: { bold: true, breakLine: true } },
    { text: "", options: { breakLine: true, fontSize: 4 } },
    { text: "充分条件:   r\u0304_S / r\u0304_B < |S| / |B|    \u21D2  选择性蒸馏降低梯度噪声", options: { bold: true, color: C.primary } },
  ], {
    x: 0.7, y: 1.5, w: 8.6, h: 1.45,
    fontSize: 13, fontFace: "Calibri", color: C.textDark, align: "left", valign: "top", paraSpaceAfter: 2
  });

  // Interpretation
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.25, w: 4.3, h: 1.75, fill: { color: C.white }, shadow: makeShadow() });
  s.addText("直观理解", {
    x: 0.7, y: 3.3, w: 4, h: 0.3,
    fontSize: 14, fontFace: "Calibri", color: C.primary, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "误差分布是重尾的", options: { bold: true, breakLine: true } },
    { text: "（少数样本贡献大量噪声）", options: { breakLine: true, color: C.textMid } },
    { text: "", options: { breakLine: true, fontSize: 4 } },
    { text: "去掉这些样本后：", options: { breakLine: true } },
    { text: "噪声缩减幅度 > 样本减少比例", options: { bold: true, color: C.success } },
  ], {
    x: 0.7, y: 3.65, w: 3.9, h: 1.2,
    fontSize: 12, fontFace: "Calibri", color: C.textDark, align: "left", valign: "top", paraSpaceAfter: 2
  });

  // Assumptions
  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 3.25, w: 4.3, h: 1.75, fill: { color: C.white }, shadow: makeShadow() });
  s.addText("假设条件", {
    x: 5.4, y: 3.3, w: 4, h: 0.3,
    fontSize: 14, fontFace: "Calibri", color: C.textMid, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "||\u03B5_i|| 与 ||J_i||_F 独立", options: { bullet: true, breakLine: true } },
    { text: "Jacobian 范数有界", options: { bullet: true, breakLine: true } },
    { text: "误差分布具有重尾特征", options: { bullet: true, breakLine: true } },
    { text: "（实验中通过尾部分析间接验证）", options: { color: C.textMid, fontSize: 10 } },
  ], {
    x: 5.4, y: 3.65, w: 3.9, h: 1.2,
    fontSize: 12, fontFace: "Calibri", color: C.textDark, align: "left", valign: "top", paraSpaceAfter: 3
  });
}

// ================================================================
// SLIDE 15 — Zero Overhead Inference
// ================================================================
{
  const s = contentSlide("推理阶段：零额外开销", 15, "方法设计");

  // Training column
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.05, w: 4.3, h: 3.0, fill: { color: C.white }, shadow: makeShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.05, w: 4.3, h: 0.5, fill: { color: C.secondary } });
  s.addText("训练阶段", {
    x: 0.5, y: 1.05, w: 4.3, h: 0.5,
    fontSize: 16, fontFace: "Calibri", color: C.white, bold: true, align: "center", valign: "middle"
  });
  const trainItems = [
    { name: "\u2713  几何学生 f_G", keep: true },
    { name: "\u2713  任务预测器 f_T", keep: true },
    { name: "\u2713  ED 教师模型", keep: false },
    { name: "\u2713  可靠性评估器", keep: false },
    { name: "\u2713  选择性蒸馏模块", keep: false },
  ];
  trainItems.forEach((it, i) => {
    const iy = 1.7 + i * 0.45;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.8, y: iy, w: 3.7, h: 0.38,
      fill: { color: it.keep ? "F0FDF4" : "F0F9FF" },
      line: { color: it.keep ? C.success : C.secondary, width: 0.5 }
    });
    s.addText(it.name, {
      x: 0.8, y: iy, w: 3.7, h: 0.38,
      fontSize: 12, fontFace: "Calibri", color: C.textDark, align: "center", valign: "middle"
    });
  });

  // Arrow
  s.addText("\u2192", {
    x: 4.5, y: 2.0, w: 1, h: 1,
    fontSize: 40, color: C.accent, align: "center", valign: "middle"
  });

  // Inference column
  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.05, w: 4.3, h: 3.0, fill: { color: C.white }, shadow: makeShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.05, w: 4.3, h: 0.5, fill: { color: C.success } });
  s.addText("推理阶段", {
    x: 5.2, y: 1.05, w: 4.3, h: 0.5,
    fontSize: 16, fontFace: "Calibri", color: C.white, bold: true, align: "center", valign: "middle"
  });
  const inferItems = [
    { name: "\u2713  几何学生 f_G", keep: true },
    { name: "\u2713  任务预测器 f_T", keep: true },
    { name: "\u2717  ED 教师模型", keep: false },
    { name: "\u2717  可靠性评估器", keep: false },
    { name: "\u2717  选择性蒸馏模块", keep: false },
  ];
  inferItems.forEach((it, i) => {
    const iy = 1.7 + i * 0.45;
    s.addShape(pres.shapes.RECTANGLE, {
      x: 5.5, y: iy, w: 3.7, h: 0.38,
      fill: { color: it.keep ? "F0FDF4" : "FEF2F2" },
      line: { color: it.keep ? C.success : C.error, width: 0.5 }
    });
    s.addText(it.name, {
      x: 5.5, y: iy, w: 3.7, h: 0.38,
      fontSize: 12, fontFace: "Calibri", color: it.keep ? C.textDark : C.error, align: "center", valign: "middle"
    });
  });

  // Summary takeaway
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.25, w: 9.0, h: 0.75, fill: { color: "F0FDF4" }, line: { color: C.success, width: 0.5 } });
  s.addText([
    { text: "核心优势：", options: { bold: true, color: C.success } },
    { text: "EDG++ 的所有额外组件（教师、评估器、蒸馏）仅在训练时使用。推理阶段与基线模型完全相同 \u2014 相同速度、相同内存、更高精度。", options: {} },
  ], {
    x: 0.7, y: 4.25, w: 8.6, h: 0.75,
    fontSize: 12, fontFace: "Calibri", color: C.textDark, align: "left", valign: "middle"
  });
}

// ================================================================
// SLIDE 16 — Section 3 Cover
// ================================================================
sectionCover("03", "实验结果与分析", "Experiments & Analysis", 16);

// ================================================================
// SLIDE 17 — Experimental Setup
// ================================================================
{
  const s = contentSlide("实验设置", 17, "实验结果与分析");

  // Pre-training data banner
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.05, w: 9.0, h: 0.65, fill: { color: "F0F9FF" }, line: { color: C.primary, width: 0.5 } });
  s.addText([
    { text: "预训练数据：", options: { bold: true } },
    { text: "200 万无标签分子构象 (EDBench, DFT B3LYP/6-31G(d,p), grid spacing 0.4 Bohr)", options: {} },
  ], {
    x: 0.7, y: 1.05, w: 8.6, h: 0.65,
    fontSize: 12, fontFace: "Calibri", color: C.textDark, align: "left", valign: "middle"
  });

  // QM9 card
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.9, w: 4.3, h: 3.1, fill: { color: C.white }, shadow: makeShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.9, w: 4.3, h: 0.06, fill: { color: C.accent } });
  s.addText("QM9 数据集", {
    x: 0.7, y: 2.0, w: 4, h: 0.35,
    fontSize: 15, fontFace: "Calibri", color: C.primary, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "12 个量子化学性质", options: { bullet: true, breakLine: true } },
    { text: "\u03B1, \u0394\u03B5, \u03B5_HOMO, \u03B5_LUMO, \u03BC, C_v, G, H, R\u00B2, U, U\u2080, ZPVE", options: { breakLine: true, color: C.textMid, fontSize: 9 } },
    { text: "划分：110K / 10K / 11K (train/val/test)", options: { bullet: true, breakLine: true } },
    { text: "3 个架构 \u00D7 12 性质 = 36 任务", options: { bullet: true, breakLine: true } },
    { text: "评估指标：Mean Absolute Error (MAE)", options: { bullet: true, breakLine: true } },
    { text: "基线：SchNet, Equiformer, SphereNet", options: { bullet: true } },
  ], {
    x: 0.7, y: 2.4, w: 3.9, h: 2.4,
    fontSize: 11, fontFace: "Calibri", color: C.textDark, align: "left", valign: "top", paraSpaceAfter: 4
  });

  // rMD17 card
  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.9, w: 4.3, h: 3.1, fill: { color: C.white }, shadow: makeShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 1.9, w: 4.3, h: 0.06, fill: { color: C.secondary } });
  s.addText("rMD17 数据集", {
    x: 5.4, y: 2.0, w: 4, h: 0.35,
    fontSize: 15, fontFace: "Calibri", color: C.primary, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "10 个有机分子", options: { bullet: true, breakLine: true } },
    { text: "Aspirin, Benzene, Ethanol, Toluene, ...", options: { breakLine: true, color: C.textMid, fontSize: 9 } },
    { text: "划分：950 / 50 / 1000 (train/val/test)", options: { bullet: true, breakLine: true } },
    { text: "20 任务 (10 能量 + 10 力预测)", options: { bullet: true, breakLine: true } },
    { text: "力 = \u2212\u2207(能量), 自动微分计算", options: { bullet: true, breakLine: true } },
    { text: "基线：SchNet, ViSNet, SphereNet", options: { bullet: true } },
  ], {
    x: 5.4, y: 2.4, w: 3.9, h: 2.4,
    fontSize: 11, fontFace: "Calibri", color: C.textDark, align: "left", valign: "top", paraSpaceAfter: 4
  });
}

// ================================================================
// SLIDE 18 — QM9 Main Results
// ================================================================
{
  const s = contentSlide("QM9：34/36 任务提升", 18, "实验结果与分析");

  // Stat callouts
  const stats = [
    { v: "34/36", l: "任务提升", d: "sign test p < 10\u207B\u2078", col: C.accent },
    { v: "+3.2%", l: "SchNet", d: "平均相对提升", col: C.primary },
    { v: "+9.0%", l: "Equiformer", d: "平均相对提升", col: C.primary },
    { v: "+3.6%", l: "SphereNet", d: "平均相对提升", col: C.primary },
  ];
  stats.forEach((st, i) => {
    const sx = 0.5 + i * 2.35;
    s.addShape(pres.shapes.RECTANGLE, { x: sx, y: 1.0, w: 2.1, h: 1.35, fill: { color: C.white }, shadow: makeShadow() });
    s.addText(st.v, {
      x: sx, y: 1.02, w: 2.1, h: 0.55,
      fontSize: 28, fontFace: "Georgia", color: st.col, bold: true, align: "center", valign: "middle"
    });
    s.addText(st.l, {
      x: sx, y: 1.55, w: 2.1, h: 0.3,
      fontSize: 13, fontFace: "Calibri", color: C.textDark, bold: true, align: "center", valign: "middle"
    });
    s.addText(st.d, {
      x: sx, y: 1.83, w: 2.1, h: 0.25,
      fontSize: 9, fontFace: "Calibri", color: C.textLight, align: "center", valign: "middle"
    });
  });

  // EDG++ vs EDG note
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 2.5, w: 9.0, h: 0.45, fill: { color: "F0FDF4" }, line: { color: C.success, width: 0.5 } });
  s.addText([
    { text: "EDG++ 优于 EDG：", options: { bold: true, color: C.success } },
    { text: "SchNet 10/12，SphereNet 7/12 \u2014 选择性蒸馏有效缓解负迁移", options: {} },
  ], {
    x: 0.7, y: 2.5, w: 8.6, h: 0.45,
    fontSize: 12, fontFace: "Calibri", color: C.textDark, align: "left", valign: "middle"
  });

  // Property group improvement figure
  s.addImage({
    path: fig("experiments/property_group_improvement.png"),
    x: 0.5, y: 3.1, w: 9.0, h: 1.9, sizing: { type: "contain", w: 9.0, h: 1.9 }
  });
}

// ================================================================
// SLIDE 19 — Property Group Analysis
// ================================================================
{
  const s = contentSlide("性质分组提升：与物理原理一致", 19, "实验结果与分析");

  const groups = [
    { n: "热力学性质 (U, U\u2080, H, G)", g: "+3.70% ~ +16.30%", r: "HK 定理：基态可观测量由 ED 唯一决定", c: C.success, w: "5/5 全胜" },
    { n: "电子性质 (\u03B5_HOMO, \u03B5_LUMO)", g: "+1.8% ~ +5.9%", r: "轨道能量与电子密度强相关", c: C.accent, w: "一致提升" },
    { n: "其他性质 (\u03B1, \u03BC, ZPVE)", g: "小幅/混合", r: "受多种物理因素影响，ED 贡献有限", c: C.textLight, w: "部分提升" },
  ];
  groups.forEach((g, i) => {
    const gy = 1.1 + i * 1.2;
    s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: gy, w: 9.0, h: 1.0, fill: { color: C.white }, shadow: makeShadow() });
    s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: gy, w: 0.08, h: 1.0, fill: { color: g.c } });
    s.addText(g.n, {
      x: 0.8, y: gy + 0.05, w: 5, h: 0.35,
      fontSize: 14, fontFace: "Calibri", color: C.textDark, bold: true, align: "left", valign: "middle", margin: 0
    });
    s.addText(g.g, {
      x: 7.5, y: gy + 0.05, w: 1.8, h: 0.35,
      fontSize: 14, fontFace: "Georgia", color: g.c, bold: true, align: "right", valign: "middle", margin: 0
    });
    s.addText(g.r, {
      x: 0.8, y: gy + 0.42, w: 6.5, h: 0.3,
      fontSize: 11, fontFace: "Calibri", color: C.textMid, align: "left", valign: "middle", margin: 0
    });
    s.addText(g.w, {
      x: 7.5, y: gy + 0.42, w: 1.8, h: 0.3,
      fontSize: 11, fontFace: "Calibri", color: C.textMid, align: "right", valign: "middle", margin: 0
    });
  });

  // Note
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.5, w: 9.0, h: 0.5, fill: { color: "FFFBEB" }, line: { color: C.warning, width: 0.5 } });
  s.addText([
    { text: "仅 2 个退化：", options: { bold: true } },
    { text: "Equiformer \u03BC (-1.9%), ZPVE (-3.2%)，但在其他架构上均有提升", options: {} },
  ], {
    x: 0.7, y: 4.5, w: 8.6, h: 0.5,
    fontSize: 12, fontFace: "Calibri", color: C.textDark, align: "left", valign: "middle"
  });
}

// ================================================================
// SLIDE 20 — rMD17 Results
// ================================================================
{
  const s = contentSlide("rMD17：能量与力预测结果", 20, "实验结果与分析");

  s.addImage({
    path: fig("experiments/rMD17_baseline_vs_edgpp.png"),
    x: 0.2, y: 0.95, w: 5.2, h: 4.1, sizing: { type: "contain", w: 5.2, h: 4.1 }
  });

  // Energy card
  s.addShape(pres.shapes.RECTANGLE, { x: 5.5, y: 1.0, w: 4.1, h: 1.7, fill: { color: C.white }, shadow: makeShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.5, y: 1.0, w: 4.1, h: 0.06, fill: { color: C.accent } });
  s.addText("能量预测", {
    x: 5.7, y: 1.1, w: 3.7, h: 0.3,
    fontSize: 14, fontFace: "Calibri", color: C.primary, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "平均提升 25.2% ~ 33.7%", options: { bold: true, color: C.success, breakLine: true } },
    { text: "EDG++ SphereNet: 7/10 最佳", options: { breakLine: true } },
    { text: "Aspirin \u221231.6%, Salicylic acid \u221231.4%", options: { color: C.textMid, fontSize: 10 } },
  ], {
    x: 5.7, y: 1.45, w: 3.7, h: 1.1,
    fontSize: 12, fontFace: "Calibri", color: C.textDark, align: "left", valign: "top", paraSpaceAfter: 4
  });

  // Force card
  s.addShape(pres.shapes.RECTANGLE, { x: 5.5, y: 2.9, w: 4.1, h: 1.7, fill: { color: C.white }, shadow: makeShadow() });
  s.addShape(pres.shapes.RECTANGLE, { x: 5.5, y: 2.9, w: 4.1, h: 0.06, fill: { color: C.secondary } });
  s.addText("力预测", {
    x: 5.7, y: 3.0, w: 3.7, h: 0.3,
    fontSize: 14, fontFace: "Calibri", color: C.primary, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "提升 1.5% ~ 5.3%（较温和）", options: { breakLine: true } },
    { text: "EDG++ SphereNet: 6/10 最佳", options: { breakLine: true } },
    { text: "力依赖局部原子交互，受全局 ED 影响较小", options: { color: C.textMid, fontSize: 10 } },
  ], {
    x: 5.7, y: 3.35, w: 3.7, h: 1.1,
    fontSize: 12, fontFace: "Calibri", color: C.textDark, align: "left", valign: "top", paraSpaceAfter: 4
  });
}

// ================================================================
// SLIDE 21 — Ablation Studies
// ================================================================
{
  const s = contentSlide("样本难度分层与超参数消融", 21, "实验结果与分析");

  // Difficulty stratification figure (Fig. S1)
  s.addImage({
    path: fig("experiments/difficulty_stratification_lines.png"),
    x: 0.2, y: 0.95, w: 4.9, h: 2.1, sizing: { type: "contain", w: 4.9, h: 2.1 }
  });
  // Heatmap figure (Fig. S3)
  s.addImage({
    path: fig("experiments/heatmap_hyperparam.png"),
    x: 5.1, y: 0.95, w: 4.6, h: 2.1, sizing: { type: "contain", w: 4.6, h: 2.1 }
  });

  // Left findings
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.2, w: 4.3, h: 1.8, fill: { color: C.white }, shadow: makeShadow() });
  s.addText("样本难度分层分析", {
    x: 0.7, y: 3.25, w: 4, h: 0.3,
    fontSize: 13, fontFace: "Calibri", color: C.primary, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "简单样本 (P0-25)：EDG++ 无优势", options: { bullet: true, breakLine: true } },
    { text: "困难样本 (P95-100)：误差降低 27-38%", options: { bullet: true, breakLine: true, bold: true, color: C.success } },
    { text: "越难的样本受益越大 \u2192 支撑重尾理论", options: { bullet: true } },
  ], {
    x: 0.7, y: 3.6, w: 3.9, h: 1.2,
    fontSize: 11, fontFace: "Calibri", color: C.textDark, align: "left", valign: "top", paraSpaceAfter: 5
  });

  // Right findings
  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 3.2, w: 4.3, h: 1.8, fill: { color: C.white }, shadow: makeShadow() });
  s.addText("\u03BA 调优增量收益", {
    x: 5.4, y: 3.25, w: 4, h: 0.3,
    fontSize: 13, fontFace: "Calibri", color: C.primary, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "默认 \u03BA=0 已捕获 80-90% 收益", options: { bullet: true, breakLine: true } },
    { text: "SchNet: +0.97%, max +2.46% (R\u00B2)", options: { bullet: true, breakLine: true } },
    { text: "Equiformer: +2.87%, max +12.82% (H)", options: { bullet: true } },
  ], {
    x: 5.4, y: 3.6, w: 3.9, h: 1.2,
    fontSize: 11, fontFace: "Calibri", color: C.textDark, align: "left", valign: "top", paraSpaceAfter: 5
  });
}

// ================================================================
// SLIDE 22 — Robustness & Tail Error
// ================================================================
{
  const s = contentSlide("\u03BB 灵敏度与尾部误差分析", 22, "实验结果与分析");

  // Lambda sensitivity figure
  s.addImage({
    path: fig("experiments/lambda_ED.png"),
    x: 0.2, y: 0.95, w: 4.9, h: 2.1, sizing: { type: "contain", w: 4.9, h: 2.1 }
  });
  // Tail ECDF figure (Fig. 4)
  s.addImage({
    path: fig("experiments/tail_error_ecdf.png"),
    x: 5.1, y: 0.95, w: 4.6, h: 2.1, sizing: { type: "contain", w: 4.6, h: 2.1 }
  });

  // Lambda sensitivity card
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.2, w: 4.3, h: 1.8, fill: { color: C.white }, shadow: makeShadow() });
  s.addText("蒸馏权重 \u03BB 灵敏度", {
    x: 0.7, y: 3.25, w: 4, h: 0.3,
    fontSize: 13, fontFace: "Calibri", color: C.primary, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "EGNN/QM9: 性能随 \u03BB 变化平稳", options: { bullet: true, breakLine: true } },
    { text: "ViSNet/rMD17: 各分子表现稳定", options: { bullet: true, breakLine: true } },
    { text: "宽 \u03BB 范围内均有效 (10\u207B\u2074~1.0)", options: { bullet: true, breakLine: true } },
    { text: "默认配置已捕获大部分收益", options: { bullet: true, color: C.success, bold: true } },
  ], {
    x: 0.7, y: 3.6, w: 3.9, h: 1.2,
    fontSize: 11, fontFace: "Calibri", color: C.textDark, align: "left", valign: "top", paraSpaceAfter: 3
  });

  // Tail error card
  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 3.2, w: 4.3, h: 1.8, fill: { color: "F0FDF4" }, line: { color: C.success, width: 0.5 } });
  s.addText("尾部误差缩减", {
    x: 5.4, y: 3.25, w: 4, h: 0.3,
    fontSize: 13, fontFace: "Calibri", color: C.success, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "最难 1% 样本：误差降低 27-38%", options: { bold: true, color: C.success, breakLine: true } },
    { text: "跨架构、跨性质一致", options: { bullet: true, breakLine: true } },
    { text: "间接验证重尾分布假设", options: { bullet: true, breakLine: true } },
    { text: "支撑 Proposition 1 的理论预测", options: { bullet: true } },
  ], {
    x: 5.4, y: 3.6, w: 3.9, h: 1.2,
    fontSize: 11, fontFace: "Calibri", color: C.textDark, align: "left", valign: "top", paraSpaceAfter: 3
  });
}

// ================================================================
// SLIDE 23 — Contributions & Summary
// ================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.darkBg };
  // Title
  s.addText("主要贡献与总结", {
    x: 1, y: 0.3, w: 8, h: 0.6,
    fontSize: 28, fontFace: "Georgia", color: C.white, bold: true, align: "center", valign: "middle"
  });

  // Three contribution cards
  const contribs = [
    { n: "1", t: "特权信息学习范式", d: "将电子密度作为特权信息\n训练时用，推理时不需要\n零额外推理开销" },
    { n: "2", t: "EDG 跨模态蒸馏框架", d: "多视角 RGB-D 图像表示\nImageED (200万分子预训练)\n架构无关，通用性强" },
    { n: "3", t: "EDG++ 选择性蒸馏", d: "可靠性感知筛选机制\n自适应全局-局部混合阈值\n理论保证 (Proposition 1)" },
  ];
  contribs.forEach((c, i) => {
    const cx = 0.5 + i * 3.15;
    s.addShape(pres.shapes.RECTANGLE, { x: cx, y: 1.1, w: 2.85, h: 2.3, fill: { color: "0D3356" }, shadow: makeCardShadow() });
    // Number
    s.addShape(pres.shapes.OVAL, { x: cx + 1.0, y: 1.2, w: 0.65, h: 0.65, fill: { color: C.accent } });
    s.addText(c.n, {
      x: cx + 1.0, y: 1.2, w: 0.65, h: 0.65,
      fontSize: 20, fontFace: "Georgia", color: C.white, bold: true, align: "center", valign: "middle"
    });
    s.addText(c.t, {
      x: cx + 0.15, y: 1.95, w: 2.55, h: 0.35,
      fontSize: 13, fontFace: "Calibri", color: C.accent, bold: true, align: "center", valign: "middle"
    });
    s.addText(c.d, {
      x: cx + 0.15, y: 2.3, w: 2.55, h: 0.95,
      fontSize: 11, fontFace: "Calibri", color: "CADCFC", align: "center", valign: "top"
    });
  });

  // Limitations & future
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.6, w: 4.3, h: 1.35, fill: { color: "0D3356" } });
  s.addText("局限性", {
    x: 0.7, y: 3.65, w: 4, h: 0.25,
    fontSize: 12, fontFace: "Calibri", color: C.warning, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "单种子实验 (seed=42)", options: { bullet: true, breakLine: true } },
    { text: "per-task-best 为上界", options: { bullet: true, breakLine: true } },
    { text: "rMD17 仅 SphereNet 测试 EDG++", options: { bullet: true } },
  ], {
    x: 0.7, y: 3.92, w: 3.9, h: 0.9,
    fontSize: 10, fontFace: "Calibri", color: "94A3B8", align: "left", valign: "top", paraSpaceAfter: 3
  });

  s.addShape(pres.shapes.RECTANGLE, { x: 5.2, y: 3.6, w: 4.3, h: 1.35, fill: { color: "0D3356" } });
  s.addText("未来方向", {
    x: 5.4, y: 3.65, w: 4, h: 0.25,
    fontSize: 12, fontFace: "Calibri", color: C.success, bold: true, align: "left", valign: "middle", margin: 0
  });
  s.addText([
    { text: "软加权替代硬阈值", options: { bullet: true, breakLine: true } },
    { text: "扩展至分子轨道、光谱等特权信息", options: { bullet: true, breakLine: true } },
    { text: "联合微调可靠性评估器", options: { bullet: true } },
  ], {
    x: 5.4, y: 3.92, w: 3.9, h: 0.9,
    fontSize: 10, fontFace: "Calibri", color: "94A3B8", align: "left", valign: "top", paraSpaceAfter: 3
  });

  addBottomBar(s, 23);
  addSectionTag(s, "总结与展望");
}

// ================================================================
// SLIDE 24 — Thank You
// ================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.darkBg };
  // Top accent
  s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 10, h: 0.06, fill: { color: C.accent } });

  s.addText("感谢聆听", {
    x: 1, y: 1.7, w: 8, h: 0.9,
    fontSize: 44, fontFace: "Georgia", color: C.white, bold: true, align: "center", valign: "middle"
  });
  s.addText("请各位老师批评指正", {
    x: 1, y: 2.7, w: 8, h: 0.5,
    fontSize: 18, fontFace: "Calibri", color: "CADCFC", align: "center", valign: "middle"
  });

  // Divider
  s.addShape(pres.shapes.RECTANGLE, { x: 3.5, y: 3.5, w: 3, h: 0.02, fill: { color: C.accent } });

  // Code link
  s.addText("Code: https://github.com/HongxinXiang/EDG", {
    x: 1, y: 3.75, w: 8, h: 0.35,
    fontSize: 12, fontFace: "Calibri", color: C.accent, align: "center", valign: "middle"
  });

  s.addText("Hunan University  \u00B7  Westlake University  \u00B7  USTC", {
    x: 1, y: 4.2, w: 8, h: 0.35,
    fontSize: 11, fontFace: "Calibri", color: "64748B", align: "center", valign: "middle"
  });

  addBottomBar(s, 24);
}

// ================================================================
// SLIDE 25 — Appendix: QM9 Full Table
// ================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  addSlideTitle(s, "附录：QM9 完整实验结果");
  addBottomBar(s, 25);
  s.addText("附录", {
    x: 0.3, y: 5.225, w: 3, h: 0.4,
    fontSize: 9, color: "CADCFC", align: "left", valign: "middle", fontFace: "Calibri", italic: true
  });

  const headerOpts = { fill: { color: C.primary }, color: C.white, bold: true, fontSize: 7, fontFace: "Calibri", align: "center", valign: "middle", margin: [1,2,1,2] };
  const groupOpts = { fill: { color: "E8EDF2" }, bold: true, fontSize: 7, fontFace: "Calibri", align: "center", valign: "middle", margin: [1,2,1,2] };
  const cellOpts = { fontSize: 7, fontFace: "Calibri", align: "center", valign: "middle", margin: [1,2,1,2] };
  const bestOpts = { ...cellOpts, bold: true, color: C.primary };
  const deltaUp = { ...cellOpts, color: C.success, bold: true };
  const deltaDn = { ...cellOpts, color: C.error, bold: true };

  const qm9Rows = [
    // Header
    [
      { text: "", options: headerOpts },
      { text: "Method", options: headerOpts },
      { text: "\u03B1\n(a\u2080\u00B3)", options: headerOpts },
      { text: "\u0394\u03B5\n(meV)", options: headerOpts },
      { text: "\u03B5_HOMO\n(meV)", options: headerOpts },
      { text: "\u03B5_LUMO\n(meV)", options: headerOpts },
      { text: "\u03BC\n(D)", options: headerOpts },
      { text: "C_v\n(cal/mol\u00B7K)", options: headerOpts },
      { text: "G\n(meV)", options: headerOpts },
      { text: "H\n(meV)", options: headerOpts },
      { text: "R\u00B2\n(a\u2080\u00B2)", options: headerOpts },
      { text: "U\n(meV)", options: headerOpts },
      { text: "U\u2080\n(meV)", options: headerOpts },
      { text: "ZPVE\n(meV)", options: headerOpts },
    ],
    // SchNet
    [{ text: "SchNet", options: { ...groupOpts, rowSpan: 4 } }, { text: "Baseline", options: cellOpts }, { text: "0.0702", options: cellOpts }, { text: "50.83", options: cellOpts }, { text: "31.95", options: cellOpts }, { text: "26.17", options: cellOpts }, { text: "0.0301", options: cellOpts }, { text: "0.0323", options: cellOpts }, { text: "14.68", options: cellOpts }, { text: "14.09", options: cellOpts }, { text: "0.1346", options: cellOpts }, { text: "14.14", options: cellOpts }, { text: "13.92", options: cellOpts }, { text: "1.714", options: cellOpts }],
    [{ text: "", options: groupOpts }, { text: "EDG", options: cellOpts }, { text: "0.0687", options: cellOpts }, { text: "49.78", options: cellOpts }, { text: "31.88", options: cellOpts }, { text: "25.97", options: cellOpts }, { text: "0.0298", options: bestOpts }, { text: "0.0316", options: cellOpts }, { text: "14.02", options: cellOpts }, { text: "13.84", options: cellOpts }, { text: "0.1246", options: bestOpts }, { text: "13.79", options: cellOpts }, { text: "13.83", options: cellOpts }, { text: "1.688", options: cellOpts }],
    [{ text: "", options: groupOpts }, { text: "EDG++", options: bestOpts }, { text: "0.0681", options: bestOpts }, { text: "49.43", options: bestOpts }, { text: "31.39", options: bestOpts }, { text: "25.50", options: bestOpts }, { text: "0.0298", options: cellOpts }, { text: "0.0315", options: bestOpts }, { text: "13.73", options: bestOpts }, { text: "13.77", options: bestOpts }, { text: "0.1259", options: cellOpts }, { text: "13.61", options: bestOpts }, { text: "13.42", options: bestOpts }, { text: "1.674", options: bestOpts }],
    [{ text: "", options: groupOpts }, { text: "\u0394", options: deltaUp }, { text: "\u21913.0%", options: deltaUp }, { text: "\u21912.8%", options: deltaUp }, { text: "\u21911.8%", options: deltaUp }, { text: "\u21912.6%", options: deltaUp }, { text: "\u21911.0%", options: deltaUp }, { text: "\u21912.5%", options: deltaUp }, { text: "\u21916.5%", options: deltaUp }, { text: "\u21912.2%", options: deltaUp }, { text: "\u21916.4%", options: deltaUp }, { text: "\u21913.7%", options: deltaUp }, { text: "\u21913.6%", options: deltaUp }, { text: "\u21912.3%", options: deltaUp }],
    // Equiformer
    [{ text: "Equiformer", options: { ...groupOpts, rowSpan: 4 } }, { text: "Baseline", options: cellOpts }, { text: "0.0676", options: cellOpts }, { text: "46.31", options: cellOpts }, { text: "26.02", options: cellOpts }, { text: "23.68", options: cellOpts }, { text: "0.0207", options: cellOpts }, { text: "0.0273", options: cellOpts }, { text: "18.44", options: cellOpts }, { text: "16.45", options: cellOpts }, { text: "0.4583", options: cellOpts }, { text: "15.34", options: cellOpts }, { text: "23.93", options: cellOpts }, { text: "1.537", options: cellOpts }],
    [{ text: "", options: groupOpts }, { text: "EDG", options: cellOpts }, { text: "0.0648", options: cellOpts }, { text: "45.81", options: cellOpts }, { text: "25.49", options: cellOpts }, { text: "23.27", options: cellOpts }, { text: "0.0199", options: bestOpts }, { text: "0.0264", options: cellOpts }, { text: "15.98", options: cellOpts }, { text: "14.45", options: bestOpts }, { text: "0.4395", options: cellOpts }, { text: "15.47", options: cellOpts }, { text: "16.52", options: bestOpts }, { text: "1.529", options: bestOpts }],
    [{ text: "", options: groupOpts }, { text: "EDG++", options: bestOpts }, { text: "0.0618", options: bestOpts }, { text: "45.09", options: bestOpts }, { text: "25.27", options: bestOpts }, { text: "22.28", options: bestOpts }, { text: "0.0211", options: cellOpts }, { text: "0.0264", options: bestOpts }, { text: "12.06", options: bestOpts }, { text: "14.71", options: cellOpts }, { text: "0.4035", options: bestOpts }, { text: "14.80", options: bestOpts }, { text: "16.92", options: cellOpts }, { text: "1.586", options: cellOpts }],
    [{ text: "", options: groupOpts }, { text: "\u0394", options: deltaUp }, { text: "\u21918.7%", options: deltaUp }, { text: "\u21912.6%", options: deltaUp }, { text: "\u21912.9%", options: deltaUp }, { text: "\u21915.9%", options: deltaUp }, { text: "\u2191-1.9%", options: deltaDn }, { text: "\u21913.5%", options: deltaUp }, { text: "\u219134.6%", options: deltaUp }, { text: "\u219110.6%", options: deltaUp }, { text: "\u219111.9%", options: deltaUp }, { text: "\u21913.5%", options: deltaUp }, { text: "\u219129.3%", options: deltaUp }, { text: "\u2191-3.2%", options: deltaDn }],
    // SphereNet
    [{ text: "SphereNet", options: { ...groupOpts, rowSpan: 4 } }, { text: "Baseline", options: cellOpts }, { text: "0.0467", options: cellOpts }, { text: "40.13", options: cellOpts }, { text: "22.01", options: cellOpts }, { text: "19.44", options: cellOpts }, { text: "0.0269", options: cellOpts }, { text: "0.0244", options: cellOpts }, { text: "7.875", options: cellOpts }, { text: "7.199", options: cellOpts }, { text: "0.2582", options: cellOpts }, { text: "6.999", options: cellOpts }, { text: "6.641", options: cellOpts }, { text: "1.253", options: cellOpts }],
    [{ text: "", options: groupOpts }, { text: "EDG", options: cellOpts }, { text: "0.0459", options: cellOpts }, { text: "39.69", options: cellOpts }, { text: "21.84", options: cellOpts }, { text: "19.01", options: cellOpts }, { text: "0.0265", options: bestOpts }, { text: "0.0238", options: bestOpts }, { text: "7.769", options: cellOpts }, { text: "6.283", options: bestOpts }, { text: "0.2494", options: bestOpts }, { text: "6.502", options: cellOpts }, { text: "6.101", options: bestOpts }, { text: "1.206", options: cellOpts }],
    [{ text: "", options: groupOpts }, { text: "EDG++", options: bestOpts }, { text: "0.0457", options: bestOpts }, { text: "39.47", options: bestOpts }, { text: "21.77", options: bestOpts }, { text: "18.75", options: bestOpts }, { text: "0.0265", options: cellOpts }, { text: "0.0240", options: cellOpts }, { text: "7.656", options: bestOpts }, { text: "6.364", options: cellOpts }, { text: "0.2534", options: cellOpts }, { text: "6.409", options: bestOpts }, { text: "6.424", options: cellOpts }, { text: "1.198", options: bestOpts }],
    [{ text: "", options: groupOpts }, { text: "\u0394", options: deltaUp }, { text: "\u21912.2%", options: deltaUp }, { text: "\u21911.6%", options: deltaUp }, { text: "\u21911.1%", options: deltaUp }, { text: "\u21913.5%", options: deltaUp }, { text: "\u21911.4%", options: deltaUp }, { text: "\u21911.5%", options: deltaUp }, { text: "\u21912.8%", options: deltaUp }, { text: "\u219111.6%", options: deltaUp }, { text: "\u21911.9%", options: deltaUp }, { text: "\u21918.4%", options: deltaUp }, { text: "\u21913.3%", options: deltaUp }, { text: "\u21914.4%", options: deltaUp }],
  ];

  const colW = [0.7, 0.55, 0.6, 0.6, 0.63, 0.63, 0.58, 0.72, 0.55, 0.55, 0.55, 0.55, 0.55, 0.55];
  s.addTable(qm9Rows, {
    x: 0.25, y: 0.95, colW: colW,
    border: { type: "solid", pt: 0.3, color: C.border },
    rowH: 0.28
  });

  s.addText("Avg. \u0394: SchNet \u21913.2%, Equiformer \u21919.0%, SphereNet \u21913.6%.  Overall 34/36 tasks improved (sign test p < 10\u207B\u2078).", {
    x: 0.5, y: 4.7, w: 9, h: 0.35,
    fontSize: 9, fontFace: "Calibri", color: C.textMid, italic: true, align: "left", valign: "middle"
  });
}

// ================================================================
// SLIDE 26 — Appendix: rMD17 Full Tables
// ================================================================
{
  const s = pres.addSlide();
  s.background = { color: C.lightBg };
  addSlideTitle(s, "附录：rMD17 完整实验结果");
  addBottomBar(s, 26);
  s.addText("附录", {
    x: 0.3, y: 5.225, w: 3, h: 0.4,
    fontSize: 9, color: "CADCFC", align: "left", valign: "middle", fontFace: "Calibri", italic: true
  });

  const hOpts = { fill: { color: C.primary }, color: C.white, bold: true, fontSize: 7, fontFace: "Calibri", align: "center", valign: "middle", margin: [1,2,1,2] };
  const gOpts = { fill: { color: "E8EDF2" }, bold: true, fontSize: 7, fontFace: "Calibri", align: "center", valign: "middle", margin: [1,2,1,2] };
  const cOpts = { fontSize: 7, fontFace: "Calibri", align: "center", valign: "middle", margin: [1,2,1,2] };
  const bOpts = { ...cOpts, bold: true, color: C.primary };

  // Energy table label
  s.addText("Energy (kcal/mol)", {
    x: 0.25, y: 0.9, w: 3, h: 0.25,
    fontSize: 10, fontFace: "Calibri", color: C.primary, bold: true, align: "left", valign: "middle", margin: 0
  });

  const eHeader = [
    { text: "", options: hOpts }, { text: "Method", options: hOpts },
    { text: "Aspirin", options: hOpts }, { text: "Azobenz.", options: hOpts },
    { text: "Benzene", options: hOpts }, { text: "Ethanol", options: hOpts },
    { text: "Malona.", options: hOpts }, { text: "Naphth.", options: hOpts },
    { text: "Paracet.", options: hOpts }, { text: "Salicylic", options: hOpts },
    { text: "Toluene", options: hOpts }, { text: "Uracil", options: hOpts },
  ];
  const eRows = [
    eHeader,
    [{ text: "SchNet", options: { ...gOpts, rowSpan: 2 } }, { text: "Baseline", options: cOpts }, { text: "0.7391", options: cOpts }, { text: "0.3968", options: cOpts }, { text: "0.0205", options: cOpts }, { text: "0.1252", options: cOpts }, { text: "0.1614", options: cOpts }, { text: "0.2116", options: cOpts }, { text: "0.3710", options: cOpts }, { text: "0.1908", options: cOpts }, { text: "0.2080", options: cOpts }, { text: "0.0787", options: cOpts }],
    [{ text: "", options: gOpts }, { text: "EDG", options: cOpts }, { text: "0.3553", options: cOpts }, { text: "0.3344", options: cOpts }, { text: "0.0171", options: cOpts }, { text: "0.0606", options: cOpts }, { text: "0.1118", options: cOpts }, { text: "0.0734", options: cOpts }, { text: "0.2830", options: cOpts }, { text: "0.1570", options: cOpts }, { text: "0.0878", options: cOpts }, { text: "0.0743", options: cOpts }],
    [{ text: "SphereNet", options: { ...gOpts, rowSpan: 4 } }, { text: "Baseline", options: cOpts }, { text: "0.1809", options: cOpts }, { text: "0.0979", options: cOpts }, { text: "0.0065", options: cOpts }, { text: "0.0378", options: cOpts }, { text: "0.0601", options: cOpts }, { text: "0.0382", options: cOpts }, { text: "0.1043", options: cOpts }, { text: "0.1412", options: cOpts }, { text: "0.0345", options: cOpts }, { text: "0.0809", options: cOpts }],
    [{ text: "", options: gOpts }, { text: "EDG", options: cOpts }, { text: "0.1362", options: cOpts }, { text: "0.0679", options: bOpts }, { text: "0.0041", options: cOpts }, { text: "0.0358", options: bOpts }, { text: "0.0566", options: bOpts }, { text: "0.0275", options: cOpts }, { text: "0.0993", options: cOpts }, { text: "0.0957", options: cOpts }, { text: "0.0241", options: cOpts }, { text: "0.0368", options: bOpts }],
    [{ text: "", options: gOpts }, { text: "EDG++(\u03BA=0)", options: cOpts }, { text: "0.1258", options: cOpts }, { text: "0.0713", options: cOpts }, { text: "0.0034", options: bOpts }, { text: "0.0408", options: cOpts }, { text: "0.0677", options: cOpts }, { text: "0.0237", options: cOpts }, { text: "0.0973", options: cOpts }, { text: "0.0656", options: bOpts }, { text: "0.0237", options: cOpts }, { text: "0.0557", options: cOpts }],
    [{ text: "", options: gOpts }, { text: "EDG++", options: bOpts }, { text: "0.1239", options: bOpts }, { text: "0.0713", options: cOpts }, { text: "0.0034", options: bOpts }, { text: "0.0358", options: bOpts }, { text: "0.0661", options: cOpts }, { text: "0.0226", options: bOpts }, { text: "0.0963", options: bOpts }, { text: "0.0656", options: bOpts }, { text: "0.0222", options: bOpts }, { text: "0.0484", options: cOpts }],
    [{ text: "ViSNet", options: { ...gOpts, rowSpan: 2 } }, { text: "Baseline", options: cOpts }, { text: "0.0555", options: cOpts }, { text: "0.0208", options: cOpts }, { text: "0.0063", options: cOpts }, { text: "0.0110", options: cOpts }, { text: "0.0152", options: cOpts }, { text: "0.0131", options: cOpts }, { text: "0.0270", options: cOpts }, { text: "0.0197", options: cOpts }, { text: "0.0109", options: cOpts }, { text: "0.0124", options: cOpts }],
    [{ text: "", options: gOpts }, { text: "EDG", options: cOpts }, { text: "0.0465", options: cOpts }, { text: "0.0184", options: cOpts }, { text: "0.0062", options: cOpts }, { text: "0.0099", options: cOpts }, { text: "0.0140", options: cOpts }, { text: "0.0118", options: cOpts }, { text: "0.0249", options: cOpts }, { text: "0.0191", options: cOpts }, { text: "0.0100", options: cOpts }, { text: "0.0119", options: cOpts }],
  ];

  const rColW = [0.65, 0.65, 0.7, 0.7, 0.65, 0.65, 0.65, 0.65, 0.68, 0.68, 0.65, 0.62];
  s.addTable(eRows, {
    x: 0.25, y: 1.15, colW: rColW,
    border: { type: "solid", pt: 0.3, color: C.border },
    rowH: 0.22
  });

  // Force table label
  s.addText("Force (kcal/mol\u00B7\u00C5)", {
    x: 0.25, y: 3.45, w: 3, h: 0.25,
    fontSize: 10, fontFace: "Calibri", color: C.primary, bold: true, align: "left", valign: "middle", margin: 0
  });

  const fRows = [
    eHeader,
    [{ text: "SchNet", options: { ...gOpts, rowSpan: 2 } }, { text: "Baseline", options: cOpts }, { text: "1.0425", options: cOpts }, { text: "0.9008", options: cOpts }, { text: "0.1857", options: cOpts }, { text: "0.3852", options: cOpts }, { text: "0.6554", options: cOpts }, { text: "0.3985", options: cOpts }, { text: "0.8254", options: cOpts }, { text: "0.7749", options: cOpts }, { text: "0.4832", options: cOpts }, { text: "0.5140", options: cOpts }],
    [{ text: "", options: gOpts }, { text: "EDG", options: cOpts }, { text: "1.0491", options: cOpts }, { text: "0.9169", options: cOpts }, { text: "0.1711", options: cOpts }, { text: "0.3790", options: cOpts }, { text: "0.6468", options: cOpts }, { text: "0.3951", options: cOpts }, { text: "0.8330", options: cOpts }, { text: "0.7431", options: cOpts }, { text: "0.4779", options: cOpts }, { text: "0.5078", options: cOpts }],
    [{ text: "SphereNet", options: { ...gOpts, rowSpan: 4 } }, { text: "Baseline", options: cOpts }, { text: "0.3913", options: cOpts }, { text: "0.2178", options: cOpts }, { text: "0.0215", options: cOpts }, { text: "0.1943", options: cOpts }, { text: "0.2928", options: cOpts }, { text: "0.1114", options: cOpts }, { text: "0.3227", options: cOpts }, { text: "0.2869", options: cOpts }, { text: "0.1098", options: cOpts }, { text: "0.2770", options: cOpts }],
    [{ text: "", options: gOpts }, { text: "EDG", options: cOpts }, { text: "0.3860", options: cOpts }, { text: "0.2167", options: bOpts }, { text: "0.0210", options: cOpts }, { text: "0.1874", options: cOpts }, { text: "0.2846", options: bOpts }, { text: "0.1110", options: cOpts }, { text: "0.3202", options: cOpts }, { text: "0.2830", options: bOpts }, { text: "0.1080", options: cOpts }, { text: "0.1718", options: bOpts }],
    [{ text: "", options: gOpts }, { text: "EDG++(\u03BA=0)", options: cOpts }, { text: "0.3796", options: cOpts }, { text: "0.2224", options: cOpts }, { text: "0.0208", options: cOpts }, { text: "0.1887", options: cOpts }, { text: "0.3160", options: cOpts }, { text: "0.0971", options: cOpts }, { text: "0.3231", options: cOpts }, { text: "0.2987", options: cOpts }, { text: "0.1034", options: bOpts }, { text: "0.2312", options: cOpts }],
    [{ text: "", options: gOpts }, { text: "EDG++", options: bOpts }, { text: "0.3793", options: bOpts }, { text: "0.2171", options: cOpts }, { text: "0.0201", options: bOpts }, { text: "0.1870", options: bOpts }, { text: "0.3160", options: cOpts }, { text: "0.0954", options: bOpts }, { text: "0.3185", options: bOpts }, { text: "0.2945", options: cOpts }, { text: "0.1034", options: bOpts }, { text: "0.2254", options: cOpts }],
    [{ text: "ViSNet", options: { ...gOpts, rowSpan: 2 } }, { text: "Baseline", options: cOpts }, { text: "0.1516", options: cOpts }, { text: "0.0573", options: cOpts }, { text: "0.0066", options: cOpts }, { text: "0.0569", options: cOpts }, { text: "0.0928", options: cOpts }, { text: "0.0281", options: cOpts }, { text: "0.1049", options: cOpts }, { text: "0.0835", options: cOpts }, { text: "0.0298", options: cOpts }, { text: "0.0525", options: cOpts }],
    [{ text: "", options: gOpts }, { text: "EDG", options: cOpts }, { text: "0.1500", options: cOpts }, { text: "0.0569", options: cOpts }, { text: "0.0065", options: cOpts }, { text: "0.0556", options: cOpts }, { text: "0.0899", options: cOpts }, { text: "0.0280", options: cOpts }, { text: "0.1060", options: cOpts }, { text: "0.0811", options: cOpts }, { text: "0.0278", options: cOpts }, { text: "0.0510", options: cOpts }],
  ];

  s.addTable(fRows, {
    x: 0.25, y: 3.7, colW: rColW,
    border: { type: "solid", pt: 0.3, color: C.border },
    rowH: 0.22
  });
}

// ================================================================
// Write file
// ================================================================
pres.writeFile({ fileName: "/home/lzeng/workspace/EDG++_Presentation.pptx" })
  .then(() => console.log("SUCCESS: EDG++_Presentation.pptx created"))
  .catch((err) => console.error("ERROR:", err));
