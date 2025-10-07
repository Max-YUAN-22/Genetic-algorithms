## Title
Multimodal Brain Tumor Segmentation with Cross-Modal Attention and Multi-Objective Genetic Optimization

## Abstract (150–250 words)
- Problem: CT+MRI 融合分割存在模态偏差与结构不一致。
- Method: 多模态 YOLO 分割 + 交叉模态注意力；多目标遗传算法（准确性/效率/不确定性）联合优化。
- Results: Dice、WT/TC/ET 分区指标显著优于基线；临床不确定性可视化。
- Impact: 复现性强、推理高效，可落地临床验证。

## 1. Introduction
- 背景与挑战（多模态对齐、域偏移、临床可靠性）。
- 贡献列表：
  1) 交叉模态注意力融合模块；
  2) 多目标遗传优化（准确性、效率、不确定性）；
  3) 标准化评估与临床指标；
  4) 开源可复现框架（MLflow + metadata）。

## 2. Related Work
- 多模态医学分割；注意力与特征对齐；神经架构/超参搜索；不确定性与校准。

## 3. Methods
### 3.1 Multimodal YOLO Segmentation
- Backbone/Neck/Head 概述；分割头与损失函数；训练细节（imgsz、增广）。
### 3.2 Cross-Modal Attention Fusion
- 模块示意与公式：将 CT/MRI 特征经 Query/Key/Value 融合；跨尺度交互。
- 正则与稳定性（LayerNorm、残差、温度缩放）。
### 3.3 Multi-Objective Genetic Tuning
- 目标：maximize Dice；minimize FLOPs/latency；minimize predictive uncertainty。
- 基因编码：架构宽深、卷积核、注意力开关、阈值等。
- 选择/交叉/变异策略与约束；早停与精英保留。
### 3.4 Uncertainty & Postprocessing
- MC Dropout/TTA 估计不确定性；形态学后处理；阈值选择与临床友好策略。

## 4. Experiments
### 4.1 Datasets & Preprocessing
- BRaTS/Real-Clinic 数据；配准、归一化、切片策略；训练/验证划分与种子。
### 4.2 Baselines & SOTA
- U-Net、Attention U-Net、nnU-Net、YOLO 分割系列。
### 4.3 Metrics
- Dice、Jaccard、Hausdorff95、Sensitivity/Specificity；校准（ECE）。
### 4.4 Implementation & Reproducibility
- 种子、硬件、框架版本；MLflow 与 `metadata.json`；CI 验证。
### 4.5 Statistical Significance
- N 次独立运行（N≥5），汇报 mean±std；配对/独立 t 检验与效应量（Cohen’s d）。

## 5. Results
- 主结果表（总体 Dice 与分区 WT/TC/ET）；
- 资源对比（时延/显存/FLOPs）；
- 不确定性热图与临床病例可视化。

## 6. Ablation Studies
- 去除交叉模态注意力；
- 去除 GA 优化或单目标优化；
- 不同后处理参数；
- 训练增广与输入分辨率敏感性。

## 7. Discussion
- 优势与局限（数据依赖、跨中心泛化、注册误差）；
- 伦理与隐私；
- 临床集成与未来工作（主动学习、少样本迁移）。

## 8. Conclusion
- 总结贡献与临床价值；复现性与可扩展性。

## Appendix
- 伪代码/配置清单；更多可视化与失败案例；统计详情与 p 值表。


