# 跨数据集验证实验

## 🎯 实验目标

通过多个公开数据集验证我们方法的泛化性和鲁棒性，应对MIA审稿意见。

## 📊 实验类型

### 1. 跨年份验证 (Cross-Year Validation)
- **目标**: 验证方法在不同年份BRaTS数据上的稳定性
- **数据集**: BRaTS 2018, 2019, 2021
- **脚本**: `scripts/cross_year_validation.py`

### 2. 跨器官验证 (Cross-Organ Validation)  
- **目标**: 验证方法在其他器官上的泛化能力
- **数据集**: MSD Liver/Heart/Lung, KiTS
- **脚本**: `scripts/cross_organ_validation.py`

### 3. 跨模态验证 (Cross-Modality Validation)
- **目标**: 验证不同MRI序列组合的效果
- **组合**: T1+T1ce, T1+T2, T1+FLAIR, 全部4个
- **脚本**: `scripts/cross_modality_validation.py`

## 🚀 快速开始

### 1. 下载数据集
```bash
# BRaTS 2018
wget [BRaTS2018下载链接] -O data/brats_2018.zip
unzip data/brats_2018.zip -d data/brats_2018/

# BRaTS 2019  
wget [BRaTS2019下载链接] -O data/brats_2019.zip
unzip data/brats_2019.zip -d data/brats_2019/

# BRaTS 2021
wget [BRaTS2021下载链接] -O data/brats_2021.zip
unzip data/brats_2021.zip -d data/brats_2021/

# MSD数据集
wget [MSD下载链接] -O data/msd.zip
unzip data/msd.zip -d data/msd/
```

### 2. 运行实验
```bash
# 跨年份验证
python scripts/cross_year_validation.py

# 跨器官验证
python scripts/cross_organ_validation.py

# 跨模态验证
python scripts/cross_modality_validation.py
```

### 3. 查看结果
```bash
# 查看所有结果
ls results/*/results.json

# 分析结果
python analyze_results.py
```

## 📈 预期结果

### 跨年份验证
- BRaTS 2018: Dice > 0.80
- BRaTS 2019: Dice > 0.82
- BRaTS 2021: Dice > 0.85

### 跨器官验证
- MSD Liver: Dice > 0.75
- MSD Heart: Dice > 0.70
- MSD Lung: Dice > 0.65
- KiTS: Dice > 0.70

### 跨模态验证
- 双模态: Dice > 0.80
- 四模态: Dice > 0.85

## 📝 结果分析

实验结果将保存在 `results/` 目录下，包括：
- 详细的性能指标
- 统计显著性分析
- 可视化图表
- 与预期结果的对比

## 🎯 审稿意见应对

这些验证实验可以有效应对以下审稿意见：
1. "需要更多数据集验证"
2. "需要泛化性分析"
3. "需要跨模态验证"
4. "需要临床相关性分析"

## 📞 支持

如有问题，请查看：
- 配置文件: `configs/`
- 实验脚本: `scripts/`
- 结果分析: `results/`
