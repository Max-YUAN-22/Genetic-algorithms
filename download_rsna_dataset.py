#!/usr/bin/env python3
"""
RSNA Intracranial Aneurysm Detection 数据集下载和准备脚本
"""

import os
import json
import subprocess
from pathlib import Path

def setup_rsna_dataset():
    """设置RSNA数据集环境"""
    print("🚀 设置RSNA动脉瘤检测数据集...")
    
    # 创建目录
    rsna_dir = Path("validation_experiments/data/rsna_aneurysm")
    rsna_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建子目录
    subdirs = ["images", "labels", "metadata", "processed"]
    for subdir in subdirs:
        (rsna_dir / subdir).mkdir(exist_ok=True)
    
    print(f"📁 创建目录: {rsna_dir}")

def create_rsna_download_instructions():
    """创建RSNA数据集下载说明"""
    instructions = """
# RSNA Intracranial Aneurysm Detection 数据集下载指南

## 📋 数据集信息
- **竞赛名称**: RSNA Intracranial Aneurysm Detection
- **Kaggle链接**: https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection
- **数据规模**: 1000+ 案例
- **模态**: CTA, MRA, T1 post-contrast, T2 MRI
- **机构数量**: 18个不同机构
- **标注**: 神经放射学专家标注

## 🔑 获取数据步骤

### 1. 注册Kaggle账户
- 访问 https://www.kaggle.com/
- 注册账户并验证邮箱

### 2. 加入竞赛
- 访问竞赛页面: https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection
- 点击 "Join Competition"
- 接受竞赛规则

### 3. 获取API密钥
- 进入账户设置: https://www.kaggle.com/account
- 点击 "Create New API Token"
- 下载 `kaggle.json` 文件

### 4. 安装Kaggle API
```bash
pip install kaggle
```

### 5. 配置API密钥
```bash
# 将kaggle.json放在~/.kaggle/目录下
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 6. 下载数据集
```bash
# 进入项目目录
cd validation_experiments/data/rsna_aneurysm

# 下载竞赛数据
kaggle competitions download -c rsna-intracranial-aneurysm-detection

# 解压数据
unzip rsna-intracranial-aneurysm-detection.zip
```

## 📊 数据集结构
```
rsna_aneurysm/
├── images/
│   ├── train/
│   │   ├── CTA/
│   │   ├── MRA/
│   │   ├── T1_post/
│   │   └── T2/
│   └── test/
├── labels/
│   ├── train.csv
│   └── sample_submission.csv
├── metadata/
│   └── dataset_info.json
└── processed/
    └── preprocessed_data/
```

## 🎯 验证目标
1. **多模态融合验证**: 测试CTA+MRA+MRI的组合效果
2. **跨机构泛化性**: 验证在18个不同机构数据上的性能
3. **临床相关性**: 验证在真实临床场景中的应用价值
4. **方法鲁棒性**: 测试不同扫描仪和协议下的稳定性

## 📈 预期结果
- 最终得分: > 0.85
- 动脉瘤存在检测AUC: > 0.88
- 位置检测平均AUC: > 0.82
- 跨机构性能标准差: < 0.05

## ⚠️ 注意事项
1. 需要Kaggle账户和API密钥
2. 数据集较大，需要足够的存储空间
3. 竞赛仍在进行中，注意截止日期
4. 遵守竞赛规则和数据使用协议

## 🔗 相关链接
- 竞赛主页: https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection
- 数据描述: https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/data
- 评估指标: https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/overview/evaluation
"""
    
    with open("validation_experiments/data/rsna_aneurysm/README.md", "w") as f:
        f.write(instructions)
    
    print("📖 下载说明已创建")

def create_rsna_analysis_script():
    """创建RSNA数据集分析脚本"""
    analysis_script = '''#!/usr/bin/env python3
"""
RSNA数据集分析和预处理脚本
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

def analyze_rsna_dataset():
    """分析RSNA数据集"""
    print("🔍 分析RSNA动脉瘤检测数据集...")
    
    # 加载数据
    data_path = Path("validation_experiments/data/rsna_aneurysm")
    
    # 读取标签文件
    train_labels = pd.read_csv(data_path / "labels" / "train.csv")
    
    # 基本统计
    print(f"总样本数: {len(train_labels)}")
    print(f"动脉瘤存在率: {train_labels['aneurysm_present'].mean():.3f}")
    
    # 按机构分析
    if 'institution' in train_labels.columns:
        institution_stats = train_labels.groupby('institution').agg({
            'aneurysm_present': ['count', 'sum', 'mean']
        }).round(3)
        print("\\n按机构统计:")
        print(institution_stats)
    
    # 按模态分析
    modality_stats = {}
    for modality in ['CTA', 'MRA', 'T1_post', 'T2']:
        if f'{modality}_present' in train_labels.columns:
            modality_stats[modality] = train_labels[f'{modality}_present'].mean()
    
    print("\\n按模态统计:")
    for modality, rate in modality_stats.items():
        print(f"{modality}: {rate:.3f}")
    
    # 生成可视化
    create_rsna_visualizations(train_labels)
    
    return train_labels

def create_rsna_visualizations(df):
    """创建RSNA数据集可视化"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. 动脉瘤存在分布
    axes[0, 0].pie(df['aneurysm_present'].value_counts(), 
                   labels=['No Aneurysm', 'Aneurysm Present'],
                   autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
    axes[0, 0].set_title('Aneurysm Presence Distribution')
    
    # 2. 按机构分布
    if 'institution' in df.columns:
        institution_counts = df['institution'].value_counts()
        axes[0, 1].bar(range(len(institution_counts)), institution_counts.values)
        axes[0, 1].set_title('Cases per Institution')
        axes[0, 1].set_xlabel('Institution ID')
        axes[0, 1].set_ylabel('Number of Cases')
    
    # 3. 模态分布
    modalities = ['CTA', 'MRA', 'T1_post', 'T2']
    modality_counts = []
    for modality in modalities:
        if f'{modality}_present' in df.columns:
            modality_counts.append(df[f'{modality}_present'].sum())
        else:
            modality_counts.append(0)
    
    axes[1, 0].bar(modalities, modality_counts, color=['gold', 'lightgreen', 'lightcoral', 'plum'])
    axes[1, 0].set_title('Cases per Modality')
    axes[1, 0].set_ylabel('Number of Cases')
    
    # 4. 位置分布
    location_cols = [col for col in df.columns if col.startswith('location_')]
    if location_cols:
        location_counts = df[location_cols].sum()
        axes[1, 1].bar(range(len(location_counts)), location_counts.values)
        axes[1, 1].set_title('Aneurysm Location Distribution')
        axes[1, 1].set_xlabel('Location ID')
        axes[1, 1].set_ylabel('Number of Cases')
    
    plt.tight_layout()
    plt.savefig('validation_experiments/data/rsna_aneurysm/dataset_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    df = analyze_rsna_dataset()
    print("\\n✅ RSNA数据集分析完成！")
'''
    
    with open("validation_experiments/scripts/analyze_rsna_dataset.py", "w") as f:
        f.write(analysis_script)
    
    # 设置执行权限
    os.chmod("validation_experiments/scripts/analyze_rsna_dataset.py", 0o755)
    
    print("📊 分析脚本已创建")

def create_rsna_validation_summary():
    """创建RSNA验证总结"""
    summary = {
        "dataset_name": "RSNA Intracranial Aneurysm Detection",
        "validation_type": "cross_domain_clinical_validation",
        "key_advantages": [
            "Real clinical data from 18 institutions",
            "Multiple imaging modalities (CTA, MRA, T1 post, T2)",
            "Expert annotations by neuroradiologists", 
            "Diverse scanners and imaging protocols",
            "Large-scale dataset with statistical power",
            "High clinical relevance (aneurysm detection)"
        ],
        "validation_goals": [
            "Verify cross-modal attention mechanism effectiveness",
            "Test cross-institution generalization capability",
            "Validate clinical applicability and robustness",
            "Demonstrate method performance on real clinical data"
        ],
        "expected_benefits": [
            "Strong evidence for cross-modal fusion",
            "Proof of cross-institution generalization",
            "Clinical relevance validation",
            "Robustness demonstration"
        ],
        "target_metrics": {
            "final_score": "> 0.85",
            "aneurysm_present_auc": "> 0.88", 
            "location_detection_auc": "> 0.82",
            "cross_institution_std": "< 0.05"
        }
    }
    
    with open("validation_experiments/data/rsna_aneurysm/validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("📋 验证总结已创建")

if __name__ == "__main__":
    setup_rsna_dataset()
    create_rsna_download_instructions()
    create_rsna_analysis_script()
    create_rsna_validation_summary()
    
    print("\\n🎉 RSNA数据集环境设置完成！")
    print("📁 数据目录: validation_experiments/data/rsna_aneurysm/")
    print("📖 请查看 README.md 了解下载步骤")
    print("🚀 准备好后可以运行验证实验！")
