#!/usr/bin/env python3
"""RSNA数据集分析和预处理脚本."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def analyze_rsna_dataset():
    """分析RSNA数据集."""
    print("🔍 分析RSNA动脉瘤检测数据集...")

    # 加载数据
    data_path = Path("validation_experiments/data/rsna_aneurysm")

    # 读取标签文件
    train_labels = pd.read_csv(data_path / "labels" / "train.csv")

    # 基本统计
    print(f"总样本数: {len(train_labels)}")
    print(f"动脉瘤存在率: {train_labels['aneurysm_present'].mean():.3f}")

    # 按机构分析
    if "institution" in train_labels.columns:
        institution_stats = (
            train_labels.groupby("institution").agg({"aneurysm_present": ["count", "sum", "mean"]}).round(3)
        )
        print("\n按机构统计:")
        print(institution_stats)

    # 按模态分析
    modality_stats = {}
    for modality in ["CTA", "MRA", "T1_post", "T2"]:
        if f"{modality}_present" in train_labels.columns:
            modality_stats[modality] = train_labels[f"{modality}_present"].mean()

    print("\n按模态统计:")
    for modality, rate in modality_stats.items():
        print(f"{modality}: {rate:.3f}")

    # 生成可视化
    create_rsna_visualizations(train_labels)

    return train_labels


def create_rsna_visualizations(df):
    """创建RSNA数据集可视化."""
    _fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # 1. 动脉瘤存在分布
    axes[0, 0].pie(
        df["aneurysm_present"].value_counts(),
        labels=["No Aneurysm", "Aneurysm Present"],
        autopct="%1.1f%%",
        colors=["lightblue", "lightcoral"],
    )
    axes[0, 0].set_title("Aneurysm Presence Distribution")

    # 2. 按机构分布
    if "institution" in df.columns:
        institution_counts = df["institution"].value_counts()
        axes[0, 1].bar(range(len(institution_counts)), institution_counts.values)
        axes[0, 1].set_title("Cases per Institution")
        axes[0, 1].set_xlabel("Institution ID")
        axes[0, 1].set_ylabel("Number of Cases")

    # 3. 模态分布
    modalities = ["CTA", "MRA", "T1_post", "T2"]
    modality_counts = []
    for modality in modalities:
        if f"{modality}_present" in df.columns:
            modality_counts.append(df[f"{modality}_present"].sum())
        else:
            modality_counts.append(0)

    axes[1, 0].bar(modalities, modality_counts, color=["gold", "lightgreen", "lightcoral", "plum"])
    axes[1, 0].set_title("Cases per Modality")
    axes[1, 0].set_ylabel("Number of Cases")

    # 4. 位置分布
    location_cols = [col for col in df.columns if col.startswith("location_")]
    if location_cols:
        location_counts = df[location_cols].sum()
        axes[1, 1].bar(range(len(location_counts)), location_counts.values)
        axes[1, 1].set_title("Aneurysm Location Distribution")
        axes[1, 1].set_xlabel("Location ID")
        axes[1, 1].set_ylabel("Number of Cases")

    plt.tight_layout()
    plt.savefig("validation_experiments/data/rsna_aneurysm/dataset_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    df = analyze_rsna_dataset()
    print("\n✅ RSNA数据集分析完成！")
