#!/usr/bin/env python3
"""BRaTS跨年份验证实验."""

import json

import torch
from datasets import BRaTSDataset
from models import MultimodalSegmentation
from utils import evaluate_metrics, set_seed


def run_cross_year_validation():
    """运行跨年份验证实验."""
    print("🔬 开始BRaTS跨年份验证实验...")

    # 加载配置
    with open("validation_experiments/configs/cross_year_config.json") as f:
        config = json.load(f)

    # 设置随机种子
    set_seed(42)

    # 加载预训练模型
    model = MultimodalSegmentation()
    model.load_state_dict(torch.load("real_training_results/best_real_model.pth"))
    model.eval()

    results = {}

    # 在不同年份数据上测试
    for dataset_name in config["test_datasets"]:
        print(f"📊 测试数据集: {dataset_name}")

        # 加载测试数据
        test_dataset = BRaTSDataset(data_path=f"validation_experiments/data/{dataset_name}", split="test")

        # 评估模型
        metrics = evaluate_metrics(model, test_dataset)
        results[dataset_name] = metrics

        print(f"✅ {dataset_name} 结果: {metrics}")

    # 保存结果
    with open("validation_experiments/results/cross_year/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("🎉 跨年份验证实验完成！")
    return results


if __name__ == "__main__":
    results = run_cross_year_validation()
