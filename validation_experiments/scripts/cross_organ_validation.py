#!/usr/bin/env python3
"""跨器官泛化验证实验."""

import json

import torch
from datasets import KiTSDataset, MSDDataset
from models import MultimodalSegmentation
from utils import evaluate_metrics, set_seed


def run_cross_organ_validation():
    """运行跨器官验证实验."""
    print("🔬 开始跨器官泛化验证实验...")

    # 加载配置
    with open("validation_experiments/configs/cross_organ_config.json") as f:
        config = json.load(f)

    # 设置随机种子
    set_seed(42)

    # 加载预训练模型
    model = MultimodalSegmentation()
    model.load_state_dict(torch.load("real_training_results/best_real_model.pth"))

    results = {}

    # 在不同器官数据上测试
    organ_datasets = {
        "msd_liver": MSDDataset("liver"),
        "msd_heart": MSDDataset("heart"),
        "msd_lung": MSDDataset("lung"),
        "kits": KiTSDataset(),
    }

    for organ_name, dataset in organ_datasets.items():
        print(f"📊 测试器官: {organ_name}")

        # 微调模型（如果需要）
        if config["fine_tune"]:
            print(f"🔧 微调模型用于 {organ_name}")
            model = fine_tune_model(model, dataset)

        # 评估模型
        metrics = evaluate_metrics(model, dataset)
        results[organ_name] = metrics

        print(f"✅ {organ_name} 结果: {metrics}")

    # 保存结果
    with open("validation_experiments/results/cross_organ/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("🎉 跨器官验证实验完成！")
    return results


def fine_tune_model(model, dataset, epochs=50):
    """微调模型."""
    # 实现微调逻辑
    pass


if __name__ == "__main__":
    results = run_cross_organ_validation()
