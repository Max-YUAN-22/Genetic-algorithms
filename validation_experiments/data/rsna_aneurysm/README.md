
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
