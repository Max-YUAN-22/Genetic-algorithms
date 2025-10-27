# 遗传算法优化流程图

```mermaid
graph TD
    A[开始] --> B[初始化种群]
    B --> C[编码网络架构参数]
    C --> D[评估适应度函数]
    D --> E{是否满足终止条件?}
    E -->|否| F[选择操作]
    E -->|是| G[输出最优解]
    F --> H[交叉操作]
    H --> I[变异操作]
    I --> J[精英保留]
    J --> K[更新种群]
    K --> D
    
    subgraph "适应度函数"
        D1[Dice系数]
        D2[计算效率]
        D3[不确定性]
        D4[多目标优化]
    end
    
    subgraph "基因编码"
        C1[网络宽度]
        C2[网络深度]
        C3[注意力机制]
        C4[超参数]
    end
    
    subgraph "遗传操作"
        F1[锦标赛选择]
        H1[模拟二进制交叉]
        I1[高斯变异]
        J1[精英策略]
    end
    
    D --> D1
    D --> D2
    D --> D3
    D --> D4
    
    C --> C1
    C --> C2
    C --> C3
    C --> C4
    
    F --> F1
    H --> H1
    I --> I1
    J --> J1
```

## 遗传算法参数设置

| 参数 | 值 | 说明 |
|------|-----|------|
| 种群大小 | 50 | 平衡探索与收敛 |
| 最大代数 | 100 | 防止过拟合 |
| 交叉概率 | 0.8 | 保持多样性 |
| 变异概率 | 0.1 | 局部搜索 |
| 精英比例 | 0.2 | 保留优秀个体 |

## 适应度函数设计

```python
def fitness_function(individual):
    """多目标适应度函数."""
    # 主要目标：分割精度
    dice_score = evaluate_dice(individual)

    # 次要目标：计算效率
    efficiency = 1.0 / (flops + latency)

    # 第三目标：不确定性
    uncertainty = evaluate_uncertainty(individual)

    # 加权组合
    fitness = 0.6 * dice_score + 0.3 * efficiency + 0.1 * uncertainty

    return fitness
```
