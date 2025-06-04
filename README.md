# 🐍 AI贪吃蛇 - 深度强化学习项目

基于Deep Q-Network (DQN)的智能贪吃蛇游戏，使用PyTorch实现。该项目展示了如何使用深度强化学习技术训练AI玩贪吃蛇游戏，并实现了多项高级优化策略。

![训练效果](plots/training_plot.png)

## 🎯 项目特色

### 🧠 先进的AI架构
- **Dueling DQN**: 分离状态价值和动作优势的双流网络架构
- **Double DQN**: 减少Q值过估计，提高训练稳定性
- **优先经验回放**: 重点学习重要经验，提升学习效率
- **注意力机制**: 突出重要特征，增强决策能力

### 🛡️ 智能安全策略
- **多层陷阱检测**: 预测并避免死胡同
- **动态安全评估**: 根据蛇长调整安全策略
- **前瞻路径规划**: 20-30步前瞻计算最优路径
- **反转圈检测**: 防止AI原地打转

### 🎮 精细化奖励机制
- **分层奖励系统**: 基础奖励+长度奖励+效率奖励
- **距离导向**: 强化朝向食物的移动
- **高分保护**: 分数越高，死亡惩罚越重
- **存活激励**: 鼓励长期存活和空间高效利用

## 📊 训练成果

| 指标 | 优化前 | 优化后 | 改进 |
|------|--------|--------|------|
| 最高分 | 41分 | 60+分 | **+46%** |
| 平均分 | 7-8分 | 15-20分 | **+150%** |
| 自杀率 | >50% | <15% | **-70%** |
| 转圈率 | >90% | <10% | **-89%** |

## 🚀 快速开始

### 环境要求
- Python 3.8+
- CUDA支持的GPU (推荐RTX 3060或更高)
- 8GB+ RAM

### 安装依赖
```bash
pip install -r requirements.txt
```

### 训练新模型
```bash
# 基础训练 (1000局)
python train.py --model dqn --games 1000 --speed 100

# 高级训练 (5000局，无显示，快速)
python train.py --model dqn --no-show --speed 100 --games 5000 --interval 100

# 自定义训练参数
python train.py --model dqn --games 3000 --speed 50 --lr 0.0003 --gamma 0.98
```

### 测试已训练模型
```bash
# 使用最佳模型进行游戏
python play.py --model model/best_dqn_model.pth --speed 20

# 可视化训练过程
python visualize.py --model model/best_dqn_model.pth
```

## 📁 项目结构

```
ai-snake/
├── agent.py              # DQN智能体实现
├── model.py               # 神经网络模型架构
├── snake_game.py          # 游戏环境和逻辑
├── train.py               # 训练脚本
├── play.py                # 游戏演示脚本
├── visualize.py           # 训练结果可视化
├── utils.py               # 工具函数
├── requirements.txt       # 依赖包列表
├── model/                 # 保存的模型文件
│   ├── best_dqn_model.pth
│   └── checkpoint_*.pth
├── plots/                 # 训练图表
│   ├── training_plot.png
│   ├── loss_plot.png
│   └── epsilon_plot.png
└── docs/                  # 项目文档
    ├── OPTIMIZATION_SUMMARY.md
    ├── BREAKTHROUGH_STRATEGY.md
    └── TRAINING_5000_GUIDE.md
```

## 🔧 高级配置

### 训练参数调优
```python
# agent.py 中的关键参数
MAX_MEMORY = 150_000      # 经验池大小
BATCH_SIZE = 128          # 批次大小
LR = 0.0003              # 学习率
gamma = 0.98             # 折扣因子
```

### 模型架构定制
```python
# 创建自定义模型
agent = Agent(
    model_type="dqn",       # 或 "linear"
    input_size=16,          # 状态特征数
    hidden_size=512,        # 隐藏层大小
    output_size=3           # 动作数量
)
```

## 📈 训练监控

训练过程中会实时生成以下图表：
- **training_plot.png**: 分数和蛇长变化
- **loss_plot.png**: 损失函数变化
- **epsilon_plot.png**: 探索率衰减

每100局自动保存检查点，可以随时恢复训练。

## 🎯 性能优化技巧

1. **GPU加速**: 自动检测并使用CUDA
2. **批量训练**: 大批次提高GPU利用率
3. **经验回放**: 打破数据相关性
4. **梯度裁剪**: 防止梯度爆炸
5. **学习率调度**: 动态调整学习率

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 📄 许可证

本项目基于MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- PyTorch团队提供的深度学习框架
- OpenAI Gym启发的强化学习环境设计
- 深度强化学习社区的研究成果

## 📞 联系方式

如有问题或建议，请通过以下方式联系：
- 提交Issue: [GitHub Issues](https://github.com/yourusername/ai-snake/issues)
- 邮箱: your.email@example.com

---

⭐ 如果这个项目对你有帮助，请给个Star支持一下！ 