# 镭射金枪鱼小队 - CSDN文章标签预测系统

## 项目介绍

本项目旨在通过深度学习技术实现对CSDN文章的精准标签预测。我们结合数据预处理、模型微调和指标评估，构建了一套高效的预测系统，具体优势如下：

- **运用海量数据**：我们抽取了大量

- **迁移学习能力**：基于Qwen等预训练模型，实现快速适配多样化文本数据。
- **全面评估**：利用`precision`、`recall`和`f1-score`量化模型表现，确保模型在不同场景下的稳定性。



## 环境准备

```bash
conda create -n <env_name> python=3.10
conda activate <env_name>
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 推理步骤

1. 配置模型路径`model_path`和测试数据路径`test_data_path`。
2. 运行以下命令进行推理：``

3. 输出示例：

\`\`\`bash
Precision: 0.981
Recall: 0.976
F1-Score: 0.978
\`\`\`