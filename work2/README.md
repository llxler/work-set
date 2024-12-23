# 镭射金枪鱼小队 - CSDN文章标签预测系统

## 项目介绍

本项目旨在通过大模型技术实现对CSDN文章的精准标签预测。我们结合数据预处理、模型微调，构建了一套高效的预测系统，具体优势如下：

- **运用海量数据**：写了一个脚本负责获取海量来自csdn上面的数据。
- **迁移学习能力**：基于目前最强推理模型Qwen2.5预训练模型，实现快速适配多样化文本数据。
- **全面评估**：利用`precision`、`recall`和`f1-score`量化模型表现，确保模型在不同场景下的稳定性。
- **最好结果**：
```bash
Precision: 0.982
Recall: 0.975
F1-Score: 0.9784874808380174
```

## 环境准备

```bash
conda create -n llm python=3.11
conda activate llm
pip install -r requirements.txt
pip3 install torch torchvision torchaudio
```

## 推理步骤
1. 打开目录下的`test.py`脚本
2. 找到TODO中的model_path，修改为模型路径为镜像路径`model`，或者为训练后的路径位置。
3. 运行以下命令进行测试：`python test.py`

