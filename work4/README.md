# CSDN标签预测系统  

## 结果
F1_score 97.4+ !!!

本项目专注于利用深度学习技术，为CSDN文章设计一套高效的标签预测模型。主要优势：  
- **特征工程**：提取文本关键词，并结合深度嵌入（deep embedding）构建训练样本。  
- **模型架构**：引入多头注意力机制（multi-head attention），提升文本特征捕捉能力。  
- **早停策略**：训练过程中监控验证集损失，避免过拟合。  
- **性能评估**：通过K折交叉验证（k-fold cross-validation）进一步优化预测结果。  

## 环境搭建  

```bash  
conda create -n csdn python=3.10  
conda activate csdn
pip3 install torch torchvision torchaudio
pip install -r requirements.txt  
```  

## 推理说明  

1. 修改`model_path = "change this path!!!"`中的模型路径
2. 执行以下命令开始推理：  

```bash  
python inference.py  
```  

## 推理标准介绍
precision采用对标准确率的标准，其他采用传统的估算方法(**所以precision相当于accuracy**)
```python
# True Positive (TP), False Positive (FP), and False Negative (FN) counters
TP, FP, FN = 0, 0, 0

for i in range(len(right_ans)):
    if predict_ans[i] == right_ans[i]:
        TP += 1  # 正确预测
    elif predict_ans[i] in right_ans:
        FP += 1  # 错误预测为正类
    else:
        FN += 1  # 漏掉的正类

recall = TP / (TP + FN) if (TP + FN) > 0 else 0
```
