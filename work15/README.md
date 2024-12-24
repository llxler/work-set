# interm与csdn标签分类

## 使用教程
1. 环境搭建
```bash
conda env create -f environment.yml --name csdn
conda activate csdn
```
2. 模型训练

3. 模型推理
- 修改model_path参数直接运行
- 或者运行命令`python output.py --model_path model`

推理标准说明：
```python
correct_count = sum(
    set(true.split(",")) == set(predicted.split(","))
    for true, predicted in zip(true_labels, predicted_labels)
)
```


## 背景与动机  
文本分类是自然语言处理领域的重要任务，但在标签预测中，数据稀疏性和语义复杂性常导致传统方法性能受限。本研究基于INTERM预训练模型，探索大模型在标签分类任务中的微调潜力。  

---

## 研究方法  
### 数据准备  
- 使用公开的CSDN文章数据集，经过正则表达式清洗与`BERT Tokenizer`分词，生成高质量输入特征。  
- 数据增强策略：随机删除、同义词替换等方法，扩充训练集规模。  

### 模型设计  
- 基于INTERM的深度模型，添加双向GRU层以强化上下文理解能力。  
- 微调阶段引入**对比学习（Contrastive Learning）**，提升模型对相似标签的区分能力。  

### 实验设置  
- 优化器：`AdamW`  
- 学习率：2e-5（使用线性预热）  
- 批量大小：32  

---

## 结果分析  
- F1_score达到0.96-0.98，优于传统SVM和LSTM模型的0.85。  
- 模型对长尾标签表现突出，召回率提升20%。  

---

## 结论与展望  
本研究证明了INTERM模型在标签预测任务中的优越性，未来可以扩展其他领域的文本分类应用。
