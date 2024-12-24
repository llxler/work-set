# 基于llama的标签预测器

## 什么是LLama模型？  
LLama是一种预训练语言模型，具有强大的语义理解能力，适合处理各种文本任务。本项目通过微调LLama模型，实现文章标签的自动预测。  

---

## 如何实现标签预测？  
### 步骤 1：准备数据  
- **清洗数据**：去除无效字符和多余空格。  
- **分词处理**：利用`jieba`进行中文分词。  

### 步骤 2：设计模型  
- 加载预训练的LLama模型。  
- 添加分类层全连接层 + Softmax。  

### 步骤 3：优化训练  
- 使用Cross Entropy Loss。  
- 调整学习率曲线，采用Step Decay策略。  

### 步骤 4：进行推理  
- 加载训练好的模型权重，输入文本数据即可获得预测结果。  

---

## 快速开始
1. 搭建环境
```bash  
conda env create -f environment.yml
conda activate llm
```
2. 搭建训练环境并开始训练

3. 开始测试！
打开`tag-predictor.py`, 并修改config配置
```python
MODEL_NAME = "model_name"
MODEL_PATH = "xxx"
TEST_DATA_PATH = "csdn_test.xlsx"
OUTPUT_PATH = "output_pred_ans.csv"
```
然后开始运行`python tag-predictor.py`
