# 智谱清言分类模型

## 项目概览  

### 1. 数据预处理  
对原始文本数据进行了多步清洗和转换，包括：
- **分词**：使用分词工具将文本拆解为更易处理的单元。
- **去噪**：去除不必要的字符、标点和停用词，确保训练数据的纯净性。
- **向量化**：通过TF-IDF和Word2Vec将文本转换为数值向量，准备输入到模型。

### 2. 模型微调  
采用glm训练语言模型，进行定制化的微调：
- 使用`Transformers`库中的`Trainer`接口，简化训练流程。
- 通过`AdamW`优化器和**小批量梯度下降**方法对模型进行优化。
- 使用**学习率预热**和**动态调整学习率**（技术，提升训练稳定性和收敛速度。

### 3. 训练策略与技巧  
为了提高训练效率和模型性能，采用了以下策略：
- **早停法**：防止模型过拟合，自动停止训练。
- **正则化**：通过Dropout和L2正则化防止过拟合，确保模型的泛化能力。

### 4. 模型评估  
通过多维度评估模型表现：
- 精确度（Precision）、召回率（Recall）和F1-score，全面衡量模型在不同任务上的表现，确保标签预测的准确性和稳定性。

## 环境准备  

```bash  
conda create -n glm python=3.11  
conda activate glm
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 推理过程
推理采用`sklearn`库`precision_recall_fscore_support`获取。
- 修改main函数中的model_path路径为刚刚下载的路径。
- 运行函数`python glm-predict.py`
- 自动打印结果