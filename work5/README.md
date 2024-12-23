# csdn-classifier

本项目利用深度学习技术，实现CSDN文章标签的智能化预测。项目核心特点包括：  
1. 对初赛提供的文本数据进行分词、去噪和向量化操作，提升训练效率。  
2. 基于Qwen预训练语言模型，采用常见的微调技术（如Adam优化器、学习率预热（learning rate warm-up）、Dropout等），通过小批量梯度下降（SGD）优化模型权重。  
3. 利用`Transformers`库中的`Trainer`接口，结合学习率预热和动态调整策略（如Cosine Annealing），加速模型收敛并防止过拟合。  
4. 支持多维度性能评估，精准衡量`precision`、`recall`和`f1-score`等指标的表现，确保模型效果在不同维度上达到最优。 
   

## 环境准备  

```bash  
conda create -n tagclassifier python=3.10  
conda activate tagclassifier  
pip install -r requirements.txt
```

## 推理说明
推理前先修改model_path为下载后的文件夹路径，或者用训练后的模型路径。直接运行test.py后可以看到打印后的结果
```bash
conda activate tagclassifier
python test.py
```
最终结果f1_score测试在0.97-0.987之间