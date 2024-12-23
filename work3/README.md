# QWEN2.5-label classifier

## 项目概览  
**最终我们的F1_score达到了0.97-0.98!** 🎉🎉🎉

本项目利用深度学习技术，实现CSDN文章标签的智能化预测。项目核心特点包括：  
- **数据预处理**：对初赛提供的文本数据进行分词、去噪和向量化操作，提升训练效率。  
- **模型微调**：基于Qwen预训练语言模型，通过小批量梯度下降优化模型权重。  
- **训练策略**：采用学习率预热（learning rate warm-up）和动态调整策略，提高模型收敛速度。  
- **评估指标**：支持多维度性能评估，精准衡量`precision`、`recall`和`f1-score`指标表现。 
   

## 环境准备  

```bash  
conda create -n qwenllm python=3.10  
conda activate qwenllm 
pip install -r requirements.txt
```  

## 推理流程  

在65行即main函数里面的第一句中填写模型的路径，可以为下载下来的文件夹路径或者是train后的输出路径，填写好后运行 `python predict.py`
