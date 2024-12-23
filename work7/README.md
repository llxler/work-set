# glmcsdn

## 项目目标  
本项目旨在构建一个高效的标签预测系统，基于CSDN文章文本内容，生成精准的标签预测结果，以提升自动化内容管理的能力。经过优化，最终F1_score稳定在**0.97-0.98**范围内

## 技术框架  
1. **模型基础**  
   使用**glm 预训练语言模型**，结合`Transformers`框架实现自定义微调，充分挖掘模型在文本理解任务中的潜力。  
   
2. **训练管线**  
   - 数据处理：通过分词、去噪和词向量表示，构建高质量的输入数据。  
   - 模型优化：采用**AdamW**优化器，结合动态学习率策略，提升训练效率。  
   - 训练管理：利用`Trainer` API完成训练过程，支持断点续训和动态调整超参数。  

3. **性能评估**  
   通过`precision`、`recall`和`f1-score`多指标评估，严格监控模型表现，确保结果可靠性。

## 实践亮点  
- **灵活的数据管道**  
  通过自定义`Dataset`和`DataLoader`，实现高效批处理，支持动态数据增强（如随机裁剪、文本遮掩）。  
- **多层次模型微调**  
  不仅调整全局权重，还对关键层（如注意力层）进行单独优化，提升模型对领域特定任务的适应能力。  
- **推理加速**  
  使用`ONNX`优化推理性能，显著缩短预测时间，适合大规模应用场景。

## 环境准备  

```bash  
conda create -n glmllm python=3.9
conda activate glmllm  
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 推理过程
修改TODO 对应的**model_path**即可开始直接运行！
最后通过sklearn库`precision_recall_fscore_support`通过预测出来的y_hat和y_true计算出对应的f1_score值
