# LLama-tags

## 项目概述  
如何让CSDN文章标签预测更加智能、高效？本项目以Llama预训练模型为基础，通过创新型的微调和优化策略，打造了一个精准、可扩展的标签分类系统，不仅提升了预测性能，还为未来多场景文本分类应用提供了参考方案。  

---

## 项目亮点  
1. **创新型微调策略**：  
   - 采用**分层冻结（Layer-wise Freezing）**技术，仅解冻高层编码器权重以适应新任务，减少过拟合风险。  
   - 在分类层引入**多任务学习（Multi-task Learning）**，同时优化主分类任务与辅助任务（例如标签关系建模）。  

2. **动态数据增强**：  
   - Character Shuffling与相似句式替换，模拟真实场景中的数据噪声。  
   - 融合基于规则和基于模型的增强方法，提升模型对少数类标签的识别能力。  

3. **高效训练优化**：  
   - 使用八个H800, 调用全量优化。  
   - 动态学习率调度，自动调控每个epoch的学习率曲线。  

---

## 模型性能  
- **精准性**：F1_score 达到 **0.96+**，大幅领先传统方法。  
- **鲁棒性**：在真实场景数据中表现稳定，即使输入存在一定噪声。  

---

## 快速上手  
### 环境准备  
```bash
conda env create -f environment.yml
conda activate tagpred
```

### 训练部署

### 测试阶段
直接运行`python inference.py --model_path model --test_data csdn_test.xlsx --output results.csv`，如果model_path下载在别处，更改此处的运行配置即可。