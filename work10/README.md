# GLM-Label Classifier   

---

## 项目流程  
### 1. 数据处理  
- **文本清洗**：去除HTML标签、标点符号和停用词，确保数据输入质量。  
- **向量化**：利用GLM内置的`Tokenizer`对文本进行分词和嵌入编码，保持预训练模型一致性。  

### 2. 模型微调  
- **预训练模型选择**：基于GLM-2的语言模型框架，扩展分类头以适应标签任务。  
- **训练优化**：  
  - 引入`Cross Entropy Loss`作为目标函数，确保多分类任务的稳定训练。  
  - 采用**学习率调度**策略，提升模型收敛速度。  

### 3. 推理与部署  
- **加速推理**：挂载vLLM上进行平时的调试。  
- **动态阈值策略**：根据预测分布调整分类阈值，实现精确率与召回率的平衡。  

---

## 技术亮点  
1. **模型自适应性**  
   GLM的大规模预训练参数在小样本数据上展现强大迁移能力，无需大规模标注数据即可取得高精度。  

2. **标签结构建模**  
   标签嵌入与文本嵌入进行交互匹配，利用GLM的双向编码能力提升预测效果。  

3. **高效部署**  
   导出优化后的ONNX格式模型，支持快速推理和大规模批量预测，满足实时应用需求。

---

## 使用指南  

### 环境配置  
```bash  
conda create -n glm python=3.10  
conda activate glm  
pip install -r requirements.txt  
```

## 推理介绍
本项目基于GLM，开发了一套用于CSDN文章标签预测的智能分类系统。通过大模型的语义理解能力和高效优化策略，最终实现了**F1_score 0.97-0.98**的优秀表现。 
推理过程：
- 来到`test.py`处，修改开头的配置
```python
######### config #########
model_path = "xxx"
test_data_path = "csdn_test.xlsx"
save_data_path = "pred_list_save.csv"
sys_prompt = "Tell me the tags based on the blog content."
######### config #########
```
- 运行 `python test.py`