# 智谱清言标签预测

## 项目概述  
CSDN平台需要对海量文章进行高效的标签分类，但现有方法在复杂语义分析上表现不足。针对这一痛点，我们开发了一款基于深度学习的智能标签预测系统。最终，我们的模型在测试集上取得了**F1_score 0.97-0.98**的优异表现。

---

## 项目挑战与解决方案  

### 挑战 1：如何处理海量非结构化文本数据？  
**解决方案**：  
- 使用**文本增强**方法（如同义词替换、随机删除等），提升模型对语义多样性的理解能力。  
- 使用大量数据，扩大数据源

---

### 挑战 2：如何选择合适的模型框架？  
**解决方案**：  
- 选择**glm**作为语言模型基座，结合轻量化分类头完成标签预测任务。  
- 在模型训练中引入**多任务学习框架**，一边优化分类任务，一边对文本进行语言建模，提升模型整体的语义理解能力。  

---

### 挑战 3：如何优化训练与推理效率？  
**解决方案**：  
- **梯度累积**：在显存受限的环境下，通过梯度累积完成大批量训练，避免训练中断。  
- **分布式训练**：使用`DeepSpeed`库进行分布式优化，将训练效率提升2倍以上。  
- **推理阶段**：结合`ONNX`和`TorchScript`技术，部署高效模型，提高实际应用中的响应速度。

---

## 技术亮点  
1. **标签语义建模**：  
   - 在标签空间中训练独立嵌入，将预测过程转化为标签的语义检索问题，提升标签匹配的准确性。  

2. **数据分层建模**：  
   - 针对不同类别的标签使用分层建模策略，对多层标签结构（如父子标签）进行更细粒度的语义理解。  

3. **自适应阈值调整**：  
   - 在预测阶段动态调整分类阈值，使模型能够平衡精确率和召回率，满足不同应用场景需求。

---

## 部署与使用  

### 环境准备  
通过以下命令配置环境：  
```bash  
conda create -n zpqy python=3.12  
conda activate zpqy  
pip install -r requirements.txt
```

## 推理过程
在脚本开头处修改模型路径`model_path`, 运行后即可。
补充说明：
由于标签没有正负类集合，故在统计的时候将`FP`,`FN`归为一类。采用的方法为
```python
# 逐样本计算
    for i in range(len(right_ans)):
        if predict_ans[i] == right_ans[i]:
            TP += 1  # 正确匹配
        else:
            FP_FN += 1  # 错误预测
```
