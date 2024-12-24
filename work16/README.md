
# llama is all you need

## 对比分析  
### 传统方法  
- **TF-IDF + SVM**：  
  - 优点：实现简单，计算开销小。  
  - 缺点：无法捕获上下文语义，准确率低。  

- **LSTM**：  
  - 优点：能够建模序列信息。  
  - 缺点：训练速度慢，对长文本处理效果较差。  

### INTERM模型  
- **优势**：  
  - 提供深度语义理解能力，适合处理复杂的长文本。  
  - 支持大规模并行化训练，效率显著提升。  

---

## 性能对比  
| 方法      | Precision | Recall | F1_score |  
|-----------|-----------|--------|----------|  
| SVM       | 0.87      | 0.82   | 0.84     |  
| LSTM      | 0.90      | 0.88   | 0.89     |  
| INTERM    | 0.98      | 0.96   | 0.97     |  

---

## 使用指南  

### 环境搭建
```bash  
conda env create -f environment.yml
conda activate lxl
```
### 训练配置


### 推理测试
#### 评判标准
`precision_recall_fscore_support`

#### 评判流程
修改`main.py`中的`model_directory = "your_model_path"`为安装后的模型的路径。
然后运行`python test.py`即可。
