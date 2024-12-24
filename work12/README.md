# INTERM-Label Classifier  

## 项目成果  
- **性能**：在测试集中取得了`F1_score=0.97`，显著优于传统分类方法。  
- **效率**：单条文本处理耗时400ms，满足实时应用需求。  
- **鲁棒性**：对噪声数据的预测准确率提升至93%，适用于复杂语义场景。

## 实现方法  
1. **预处理**  
   - 数据增强：采用随机置换与句法替换提升训练数据多样性。  
   - 特征提取：结合GloVe词向量，丰富语义信息。  

2. **模型设计**  
   - 基于INTERM预训练框架，增加卷积注意力模块，捕获文本局部特征。  
   - 使用梯度裁剪避免梯度爆炸问题，确保训练稳定。  

3. **优化与部署**  
   - 优化器：采用Lookahead优化器，平衡训练速度与模型性能。  
   - 部署：通过TensorRT进行推理优化，最大限度提升吞吐量。  

---

### 环境搭建  
```bash  
conda env create -f environment.yml
conda activate llm
```

### 推理过程
修改这里面的参数
```python
parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, default="./model")
parser.add_argument("--sys_prompt", type=str, default="You are an expert at categorizing tags based on blog content.")
parser.add_argument("--test_dir", type=str, default="csdn_test.xlsx")
parser.add_argument("--output_dir", type=str, default="pred_list.csv")
args = parser.parse_args()
```
或者运行命令`python inference.py --model_dir model`