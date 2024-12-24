# llama模型预测器
## 项目简介  
本项目基于 Llama 模型，致力于实现高效的 CSDN 文章标签分类任务。通过定制化的训练策略和优化手段，成功构建了一个高性能的分类器，为文本分类任务提供了一种稳定且高效的解决方案。  

---

## 核心技术  
### 1. 数据处理  
- **语义清洗**：移除停用词和冗余符号，同时对拼写错误进行智能纠正，确保数据的清晰度。  
- **数据增强**：结合同义词替换与反向翻译技术，扩充训练集规模并提升模型对稀疏标签的泛化能力。  

### 2. 模型微调  
- **逐层解冻**：  
  - 初期冻结 Llama 模型的大部分参数，专注于训练分类头（Classification Head）。  
  - 随着训练进行，逐步解冻中高层权重，使模型更好地捕获语义信息。  
- **层次注意力**：  
  - 在分类层中引入多头注意力机制，进一步增强对上下文信息的理解。  

### 3. 训练优化  
- **动态学习率**：自适应调整学习率，提升收敛速度。  
- **权重归一化**：降低训练过程中的参数抖动，提高模型的稳定性。  

---

## 快速开始  
### 环境配置  
```bash  
conda env create -f environment.yml --name <env>
conda activate <env>
```

### 训练环节


### 测试环节
在`test.py`69行数开始修改一下配置
```python
parser = argparse.ArgumentParser(description="CSDN Tags Predict Pipeline")
parser.add_argument("--model_path", default="download_path", help="Path to the model directory")
parser.add_argument("--test_file", default="csdn_test.xlsx", help="Path to the test data file")
parser.add_argument("--output_file", default="predictions.csv", help="File to save predictions")
args = parser.parse_args()
```
**特别是**修改`model_path`路径为train后路径，或者下载的镜像路径。

然后运行`python test.py`
