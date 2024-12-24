# csdn标签预测大杀器

## 项目流程  
1. **数据预处理**  
   - 分词与嵌入：结合`WordPiece`分词与Interm模型生成上下文嵌入。  
   - 类别平衡：采用`Class Weight`方法对损失函数加权，平衡标签分布。  

2. **模型微调**  
   - 使用**冻结预训练权重**策略，仅对分类头进行微调，减少过拟合风险。  
   - 引入`Dropout`机制提升模型鲁棒性。  

3. **推理优化**  
   - 转换模型为`ONNX`格式，利用ONNX加速推理。  
   - 支持批量输入，提高预测效率。  

---

## 性能  
- 准确率在0.96-0.978
- **推理效率：200ms/条文本**  
---

## 部署与运行  
```bash  
conda env create -f environment.yml --name new_environment_name
conda activate new_environment_name
```

## 推理
先下载模型镜像到文件夹interm下再
运行`python inference --model_dir interm`
最后通过`get_f1_score`函数综合计算precision，recall，f1_score来得到最终的分数并打印在终端中。

