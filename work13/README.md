# 基于interm模型的超级标签分类器

## 项目成果  
- 基于interm模型结合大量爬取数据进行SFT微调
- 测试集`Precision=0.97`，`Recall=0.96`，`F1_score=0.97`。  
- 推理延迟：平均单条文本耗时357ms。  

---

### 使用方法  
```bash  
conda env create -f environment.yml
conda activate interm
```
若您的`cuda`版本低于12.1，请修改一下环境里面的torch版本配置

---
### 推理流程
运行命令
```bash
conda activate interm
python test.py --model_dir model
```
推理逻辑
```python
# 计算标签准确率precision，忽略标签顺序
    acc = 0
    for i in range(len(y)):
        right_tags = y[i].split(",")
        predict_tags = y_hat[i].split(",")
        if set(right_tags) == set(predict_tags):
            acc += 1
```
忽略了标签的顺序，一一对应的话，则匹配度+1，`recall`采用`sklearn.metrics`来计算。