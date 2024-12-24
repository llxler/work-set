# LLama - tags

## 复杂问题的简洁解决方案  
标签分类任务看似简单，实则复杂。它既要求对文章内容的深度理解，又需要精准提取关键特征。本项目通过LLama模型微调，将复杂问题分解为多个可控阶段，最终构建了高性能的标签分类系统。  

---

## 技术分解  
1. **阶段 1：文本特征提取**  
   - 使用`TF-IDF`作为初始筛选工具，生成初步特征矩阵。  
   - 接入LLama模型，对文章的上下文语义进行深度编码。  

2. **阶段 2：微调与适配**  
   - 微调过程中采用两种策略：  
     - **冻结基础层**：保留LLama的通用语义能力，仅调整分类相关权重。  
     - **渐进式解冻（Gradual Unfreezing）**：逐步解锁更多层以适应复杂任务。  
   - 新增自注意力（Self-Attention）机制，强化标签间的相互关系建模。  

3. **阶段 3：结果优化**  
   - 在训练后引入对抗训练（Adversarial Training），增强模型对扰动数据的鲁棒性。  
   - 通过后处理阶段的标签筛选算法，避免生成不相关的噪声标签。  

---

## 项目成效  
| **指标**        | **传统方法** | **本项目** |  
|------------------|-------------|------------|  
| Precision        | 0.85        | 0.97       |  
| Recall           | 0.82        | 0.96       |  
| F1_score         | 0.84        | 0.97       |

---

## 使用方式

### 环境搭建
```bash
conda env create -f environment.yml --name llama
conda activate llama
```

### train


### test
1. 所有的`test`的配置都在文件`inference-config.yaml`中，修改其中的内容为对应的模型路径。
```yaml
model:
  path: "model"

data:
  test_path: "csdn_test.xlsx"

output:
  result_path: "output.csv"
```
2. 然后运行命令：`python inference.py`
