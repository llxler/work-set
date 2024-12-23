# zbtrs - CSDN标签无敌预测器

## 模型下载
将以下四个文件下载到model目录下
* http://sou90ua1z.hn-bkt.clouddn.com/model-00001-of-00004.safetensors
* http://sou90ua1z.hn-bkt.clouddn.com/model-00002-of-00004.safetensors
* http://sou90ua1z.hn-bkt.clouddn.com/model-00003-of-00004.safetensors
* http://sou90ua1z.hn-bkt.clouddn.com/model-00004-of-00004.safetensors
* 这是我们微调的7B大语言模型

## 环境搭建
```bash
conda create -n <your_env_name> python=3.10
conda activate <your_env_name>
# 根据cuda版本下载torch，这里选择12.1
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## 训练模型
安装好llama factory
```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics,deepspeed]"
```
```bash
conda activate <your_env_name>
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path saves/Qwen2.5-7B-Instruct/full/train_2024-12-21-02-47-18 \
    --preprocessing_num_workers 16 \
    --finetuning_type full \
    --template qwen \
    --flash_attn auto \
    --dataset_dir data \
    --dataset csdn_data \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 20.0 \
    --max_samples 1000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir ./model \
    --bf16 True \
    --plot_loss True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --optim adamw_torch
```

## 模型推理
1. 推理前预准备
- 写入`model_path`，使用上面训练出的模型输出路径，或者已经训练好的模型`checkpoint`路径`model`。
- 修改`test_data_path`为要预测的表格数据，默认为`csdn_test.xlsx`，即初赛提供的表格数据。
- 输出结果表格默认为`csdn_test_ans.csv`。

2. 开始推理
直接运行`python inference.py`即可。

3. 评测指标说明
其中`precision`为初赛要求的标签预测准确率，`recall`为`sklearn`机器学习库调包得出，最后根据f1的公式计算，并输出f1_score。

4. 我们的分数
我们的最佳输出结果为
```bash
Precision:  0.981
Recall:  0.976
F1-Score:  0.9784936126724579
```
