from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd

model_path = "xxx"

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path, local_files_only=True, trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_path, torch_dtype="auto", device_map="auto", local_files_only=True, trust_remote_code=True,
)

test_data = pd.read_excel("csdn_test.xlsx")

ans = [] # 文章id， 匹配标签
for i in tqdm(range(len(test_data))):
    id = test_data.loc[i, "    Blog ID"]
    content = test_data.loc[i, "正文前 256符号"] 
    
    content = "根据博客内容预测其标签。\n" + content
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": content}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    ans.append([id, response])
    
# 保存成csv
ans_df = pd.DataFrame(ans, columns=["文章ID", "匹配标签"])
ans_df.to_csv("csdn_test_ans.csv", index=False, encoding='utf-8')
    

# 计算 accuracy, recall 和 F1 分数
right_ans = test_data["Tags"].tolist()
predict_ans = [ans[i][1] for i in range(len(right_ans))]

correct = 0
for i in range(len(right_ans)):
    if right_ans[i] == predict_ans[i]:
        correct += 1

accuracy = correct / len(right_ans)
print("precision:", accuracy)

# True Positive (TP), False Positive (FP), and False Negative (FN) counters
TP, FP, FN = 0, 0, 0

for i in range(len(right_ans)):
    if predict_ans[i] == right_ans[i]:
        TP += 1  # 正确预测
    elif predict_ans[i] in right_ans:
        FP += 1  # 错误预测为正类
    else:
        FN += 1  # 漏掉的正类

recall = TP / (TP + FN) if (TP + FN) > 0 else 0

# F1 Score
f1_score = 2 * (accuracy * recall) / (accuracy + recall) if (accuracy + recall) > 0 else 0

print(f"Precision: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")
