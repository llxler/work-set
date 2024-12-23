from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

model_path = "./model"
test_data_path = "csdn_test.xlsx"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
    trust_remote_code=True,
)

test_data = pd.read_excel(test_data_path)

def llm_inference(content: str):
    messages = [
        {"role": "system", "content": "You are Qwen. You are an expert at categorizing tags based on blog content."},
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

    return response

ans = [] # 文章id， 匹配标签
for i in tqdm(range(len(test_data))):
    id = test_data.loc[i, "    Blog ID"]
    content = test_data.loc[i, "正文前 256符号"] 
    response = llm_inference(content)
    ans.append([id, response])
# 保存成csv
ans_df = pd.DataFrame(ans, columns=["文章ID", "匹配标签"])
ans_df.to_csv("csdn_test_ans.csv", index=False, encoding='utf-8')

# 计算 precision, recall 和 F1 分数
right_ans = test_data["Tags"].tolist()
predict_ans = [ans[i][1] for i in range(len(right_ans))]

# 转换为 one-hot 格式或直接处理多标签
precision, recall, f1, _ = precision_recall_fscore_support(right_ans, predict_ans, average='micro')

# 计算标签准确率precision
correct = 0
for i in range(len(right_ans)):
    if right_ans[i] == predict_ans[i]:
        correct += 1
    else:
        # 拆开tag
        right_tags = right_ans[i].split(",")
        predict_tags = predict_ans[i].split(",")
        # 忽略标签顺序，如果一样的就算对
        if set(right_tags) == set(predict_tags):
            correct += 1
        
precision = correct / len(right_ans)

f1_score = 2 * precision * recall / (precision + recall)

print(f"Precision: {precision}, Recall: {recall}, F1: {f1_score}")