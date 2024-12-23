from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd

model_path = "model"

def prepare():
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path, local_files_only=True, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path, torch_dtype="auto", device_map="auto", local_files_only=True, trust_remote_code=True,
    )

    test_data = pd.read_excel("csdn_test.xlsx")
    right_ans = test_data["Tags"].tolist()
    
    return model, tokenizer, test_data, right_ans

def predict_qwen(content, model, tokenizer):
    # 拼接
    messages = [
        {"role": "system", "content": "You are Qwen. You are a helpful assistant."},
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

def save_to_csv(ans, file_name):
    ans_df = pd.DataFrame(ans, columns=["文章ID", "匹配标签"])
    ans_df.to_csv(file_name, index=False, encoding='utf-8')

def main():
    
    model, tokenizer, test_data, right_ans = prepare()
    
    pred_ans = [] # 文章id， 匹配标签
    for i in tqdm(range(len(test_data))):
        id = test_data.loc[i, "    Blog ID"]
        content =  test_data.loc[i, "正文前 256符号"] 
        pred_ans.append([id, predict_qwen(content, model, tokenizer)])
        
    # 保存成csv
    save_to_csv(pred_ans, "pred.csv")
        
    # 计算 accuracy, recall 和 F1 分数
    predict_ans = [pred_ans[i][1] for i in range(len(right_ans))]

    # Counters for TP, FP, FN, TN
    TP, FP, FN, TN = 0, 0, 0, 0

    for i in range(len(right_ans)):
        if predict_ans[i] == right_ans[i]:
            TP += 1  # 预测正确的正类
        elif predict_ans[i] in right_ans:
            FP += 1  # 预测错误的正类
        elif right_ans[i] not in predict_ans:
            FN += 1  # 漏掉的正类
        else:
            TN += 1  # 预测正确的负类

    # Accuracy
    accuracy = (TP + TN) / len(right_ans) if len(right_ans) > 0 else 0

    # Precision and Recall
    # precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    # 在这里就是precision = acc
    precision = accuracy
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

