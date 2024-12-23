from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from sklearn.metrics import recall_score

def pretrained_model():
    model_name = "your_model_name"
    model_path = "your_model_path" # TODO 请填写你的模型路径

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

    return model, tokenizer

def pred_content(content: str, model, tokenizer):
    prompt = content
    messages = [
        {"role": "system", "content": "You are Qwen. Help me reason through blog content with tags."},
        {"role": "user", "content": prompt}
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

def calculate_score(predict_ans, right_ans):
    correct = 0
    for i in range(len(right_ans)):
        if right_ans[i] == predict_ans[i]:
            correct += 1

    accuracy = correct / len(right_ans)
    recall = recall_score(right_ans, predict_ans, average='micro')
    f1_score = 2 * accuracy * recall / (accuracy + recall)
    
    return accuracy, recall, f1_score

def main():
    test_data = pd.read_excel("csdn_test.xlsx")
    right_ans = test_data["Tags"].tolist()
    
    model, tokenizer = pretrained_model()
    
    pred = [] # 文章id， 匹配标签
    for i in range(test_data.shape[0]):
        id = test_data.loc[i, "    Blog ID"]
        content = test_data.loc[i, "正文前 256符号"] 
        response = pred_content(content, model, tokenizer)
        pred.append([id, response])
    
    ans_df = pd.DataFrame(pred, columns=["文章ID", "匹配标签"])
    ans_df.to_csv("pred_ans.csv", index=False, encoding='utf-8')
        

    # 计算 precision, recall 和 F1 分数
    predict_ans = [pred[i][1] for i in range(len(right_ans))]

    accuracy, recall, f1_score = calculate_score(predict_ans, right_ans)
    
    print("precision:", accuracy)
    print("recall:", recall)
    print("f1_score:", f1_score)
    
if __name__ == '__main__':
    main()