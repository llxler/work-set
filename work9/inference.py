from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from sklearn.metrics import recall_score

def get_score(predict_ans, right_ans):
    # 初始化计数器
    TP, FP_FN = 0, 0

    # 逐样本计算
    for i in range(len(right_ans)):
        if predict_ans[i] == right_ans[i]:
            TP += 1  # 正确匹配
        else:
            FP_FN += 1  # 错误预测

    # Precision 和 Recall
    precision = TP / (TP + FP_FN) if (TP + FP_FN) > 0 else 0
    recall = TP / (TP + FP_FN) if (TP + FP_FN) > 0 else 0

    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


def tag_pred(content: str, model, tokenizer):
    content = "根据博客内容预测其标签。\n" + content
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

def qwen_output():
    test_data = pd.read_excel("csdn_test.xlsx")
    right_ans = test_data["Tags"].tolist()
    
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_path = "xxx" # TODO 请填写你的模型路径

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
    
    tag_list = [] # 文章id， 匹配标签
    for i in range(test_data.shape[0]):
        id = test_data.loc[i, "    Blog ID"]
        content = test_data.loc[i, "正文前 256符号"] 
        response = tag_pred(content, model, tokenizer)
        tag_list.append([id, response])
    
    return tag_list

def main():
    test_data = pd.read_excel("csdn_test.xlsx")
    right_ans = test_data["Tags"].tolist()
    
    tag_list = qwen_output()
    
    pred_pd = pd.DataFrame(tag_list, columns=["文章ID", "匹配标签"])
    pred_pd.to_csv("pred_ans.csv", index=False, encoding='utf-8')
        
    # 计算 precision, recall 和 F1 分数
    predict_ans = [tag_list[i][1] for i in range(len(right_ans))]

    f1_score = get_score(predict_ans, right_ans)
    
    print("f1_score:", f1_score)
    
if __name__ == '__main__':
    main()