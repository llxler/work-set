from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from sklearn.metrics import recall_score

def predict(prompt: str, model, tokenizer):
    prompt = prompt
    messages = [
        {"role": "system", "content": " Help me reason through blog content with tags."},
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

def cal(pred_ans, ori_ans):
    acc = 0
    for i in range(len(ori_ans)):
        if ori_ans[i] == pred_ans[i]:
            acc += 1

    accuracy = acc / len(ori_ans)
    recall = recall_score(ori_ans, pred_ans, average='micro')
    f1_score = 2 * accuracy * recall / (accuracy + recall)
    
    print("precision:", accuracy)
    print("recall:", recall)
    print("f1_score:", f1_score)
    
    return accuracy, recall, f1_score

def load_model(model_path):

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

def main():
    model_path = ""
    test_data = pd.read_excel("csdn_test.xlsx")
    
    model, tokenizer = load_model(model_path)
    
    pred = [] # 文章id， 匹配标签
    for i in range(test_data.shape[0]):
        id = test_data.loc[i, "    Blog ID"]
        prompt = test_data.loc[i, "正文前 256符号"] 
        response = predict(prompt, model, tokenizer)
        pred.append([id, response])
    
    ans_df = pd.DataFrame(pred, columns=["文章ID", "匹配标签"])
    ans_df.to_csv("pred_ans.csv", index=False, encoding='utf-8')
        

    # 计算 precision, recall 和 F1 分数
    ori_ans = test_data["Tags"].tolist()
    pred_ans = [pred[i][1] for i in range(pred.shape[0])]

    accuracy, recall, f1_score = cal(pred_ans, ori_ans)

if __name__ == '__main__':
    main()