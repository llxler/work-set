from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./model")
    parser.add_argument("--test_data_path", type=str, default="csdn_test.xlsx")
    parser.add_argument("--output_path", type=str, default="csdn_ans.csv")
    args = parser.parse_args()
    
    return args

def test(content, model, tokenizer):
    messages = [
        {"role": "system", "content": "You are Interm. You are an expert at categorizing tags based on blog content."},
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

def main():
    args = get_args()
    
    model_path = args.model_path
    test_data_path = args.test_data_path
    output_path = args.output_path
    
    test_data = pd.read_excel(test_data_path)
    
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

    ans_list_pred = []
    for i in tqdm(range(len(test_data))):
        id = test_data.loc[i, "    Blog ID"]
        content = test_data.loc[i, "正文前 256符号"] 
        response = test(content, model, tokenizer)
        ans_list_pred.append([id, response])
    
    ans_df = pd.DataFrame(ans_list_pred, columns=["文章ID", "匹配标签"])
    ans_df.to_csv(output_path, index=False, encoding='utf-8')

    # 计算 precision, recall 和 F1 分数
    right_ans = test_data["Tags"].tolist()
    predict_ans = [ans_list_pred[i][1] for i in range(len(right_ans))]

    # 转换为 one-hot 格式或直接处理多标签
    precision, recall, f1, _ = precision_recall_fscore_support(right_ans, predict_ans, average='micro')

    # 计算标签准确率precision
    correct = 0
    for i in range(len(right_ans)):
        right_tags = right_ans[i].split(",")
        predict_tags = predict_ans[i].split(",")
        if set(right_tags) == set(predict_tags):
            correct += 1
            
    precision = correct / len(right_ans)

    f1_score = 2 * precision * recall / (precision + recall)

    print(f"Precision: {precision}, Recall: {recall}, F1: {f1_score}")
    
if __name__ == '__main__':
    main()