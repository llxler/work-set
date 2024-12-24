from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from sklearn.metrics import recall_score
import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="The model directory")
    parser.add_argument("--test_dir", type=str, default="csdn_test.xlsx")
    parser.add_argument("--output_dir", type=str, default="csdn_tags_pred.csv")
    args = parser.parse_args()
    
    return args

def get_model_tokenizer(args):
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir,
        local_files_only=True,
        trust_remote_code=True,
    )
    
    return model, tokenizer

def get_f1_score(y, y_hat):
    recall= recall_score(y, y_hat, average='micro')

    accuracy = 0
    for i in range(len(y)):
        right_tags = y[i].split(",")
        predict_tags = y_hat[i].split(",")
        if set(right_tags) == set(predict_tags):
            accuracy += 1
            
    precision = accuracy / len(y)

    f1_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f1_score


def main():
    args = args_parser()

    model, tokenizer = get_model_tokenizer(args)

    test_data = pd.read_excel(args.test_dir)
    llm_pred_list = []
    
    # 遍历表格，并推理答案
    for _, row in test_data.iterrows():
        id = row["    Blog ID"]
        content = row["正文前 256符号"]
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
        
        llm_pred_list.append([id, response])

    df = pd.DataFrame(llm_pred_list, columns=["文章ID", "匹配标签"])
    df.to_csv(args.output_dir, index=False, encoding='utf-8')

    # 计算 precision, recall 和 F1 分数
    right_ans = test_data["Tags"].tolist()
    pred_list = [llm_pred_list[i][1] for i in range(len(right_ans))]

    precision, recall, f1_score = get_f1_score(right_ans, pred_list)
    print(f"Precision: {precision}, Recall: {recall}, F1: {f1_score}")