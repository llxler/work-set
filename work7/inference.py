from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

def get_input(content, model, tokenizer):
    
    messages = [
        {"role": "system", "content": "You're a master at giving tags based on content."},
        {"role": "user", "content": content}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    return model_inputs

def llmpred(content, model, tokenizer):
    
    model_inputs = get_input(content, model, tokenizer)

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
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model_path = "xxx"
    test_data = pd.read_excel("csdn_test.xlsx")
    
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
        local_files_only=True,
        trust_remote_code=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True,
    )
    
    pred_list = [] # 文章id， 匹配标签
    for i in range(len(test_data)):
        id = test_data.loc[i, "    Blog ID"]
        prompt = "根据博客内容预测其标签。\n" + test_data.loc[i, "正文前 256符号"]
        response = llmpred(prompt, model, tokenizer)
        pred_list.append([id, response])
    
    save_df = pd.DataFrame(pred_list, columns=["文章ID", "匹配标签"])
    save_df.to_csv("csdn_test_ans.csv", index=False, encoding='utf-8')
        
    # 计算 precision, recall 和 F1 分数
    right_ans = test_data["Tags"].tolist()
    predict_ans = [pred_list[i][1] for i in range(len(right_ans))]

    precision, recall, f1, _ = precision_recall_fscore_support(right_ans, predict_ans, average='micro')

    print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

if __name__ == '__main__':
    main()