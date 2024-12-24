import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_score, recall_score

MODEL_NAME = "model_name"
MODEL_PATH = "xxx"
TEST_DATA_PATH = "csdn_test.xlsx"
OUTPUT_PATH = "output_pred_ans.csv"

def process_test_data(test_data_path, model, tokenizer):
    test_data = pd.read_excel(test_data_path)
    results = []
    
    for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Processing test data"):
        content = row["正文前 256符号"]
        response = pred_llm(model, tokenizer, content)
        results.append([row["    Blog ID"], response])

    return test_data, results

def pred_llm(model, tokenizer, content):
    prompt = content
    messages = [
        {"role": "system", "content": "Your task is to predict tags based on content."},
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
    response = tokenizer.batch_decode(generated_ids[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return response

def load_model_and_tokenizer(model_path):
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

def save_results(results, output_path):
    ans_df = pd.DataFrame(results, columns=["文章ID", "匹配标签"])
    ans_df.to_csv(output_path, index=False, encoding='utf-8')

def evaluate_predictions(test_data, results):
    right_ans = test_data["Tags"].tolist()
    predict_ans = [result[1] for result in results]
    accuracy = sum(1 for true, pred in zip(right_ans, predict_ans) if true == pred) / len(right_ans)

    recall = recall_score(right_ans, predict_ans, average='micro')
    f1 = 2 * accuracy * recall / (accuracy + recall)

    print("precision:", accuracy)
    print("recall:", recall)
    print("f1_score:", f1)

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

    # Process data and save results
    test_data, results = process_test_data(TEST_DATA_PATH, model, tokenizer)
    save_results(results, OUTPUT_PATH)

    # Evaluate performance
    evaluate_predictions(test_data, results)
