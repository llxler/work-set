import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# TODO 下面为运行参数,修改里面的参数即可args配置
# python inference.py --model_path model --test_data csdn_test.xlsx --output results.csv

# Argument parser
parser = argparse.ArgumentParser(description="Run LLM inference and evaluation.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model.")
parser.add_argument("--test_data", type=str, required=True, help="Path to the test data in Excel format.")
parser.add_argument("--output", type=str, default="results.csv", help="Path to save the results.")
args = parser.parse_args()

# Function to initialize model and tokenizer
def initialize_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True
    )
    return model, tokenizer

# Function to generate responses
def generate_response(model, tokenizer, content):
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
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

# Main workflow
def main():
    # Load model and tokenizer
    model, tokenizer = initialize_model_and_tokenizer(args.model_path)

    # Load test data
    test_data = pd.read_excel(args.test_data)

    # Generate results
    results = []
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Generating responses"):
        content = row["正文前 256符号"]
        response = generate_response(model, tokenizer, content)
        results.append([row["    Blog ID"], response])

    # Save results
    pd.DataFrame(results, columns=["文章ID", "匹配标签"]).to_csv(args.output, index=False, encoding="utf-8")

    # Evaluation
    y_true = test_data["Tags"].tolist()
    y_pred = [result[1] for result in results]
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')

    print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1}")

if __name__ == "__main__":
    main()