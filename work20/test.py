import torch
import argparse
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support

# 整理数据，并将数据交付给模型
def process_data(test_file, model, tokenizer):
    test_data = pd.read_excel(test_file)
    predictions = []
    
    for idx, row in tqdm(test_data.iterrows(), total=test_data.shape[0], desc="Processing Rows"):
        content = row["正文前 256符号"]
        prediction = generate_label(model, tokenizer, content)
        predictions.append([row["    Blog ID"], prediction])
    
    return predictions, test_data["Tags"].tolist()

# Save Predictions
def save_predictions(predictions, output_file):
    pd.DataFrame(predictions, columns=["文章ID", "预测标签"]).to_csv(output_file, index=False, encoding="utf-8")

# Evaluate Results
def evaluate_predictions(y_true, predictions):
    y_pred = [item[1] for item in predictions]
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="micro")
    return precision, recall, f1

# Load Model and Tokenizer
def load_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_path,
        local_files_only=True,
        trust_remote_code=True
    )
    return model, tokenizer

def generate_label(model, tokenizer, content):
    content = content
    messages = [
        {"role": "system", "content": "Tell me the tags based on the blog content."},
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

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

parser = argparse.ArgumentParser(description="CSDN Tags Predict Pipeline")
parser.add_argument("--model_path", default="download_path", help="Path to the model directory")
parser.add_argument("--test_file", default="csdn_test.xlsx", help="Path to the test data file")
parser.add_argument("--output_file", default="predictions.csv", help="File to save predictions")
args = parser.parse_args()

def main():
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    predictions, y_true = process_data(args.test_file, model, tokenizer)
    save_predictions(predictions, args.output_file)
    precision, recall, f1 = evaluate_predictions(y_true, predictions)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")    

if __name__ == "__main__":
    main()
