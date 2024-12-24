import os
import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

# Load configuration
CONFIG_PATH = "inference-config.yaml"
def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

config = load_config(CONFIG_PATH)

# Load test data
def load_test_data(file_path):
    return pd.read_excel(file_path)

# Save results
def save_results(results, output_path):
    results_df = pd.DataFrame(results, columns=["文章ID", "匹配标签"])
    results_df.to_csv(output_path, index=False, encoding="utf-8")

# Evaluate results
def evaluate(test_data, predictions):
    y_true = test_data["Tags"].tolist()
    y_pred = predictions
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# Initialize model and tokenizer
def initialize_model_and_tokenizer(config):
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['path'],
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['path'],
        local_files_only=True,
        trust_remote_code=True
    )
    return model, tokenizer

# Perform inference
def infer(model, tokenizer, content):
    prompt = content
    messages = [
        {"role": "system", "content": "Your task is to predict tags based on content."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    return tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

# Main workflow
def main():
    model, tokenizer = initialize_model_and_tokenizer(config)
    test_data = load_test_data(config['data']['test_path'])

    results = []
    for _, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Processing"):
        content = row["正文前 256符号"]
        response = infer(model, tokenizer, content)
        results.append([row["    Blog ID"], response])

    save_results(results, config['output']['result_path'])
    evaluate(test_data, [result[1] for result in results])

if __name__ == "__main__":
    main()