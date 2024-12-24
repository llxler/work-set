from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from sklearn.metrics import recall_score
import argparse

def calculate_metrics(true_labels, predicted_labels):
    recall = recall_score(true_labels, predicted_labels, average='micro')

    correct_count = sum(
        set(true.split(",")) == set(predicted.split(","))
        for true, predicted in zip(true_labels, predicted_labels)
    )

    precision = correct_count / len(true_labels)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1

def load_model_and_tokenizer(model_path):
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

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="model")
    parser.add_argument("--test_file", type=str, default="csdn_test.xlsx")
    parser.add_argument("--output_file", type=str, default="csdn_tags_pred.csv")
    return parser.parse_args()

def main():
    args = parse_arguments()

    model, tokenizer = load_model_and_tokenizer(args.model_path)

    test_data = pd.read_excel(args.test_file)
    predictions = []

    for _, row in test_data.iterrows():
        blog_id = row["    Blog ID"]
        content = row["正文前 256符号"]
        
        chat_input = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are an expert at categorizing tags based on blog content."},
                {"role": "user", "content": content}
            ],
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([chat_input], return_tensors="pt").to(model.device)
        generated_output = model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [output[len(input_ids):] for input_ids, output in zip(model_inputs.input_ids, generated_output)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        predictions.append([blog_id, response])

    pd.DataFrame(predictions, columns=["文章ID", "匹配标签"]).to_csv(args.output_file, index=False, encoding='utf-8')

    true_tags = test_data["Tags"].tolist()
    predicted_tags = [pred[1] for pred in predictions]
    precision, recall, f1 = calculate_metrics(true_tags, predicted_tags)

    print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")

if __name__ == "__main__":
    main()
