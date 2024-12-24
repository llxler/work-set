from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

def prepare_input_data(content, tokenizer, model):
    input_structure = [
        {"role": "system", "content": "You're a master at giving tags based on content."},
        {"role": "user", "content": content}
    ]
    chat_template = tokenizer.apply_chat_template(
        input_structure,
        tokenize=False,
        add_generation_prompt=True
    )
    return tokenizer([chat_template], return_tensors="pt").to(model.device)

def predict_tags(content, model, tokenizer):
    inputs = prepare_input_data(content, tokenizer, model)
    outputs = model.generate(**inputs, max_new_tokens=512)
    decoded_ids = [output[len(input_ids):] for input_ids, output in zip(inputs.input_ids, outputs)]
    return tokenizer.batch_decode(decoded_ids, skip_special_tokens=True)[0]

def process_predictions(dataframe, model, tokenizer):
    predictions = []
    for _, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        blog_id = row["    Blog ID"]
        input_content = "根据博客内容预测其标签。\n" + row["正文前 256符号"]
        predicted_tags = predict_tags(input_content, model, tokenizer)
        predictions.append([blog_id, predicted_tags])
    return pd.DataFrame(predictions, columns=["文章ID", "匹配标签"])

def evaluate_predictions(true_labels, predicted_labels):
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, predicted_labels, average='micro')
    return precision, recall, f1

def main():
    model_directory = "your_model_path"
    input_file = "csdn_test.xlsx"
    output_file = "output.csv"

    tokenizer = AutoTokenizer.from_pretrained(
        model_directory,
        local_files_only=True,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_directory,
        torch_dtype="auto",
        device_map="auto",
        local_files_only=True,
        trust_remote_code=True
    )

    test_data = pd.read_excel(input_file)
    predictions_df = process_predictions(test_data, model, tokenizer)
    predictions_df.to_csv(output_file, index=False, encoding='utf-8')

    actual_tags = test_data["Tags"].tolist()
    predicted_tags = predictions_df["匹配标签"].tolist()
    precision, recall, f1_score = evaluate_predictions(actual_tags, predicted_tags)

    print(f"Precision: {precision}, Recall: {recall}, F1-Score: {f1_score}")

if __name__ == "__main__":
    main()