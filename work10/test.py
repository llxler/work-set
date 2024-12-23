from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score

######### config #########
model_path = "xxx"
test_data_path = "csdn_test.xlsx"
save_data_path = "pred_list_save.csv"
sys_prompt = "Tell me the tags based on the blog content."
######### config #########

tokenizer = AutoTokenizer.from_pretrained(
    # pretrained_model_name_or_path="Qwen-2.5-7B-Instruct",
    pretrained_model_name_or_path=model_path,
    local_files_only=True,
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    # pretrained_model_name_or_path="Qwen-2.5-7B-Instruct",
    pretrained_model_name_or_path=model_path,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True,
    trust_remote_code=True,
)

test_data = pd.read_excel(test_data_path)

def pred_list(content):
    user_prompt = content
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
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

    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return output

llm_ans = [] # 文章id， 匹配标签
for i in tqdm(range(len(test_data))):
    id = test_data.loc[i, "    Blog ID"]
    content = test_data.loc[i, "正文前 256符号"] 
    pred = pred_list(content)
    llm_ans.append([id, pred])
# 保存成csv
ans_df = pd.DataFrame(llm_ans, columns=["文章ID", "匹配标签"])
ans_df.to_csv(save_data_path, index=False, encoding='utf-8')
    

# 计算 precision, recall 和 F1 分数
right_ans = test_data["Tags"].tolist()
predict_ans = [llm_ans[i][1] for i in range(len(right_ans))]

precision, recall, f1, _ = precision_recall_fscore_support(right_ans, predict_ans, average='micro')
print(f"precision: {precision}, recall: {recall}, f1: {f1}")
