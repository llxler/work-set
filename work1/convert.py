import pandas as pd

# 读取Excel文件
df = pd.read_excel('./csdn_test.xlsx')

new_df = pd.DataFrame({
    'output': df.iloc[1:, 3].reset_index(drop=True),  # 第4列，跳过第一行
    'input': df.iloc[1:, 4].reset_index(drop=True)    # 第5列，跳过第一行
})

new_df.to_parquet('train.parquet')
