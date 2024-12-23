import pandas as pd

# 读取Excel文件
df = pd.read_excel('./csdn_test.xlsx')

# 创建新的DataFrame，只包含所需的列，并跳过第一行
new_df = pd.DataFrame({
    'output': df.iloc[1:, 3].reset_index(drop=True),  # 第4列，跳过第一行
    'input': df.iloc[1:, 4].reset_index(drop=True)    # 第5列，跳过第一行
})

# 保存为parquet文件
new_df.to_parquet('train.parquet')

# 打印确认信息
print("转换完成！")
print(f"数据行数: {len(new_df)}")
print("\n前几行数据预览:")
print(new_df.head())
