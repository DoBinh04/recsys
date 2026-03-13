import pandas as pd

df = pd.read_parquet("Retrieval/data/train_ready.parquet")

print(df.head())
print(df.columns)