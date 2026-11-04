import kagglehub
import numpy as np
import pandas as pd
import os



# Download latest version
path = kagglehub.dataset_download("mrsimple07/energy-consumption-prediction")

print("Path to dataset files:", path)
print("list of files:", os.listdir(path))

# read file 
csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
df = pd.read_csv(os.path.join(path, csv_file))

print(df.head(3))

# 가지고 있는 데이터 col 의 결과 보기 
print(df.columns)
print("//")
for col in df.columns:
    print(f"{col} ({df[col].nunique()}): {df[col].unique()[:10]}")