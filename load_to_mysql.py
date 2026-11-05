import sqlite3
import pandas as pd
from db import engine 

INPUT_CSV_PATH = r"C:\flask_app\Rawenergy_consumption.csv"   
OUTPUT_DB_PATH = r"C:\flask_app\Smartenergy_comsumption.db" 

# read the input CSV 
input_df = pd.read_csv(INPUT_CSV_PATH) 
# read the output database
sqlite_conn = sqlite3.connect(OUTPUT_DB_PATH)

# print some date for checking 
energy_logs_df = pd.read_sql("SELECT * FROM energy_logs", sqlite_conn) 
sqlite_conn.close()

input_df.to_sql("raw_input", engine, if_exists="replace", index=False)
energy_logs_df.to_sql("smartenergy_logs", engine, if_exists="replace", index=False)

print(f"MySQL save: raw_input {len(input_df)}rows, smartenergy_logs {len(energy_logs_df)}rows")
