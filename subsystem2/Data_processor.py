import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import kagglehub
import os
from pathlib import Path   
import sqlite3


# loading data from csv file and return the fully processed DataFrame

class Dataloader:
    def __init__(self, df:pd.DataFrame):
        self.df= df

    def overview(self) -> None:
        print(f"shape: {self.df.shape[0]}rows, {self.df.shape[1]}columns")
        print("column data type:" , self.df.dtypes)
        print("first five head: " , self.df.head())

    def missing_report(self) -> pd.DataFrame:
        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts/ len(self.df) *100).round(2)
        report = pd.DataFrame({
            "missing_counts:" : missing_counts,
            "missing_pct:" : missing_pct
        
        })
        report = report[report["missing_counts:" ]> 0]

        if report.empty:
            print("no missing values")
        else: 
            print(report)
        return report 




class Dataprocessor:
    def __init__(self, df:pd.DataFrame , target: str = "EnergyConsumption"):
        self.df = df.copy() # Making copy to protect original data
        self.target = target 

 
    def add_time_features (self, time_col: str = "Timestamp",
                           fmt : str = "%Y-%m-%d %H:%M:%S") -> pd.DataFrame:
        self.df[time_col] = pd.to_datetime(self.df[time_col], format = fmt)
        # string to change date oject with pd(pandas) to calculate hour and month
        self.df = self.df.sort_values(time_col)
        self.df["Hour"] = self.df[time_col].dt.hour
        # dt = 날짜 타입 columns 에서 year, month, day 를 꺼낼 때 쓰는 통로 
        # 변경 안해주면 error 나기에 무조건 해줘야함 
        self.df["Month"] = self.df[time_col].dt.month
        self.df = self.df.drop(columns=[time_col])  
        # delete previous columns 

        return self.df


    # 문자열 -> 수치형으로 바꿔주는 과정 
    # on/off 같은 경우는 1/0 으로 인코드 
    def encode_categoricals(self) -> pd.DataFrame:

        binary_maps = {
               "HVACUsage": {"On": 1, "Off": 0},
               "LightingUsage": {"On": 1, "Off": 0},
               "Holiday": {"Yes": 1, "No": 0},
        }
        for col, mapping in binary_maps.items():
            # binary_map 을 보면 "HVACUsage": {"On": 1, "Off": 0} 으로 지정했음 
            if col not in self.df.columns:
                print(f"  Warning: Column '{col}' not found for binary encoding.")
                continue
            if pd.api.types.is_numeric_dtype(self.df[col]):
                print(f"  '{col}' is already numeric, skipped.")
                continue
            # 이때 on 은 1 로 off 는 0으로 변경됨 
            self.df[col] = self.df[col].map(mapping)
            if self.df[col].isnull().any():                   
                raise ValueError(f"'{col}' has values not in {list(mapping)}")
            print(f"  Encoded '{col}' as binary (1/0)")
            # 값이 여러개인 컬럼 (요일) 은 one-hot 인코드로 monday = 1 이 되도록 인코딩 
        
        
        if "DayOfWeek" in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=["DayOfWeek"], prefix = "Day",
                                     drop_first=True, dtype=int)                
            print("  One-hot encoded 'DayOfWeek' (drop_first=True to avoid redundancy)")
        else:
            print("  Warning: 'DayOfWeek' column not found for one-hot encoding.")

            self.features = [c for c in self.df.columns if c != self.target] 
        if "building_id" in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=["building_id"], prefix="Bld",
                                     drop_first=False, dtype=int)
            print("  One-hot encoded 'building_id'") 
        return self.df

    
    def process(self) -> pd.DataFrame:
        self.add_time_features()     # timestamp -> hour, month
        self.encode_categoricals()   # 글자 -> 숫자
        return self.df
    

    


class Dataspliting:
    def __init__(self, df: pd.DataFrame, target: str = "EnergyConsumption"):
        self.df = df.copy()
        self.target = target
        # target 을 뺀 나머지 컬럼이 전부 특성(feature)
        self.features = [c for c in self.df.columns if c != self.target]
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

    def split(self, test_size: float = 0.2):
        # 인코딩 안 된 데이터가 들어오면 막기 -> 안전장치
        non_numeric = self.df[self.features].select_dtypes(exclude="number").columns
        if len(non_numeric) > 0:
            raise RuntimeError(f"Non-numeric columns left: {list(non_numeric)}. "
                               "Run Dataprocessor.process() first.") 

        X = self.df[self.features]
        y = self.df[self.target]

        self.X_train, self.X_test, self.y_train , self.y_test = train_test_split(
            X, y , test_size= test_size, shuffle= False)

        print(f"  Train: {len(self.X_train)} rows, Test: {len(self.X_test)} rows")
        return self.X_train, self.X_test, self.y_train, self.y_test






CSV_PATH = "source/enriched_energy_data.csv" 


def load_clean_data(csv_path: str = CSV_PATH) -> pd.DataFrame:
    """Load data from the CSV file and return the fully processed DataFrame."""

    # 1) CSV 에서 원본 읽기
    raw_df = pd.read_csv(csv_path)
    

    # 2) Dataprocessor 객체를 만들어서 process() 실행
    processor = Dataprocessor(raw_df)           # raw_df 를 넘겨 객체 생성
    clean_df = processor.process()              # 처리 끝난 df 를 받음

    # 3) 완성된 데이터 반환
    return clean_df








