import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import kagglehub
import os




# Download latest version
# path = kagglehub.dataset_download("mrsimple07/energy-consumption-prediction")
# # read file 
# csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
# df = pd.read_csv(os.path.join(path, csv_file))
# print(df.head(3))



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
            "missing count:", missing_counts,
            "missing pct:" , missing_pct
        
        })
        report = report[report["missing_count:" ]> 0]

        if report.empty:
            print("no missing values")
        else: 
            print(report)
        return report 


# 결측치를 soo 가 다 없애서 주는걸로 하고 위의 두 함수는 전체적인 데이터 보여주기 
# 결측치가 없는거 다시 한번 확인하기 위한 class 



class Dataprocessor:
    def __init__(self, df:pd.DataFrame , target: str = "EnergyComsuption"):
        self.df = df.copy() # 복사본을 만들면서 원본 보호 
        self.target = target 


    # 자료형 -> 수치형으로 바꿔서 모델이 사용 가능하도록 변경 
    # 가장 에너지 사용량에 변경을 주는 낮, 새벽 시간대 + 계절 여름겨울 을 숫자로 뽑아내는 과정 
    def add_time_features (self, time_col: str = "Timestamp",
                           fmt : str = "%d-%m-%Y %H:%M") -> pd.DataFrame:
        self.df[time_col] = pd.to_datetime(self.df[time_col], format = fmt)
        self.df = self.df.sort_values(time_col)
        self.df["Hour"] = self.df[time_col].dt.hour
        # dt = 날짜 타입 columns 에서 year, month, day 를 꺼낼 때 쓰는 통로 
        # 변경 안해주면 error 나기에 무조건 해줘야함 
        self.df["Month"] = self.df[time_col].dt.month
        self.df = self.df.drop(columns=[time_col])  
        # 원래 날짜 컬럼 제거 

        return self.df


    # 문자열 -> 수치형으로 바꿔주는 과정 
    # on/off 같은 경우는 1/0 으로 인코드 
    def encode_categricals(self) -> pd.DataFrame:

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
            if "DayOfWeek" is self.df.columns:
                self.df = pd.get_dummies(self.df, columns=["DayOfWeek"], prefix = "Day",
                                         drop_first=True, dtype=int)
                print("  One-hot encoded 'DayOfWeek' (drop_first=True to avoid redundancy)")
            else:
                print("  Warning: 'DayOfWeek' column not found for one-hot encoding.")

            self.features = [c for c in self.df.columns if c != self.target]  
            return self.df


class Dataspliting:
    def __init__(self, df: pd.DataFrame, target: str = "EnergyComsuption"):
        self.df = df.copy()
        self.target = target
        self.features = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

    def split(self, test_size : float = 0.2):

        if self.feature is None:
            raise RuntimeError("Run encode_categoricals() before split().")
            # 인코딩 전에 부르면 막기 -> 안전장치와 같음 

        X = self.df[self.features]
        Y = self.df[self.target]

        self.X_train, self.X_test, self.y_train , self.y_test = train_test_split(
            X, Y , test_size= test_size, shuffle= False)

        print(f"  Train: {len(self.X_train)} rows, Test: {len(self.X_test)} rows")
        return self.X_train, self.X_test, self.y_train, self.y_test



    def process(self, test_size: float = 0.2):
        """세 단계를 한 번에 실행."""
        self.add_time_features() # timestamp -> hour, month 나누는 함수 
        self.encode_categoricals() # 인코딩 글자 -> 숫자 
        return self.split(test_size)







# df = pd.read_csv("soo_File name.csv")
# Dateload = Dataloader(df)
# Dateload.overview(10)
# report = Dataloader.missing_report() # 객체 생성
# pre_data = Dataprocessor(Dateload)
# pre_data.add_time_features()
# lastdata = Dataspliting()
# X_train, X_test, y_train, y_test = lastdata.process()







