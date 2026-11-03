import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Dataloader:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def overview(self) -> None:
        print(f"shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        print("column data types:", self.df.dtypes)
        print("first five head:", self.df.head())

    def missing_report(self) -> pd.DataFrame:
        missing_counts = self.df.isnull().sum()
        missing_pct = (missing_counts / len(self.df) * 100).round(2)
        report = pd.DataFrame({
            "missing_count": missing_counts,
            "missing_pct": missing_pct
        })
        report = report[report["missing_count"] > 0]

        if report.empty:
            print("No missing values")
        else:
            print(report)
        return report 


class Dataprocessor:
    def __init__(self, df: pd.DataFrame, target: str = "EnergyConsumption"):
        self.df = df.copy()  # 원본 보호
        self.target = target 
        self.features = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

    def add_time_features(self, time_col: str = "Timestamp", fmt: str = "%d-%m-%Y %H:%M") -> None:
        """timestamp -> hour, month 나누는 함수"""
        if time_col in self.df.columns:
            self.df[time_col] = pd.to_datetime(self.df[time_col], format=fmt, errors='coerce')
            self.df = self.df.sort_values(time_col)
            self.df["Hour"] = self.df[time_col].dt.hour
            self.df["Month"] = self.df[time_col].dt.month
            self.df = self.df.drop(columns=[time_col])  # 원래 날짜 컬럼 제거

    def encode_categoricals(self) -> None:
        """문자열 -> 수치형으로 바꿔주는 과정 (Binary, One-Hot Encoding & Text Dropping)"""
        binary_maps = {
            "HVACUsage": {"On": 1, "Off": 0},
            "LightingUsage": {"On": 1, "Off": 0},
            "Holiday": {"Yes": 1, "No": 0},
        }
        for col, mapping in binary_maps.items():
            if col not in self.df.columns:
                print(f"  Warning: Column '{col}' not found for binary encoding.")
                continue
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            
            self.df[col] = self.df[col].map(mapping)
            if self.df[col].isnull().any():             
                raise ValueError(f"'{col}' has values not in {list(mapping)}")

        # One-hot encode categorical text columns (like building_type, DayOfWeek)
        categorical_cols = ["DayOfWeek", "building_type"]
        existing_cats = [c for c in categorical_cols if c in self.df.columns]
        
        if existing_cats:
            self.df = pd.get_dummies(self.df, columns=existing_cats, prefix=existing_cats, drop_first=True, dtype=int)
            print(f"  One-hot encoded categorical columns: {existing_cats}")

        # Drop non-numeric identifier columns that cannot be converted to floats (like building_id)
        if "building_id" in self.df.columns:
            self.df = self.df.drop(columns=["building_id"])
            print("  Dropped 'building_id' column to prevent string-to-float errors.")

        # 정의된 feature 목록 저장 (target 제외)
        self.features = [c for c in self.df.columns if c != self.target]

    def split(self, test_size: float = 0.2):
        """데이터셋을 train/test 세트로 분할"""
        if self.features is None:
            raise RuntimeError("Run encode_categoricals() before split().")

        X = self.df[self.features]
        y = self.df[self.target]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )

        print(f"  Train: {len(self.X_train)} rows, Test: {len(self.X_test)} rows")
        return self.X_train, self.X_test, self.y_train, self.y_test

    def process(self, test_size: float = 0.2):
        """세 단계를 한 번에 실행 (App에서 호출하는 핵심 메서드)."""
        self.add_time_features() 
        self.encode_categoricals() 
        return self.split(test_size)