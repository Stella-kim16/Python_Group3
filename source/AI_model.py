from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split



# 계산기의 역할 -> 이 조건이면 몇 kwh 를 소비한다 라고 계산하는 파트
# Calculator : calculate "how many kWh" for given conditions 


TARGET = "EnergyConsumption"

# 모델이 사용하는 컬럼 (여기에 있는 것만 학습에 들어감)
# Choose specific columns to train on, not all columns in the CSV.
FEATURE_COLS = [
    "HVACUsage", "LightingUsage", "Occupancy",        
    "Temperature", "Humidity",                        
    "SquareFootage", "max_occupancy",                 
    "Hour", "Month", "Holiday",                       
    "RenewableEnergy",                            
]
# one-hot 으로 생긴 컬럼은 이름이 데이터마다 달라서 앞부분(prefix)으로 고름
# To prevent errors when predicting, the model will only use the columns it was trained on.
FEATURE_PREFIXES = ["Day_", "Bld_"]


class AImodel:
    """Select features, split, train a Random Forest and predict."""

    def __init__(self, n_estimators: int = 200, max_depth: int | None = None,
                 random_state: int = 42, test_size: float = 0.2):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,   # 트리 개수
            max_depth=max_depth,         # 트리 깊이 제한 (None = 제한 없음)
            random_state=random_state,   # 실행할 때마다 같은 결과
            n_jobs=-1,                   # CPU 코어 전부 사용
        )
        self.test_size = test_size
        self.feature_names = None        # 실제로 학습에 쓴 컬럼 (예측 때 순서 맞추기용)
        self.is_trained = False
        # train() 후에 채워짐 — 평가 / 24시간 예측에서 사용
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None

    # -----------------------------------------------------------
    # 필요한 컬럼만 고르기
    # -----------------------------------------------------------
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in FEATURE_COLS if c in df.columns]
        cols += [c for c in df.columns if any(c.startswith(p) for p in FEATURE_PREFIXES)]

        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            print(f"  Warning: feature columns not found: {missing}")

        X = df[cols]
        non_numeric = X.select_dtypes(exclude="number").columns
        if len(non_numeric) > 0:
            raise RuntimeError(f"Non-numeric features: {list(non_numeric)}. "
                               "Run Dataprocessor.process() first.")
        return X

    # -----------------------------------------------------------
    # 학습: clean_df 만 넘기면 컬럼 선택 → 나누기 → 학습까지
    # -----------------------------------------------------------
    def train(self, clean_df: pd.DataFrame) -> "AImodel":
        # clean_df: load_clean_data() 가 돌려준 전처리 완료 데이터
        X = self.select_features(clean_df)
        y = clean_df[TARGET]

        # 시계열이라 shuffle=False (과거로 학습, 미래로 테스트)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, shuffle=False)

        self.feature_names = list(X.columns)
        self.model.fit(self.X_train, self.y_train)
        self.is_trained = True
        print(f"  Train: {len(self.X_train)} rows, Test: {len(self.X_test)} rows, "
              f"{len(self.feature_names)} features")
        return self

    # -----------------------------------------------------------
    # 예측: 어떤 DataFrame 이 와도 학습 때 컬럼만, 같은 순서로
    # -----------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> pd.Series:
        if not self.is_trained:
            raise RuntimeError("Call train() before predict().")
        X = X.reindex(columns=self.feature_names, fill_value=0)
        return pd.Series(self.model.predict(X), index=X.index, name="predicted_kWh")

    def history_X(self) -> pd.DataFrame:
        """All feature rows (train + test) — input for forecasting and savings."""
        return pd.concat([self.X_train, self.X_test])

    def feature_importance(self) -> pd.Series:
        return (pd.Series(self.model.feature_importances_, index=self.feature_names)
                  .sort_values(ascending=False))

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> "AImodel":
        return joblib.load(path)

