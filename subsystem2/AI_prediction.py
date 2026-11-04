from typing import Dict                    # 타입 힌트
import numpy as np                         # 예측값 배열 처리
import pandas as pd                        # DataFrame

# subsystem2/AI_prediction.py (Top import section)
from .Data_processor import Dataprocessor as DataProcessor
from .Model_evaluation import ModelEvaluator
from .Prediction_result import PredictionResult

class AIPrediction:

    # ① 준비물 받기
    def __init__(self, processor: DataProcessor,
                 trained_models: Dict[str, object], best_model_name: str):
        self.processor = processor             # 전처리 규칙을 아는 객체
        self.trained_models = trained_models   # 학습된 모델 두 개
        self.set_best_model(best_model_name)   # 대표로 쓸 모델 이름

    # ② 대표 모델 정하기 / 바꾸기
    def set_best_model(self, name: str) -> None:
        if name not in self.trained_models:
            raise KeyError(f"Unknown model '{name}'")
        self.best_model_name = name

    # ③ 예측하기 (핵심)
    def predict(self, scenarios: pd.DataFrame, all_models: bool = False) -> pd.DataFrame:
        X = self.processor.transform_new(scenarios)        # (a) 학습 때와 같은 형태로 변환
        result = scenarios.copy()                          # (b) 원본 표를 복사

        names = self.trained_models if all_models else [self.best_model_name]
        for name in names:                                 # (c) 모델별로 예측
            preds = np.asarray(self.trained_models[name].predict(X))
            result[f"Pred_{name}"] = preds.round(2)

        result["PredictedEnergy"] = result[f"Pred_{self.best_model_name}"]   # (d) 대표값
        result["ModelUsed"] = self.best_model_name
        return result

    def update_with_actuals(self, predicted: pd.DataFrame, actual_values,
                            evaluator: ModelEvaluator) -> str:
        results = PredictionResult.from_table(              # 결과 만들기 → PredictionResult
            predicted, actual_values, list(self.trained_models))
        table = evaluator.compare(results)                   # 평가 → ModelEvaluator
        new_best = table.iloc[0]["model"]                    # MAE가 가장 낮은 모델

        if new_best != self.best_model_name:
            print(f"  Best model changed: {self.best_model_name} -> {new_best}")
        self.set_best_model(new_best)                        # 교체만 여기서
        return new_best



    # ④ 조건 하나만 예측하기 (편의용)
    def predict_one(self, all_models: bool = False, **conditions) -> pd.DataFrame:
        return self.predict(pd.DataFrame([conditions]), all_models=all_models)

 # 계산기의 역할 - 이 조건이면 몇 kwh 를 소비한다 라고 계산하는 파트 
 # 디디가 이제 여러가지 시나리오를 조작해서 어떤게 제일 좋을지 소비자에게 전달하는것 
 