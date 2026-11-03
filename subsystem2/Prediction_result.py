from typing import Dict, List                      # 타입 힌트
import pandas as pd                                # DataFrame, Series

class PredictionResult:
    """Stores predictions, model names, and actual test values for evaluation."""
    def __init__(self, predictions: List[float], model_name: str, actuals: List[float]):
        self.predictions = predictions
        self.model_name = model_name
        self.actuals = actuals


def predict_all(trained_models: Dict[str, object],
                X_test: pd.DataFrame, y_test: pd.Series) -> List[PredictionResult]:
    """Runs predictions across all trained models and returns a list of results."""
    results = []
    for name, model in trained_models.items():     # trainer가 돌려준 dict
        preds = model.predict(X_test).tolist()     # 예측 (numpy 배열 → list)
        results.append(PredictionResult(preds, name, y_test.tolist()))
    return results # 평가 모델에 들어갈 변수