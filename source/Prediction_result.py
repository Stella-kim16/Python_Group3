from typing import Dict, List                    # 타입 힌트
import pandas as pd                              # DataFrame, Series
from Prediction_result import PredictionResult   # 결과를 담는 class


def predict_all(trained_models: Dict[str, object],
                X_test: pd.DataFrame, y_test: pd.Series) -> List[PredictionResult]:
    results = []
    for name, model in trained_models.items():     # trainer가 돌려준 dict
        preds = model.predict(X_test).tolist()     # 예측 (numpy 배열 → list)
        results.append(PredictionResult(preds, name, y_test.tolist()))
    return results # 평가 모델에 들어갈 변수 


