import numpy as np

from source.Prediction_result import PredictionResult

# 모델이 얼마나 정확한지 평가하는 파트
#   MAE : 평균적으로 몇 kWh 틀리는가
#   MSE : 오차 제곱의 평균 (큰 오차에 더 큰 벌점)


class ModelEvaluator:
    """Evaluate a PredictionResult with MAE and MSE."""

    def __init__(self):
        self.mae = None
        self.mse = None

    def evaluate(self, prediction: PredictionResult) -> dict:
        # 실제값이 있는 것만 비교 (미래 예측은 실측값이 들어온 시간만)
        pairs = [(p, a) for p, a in zip(prediction.getPredictions(), prediction.getActualValues())
                 if a is not None]
        if not pairs:
            raise ValueError("No actual values to compare. Use addActualValues() first.")

        self._y_pred = np.array([p for p, _ in pairs])   # 예측값
        self._y_true = np.array([a for _, a in pairs])   # 실제값

        self.mae = self.calculateMAE()
        self.mse = self.calculateMSE()
        return self.getMetrics()

    def calculateMAE(self) -> float:
        return float(np.mean(np.abs(self._y_true - self._y_pred)))

    def calculateMSE(self) -> float:
        return float(np.mean((self._y_true - self._y_pred) ** 2))

    def getMetrics(self) -> dict:
        return {"MAE": round(self.mae, 3), "MSE": round(self.mse, 3)}