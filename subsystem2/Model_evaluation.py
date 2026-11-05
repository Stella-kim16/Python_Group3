import numpy as np

from subsystem2.Prediction_result import PredictionResult
from subsystem2.exceptions_subsystem2 import ModelEvaluationError

class ModelEvaluator:
    """Evaluate a PredictionResult with MAE and MSE with robust custom exception handling."""

    def __init__(self):
        try:
            self.mae = None
            self.mse = None
            self._y_pred = None
            self._y_true = None
        except Exception as e:
            raise ModelEvaluationError(f"Failed to initialize ModelEvaluator: {e}") from e

    def evaluate(self, prediction: PredictionResult) -> dict:
        try:
            pairs = [(p, a) for p, a in zip(prediction.getPredictions(), prediction.getActualValues())
                     if a is not None]
            if not pairs:
                raise ValueError("No actual values to compare. Use addActualValues() first.")

            self._y_pred = np.array([p for p, _ in pairs])   
            self._y_true = np.array([a for _, a in pairs])   

            self.mae = self.calculateMAE()
            self.mse = self.calculateMSE()
            return self.getMetrics()
        except Exception as e:
            if isinstance(e, ModelEvaluationError):
                raise
            raise ModelEvaluationError(f"Error evaluating prediction results: {e}") from e

    def calculateMAE(self) -> float:
        try:
            if self._y_true is None or self._y_pred is None:
                raise RuntimeError("Evaluation data not initialized. Run evaluate() first.")
            return float(np.mean(np.abs(self._y_true - self._y_pred)))
        except Exception as e:
            if isinstance(e, ModelEvaluationError):
                raise
            raise ModelEvaluationError(f"Failed to calculate Mean Absolute Error (MAE): {e}") from e

    def calculateMSE(self) -> float:
        try:
            if self._y_true is None or self._y_pred is None:
                raise RuntimeError("Evaluation data not initialized. Run evaluate() first.")
            return float(np.mean((self._y_true - self._y_pred) ** 2))
        except Exception as e:
            if isinstance(e, ModelEvaluationError):
                raise
            raise ModelEvaluationError(f"Failed to calculate Mean Squared Error (MSE): {e}") from e

    def getMetrics(self) -> dict:
        try:
            if self.mae is None or self.mse is None:
                raise RuntimeError("Metrics have not been computed yet. Run evaluate() first.")
            return {"MAE": round(self.mae, 3), "MSE": round(self.mse, 3)}
        except Exception as e:
            if isinstance(e, ModelEvaluationError):
                raise
            raise ModelEvaluationError(f"Failed to retrieve model evaluation metrics: {e}") from e