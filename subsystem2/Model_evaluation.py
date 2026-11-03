import pandas as pd
from typing import Dict, Any
from .Prediction_result import PredictionResult  # Correct relative import from sibling file


class ModelEvaluator:
    """Evaluates and compares trained machine learning models based on prediction error metrics."""

    def compare(self, results: list[PredictionResult]) -> pd.DataFrame:
        """Compares multiple prediction results and returns a summary performance table."""
        metrics_data = []
        
        for res in results:
            # Calculate simple error metrics (e.g., Mean Absolute Error - MAE)
            actuals = pd.Series(res.actuals)
            preds = pd.Series(res.predictions)
            mae = (actuals - preds).abs().mean()
            
            metrics_data.append({
                "model": res.model_name,
                "MAE": round(mae, 4)
            })
            
        # Convert to DataFrame and sort by MAE (lowest error first)
        table = pd.DataFrame(metrics_data)
        table = table.sort_values("MAE").reset_index(drop=True)
        return table

    def get_best_model_name(self, table: pd.DataFrame, metric: str = "MAE") -> str:
        """Extracts the name of the top-performing model from the evaluation comparison table."""
        if table.empty:
            raise ValueError("The evaluation table is empty.")
        return str(table.sort_values(metric).iloc[0]["model"])