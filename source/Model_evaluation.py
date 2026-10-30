from Model_evaluation import ModelEvaluator   
import pandas as pd


# (위에서) results = predict_all(trained_models, X_test, y_test)

Evaluator = ModelEvaluator()
table = Evaluator.compare(results) # 예측 CLASS 에서 RETURN 하는 값


def get_best_model_name(self, table: pd.DataFrame, metric: str = "MAE") -> str:
    return table.sort_values(metric).iloc[0]["model"]   