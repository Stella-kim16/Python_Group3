import pandas as pd
from Data_processor import DataProcessor
from Energy_predictor import ModelTrainer          # 파일 이름에 맞게
from Prediction_result import PredictionResult
from Model_evaluation import ModelEvaluator
from AI_prediction import AIPrediction

clean_df = pd.read_csv("cleaned_energy_data.csv") #SOO 한테 받아야함                   # 팀원 A의 데이터

prep = DataProcessor(clean_df)                                          # 1
X_train, X_test, y_train, y_test = prep.process()

trainer = ModelTrainer()                                                # 2
trained_models = trainer.train(X_train, y_train)

results = [PredictionResult(m.predict(X_test).tolist(), name, y_test.tolist())   # 3
           for name, m in trained_models.items()]

evaluator = ModelEvaluator()                                            # 4
table = evaluator.compare(results)
best_name = table.iloc[0]["model"]

ai = AIPrediction(prep, trained_models, best_name)                      # 5
# → ai.predict(scenarios)를 초록 파트가 사용