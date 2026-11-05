import pandas as pd

from subsystem2.Data_processor import CSV_PATH, load_clean_data
from subsystem2.AI_model import AImodel
from subsystem2.Demandforecast import DemandForecaster
from subsystem2.AI_sustainablity import SustainabilityScorer
from subsystem2.AI_savingadviser import SavingsAdvisor,HVACOffWhenEmpty
from subsystem2.Prediction_result import PredictionResult
from subsystem2.Model_evaluation import ModelEvaluator 



# 1) 읽기 + 전처리
raw_df = pd.read_csv(CSV_PATH)                 # 원본 (DemandForecaster 가 시각 / 빌딩 / 단가를 가져감)
clean_df = load_clean_data()                   # 전처리 완료 데이터

# 2) 계산기 학습
model = AImodel().train(clean_df)

# 3) 다음 24시간 예측 — 모델과 원본만 넘기면 됨
forecaster = DemandForecaster(model, raw_df)
forecasts = forecaster.forecast_all()          # {빌딩 이름: ForecastResult}

# 4) 빌딩 하나 골라서 보기
result = forecasts["BLD_OFF_01"]
print(f"\n{result.building_id} 다음 24시간: {result.total_kwh} kWh, {result.total_cost_eur} €")
print(result.hourly.round(2).head(6))

print("================================")
print("\n[Sustainability Score]")
scorer = SustainabilityScorer(model, raw_df)
scores = scorer.score_all()                    # {빌딩 이름: ScoreResult}

s = scores["BLD_OFF_01"]
print(f"\n{s.building_id} ({s.building_type}): {s.total_score}/100, "
      f"Current Efficiency {s.efficiency_score}/100")
print(s.details)


print("\n[Savings]")
advisor = SavingsAdvisor(model, raw_df)
tips = advisor.recommend_all()                 # {빌딩 이름: [Recommendation, ...]}
 
for tip in tips["BLD_OFF_01"]:
    print(f"\n  {tip.title}")
    print(f"  {tip.message}")
    print(f"  wasted hours: {tip.wasted_hours}, saved: {tip.saved_kwh} kWh / {tip.saved_eur} € "
          f"in data period, {tip.saved_eur_per_month} € per month ({tip.saved_pct} %)")
 
# HVAC 시나리오 하나만 직접 계산하고 싶을 때
hvac = advisor.evaluate("BLD_UNI_02", HVACOffWhenEmpty())
print(f"\n  {hvac.building_id}: {hvac.message}")



print("\n[Model Evaluation]")
test_time = pd.to_datetime(raw_df.loc[model.X_test.index, "Timestamp"])   # ← 여기
rf_result = PredictionResult.from_series(
    "Random Forest", model.predict(model.X_test), model.y_test, timestamps=test_time)

evaluator = ModelEvaluator()
print(evaluator.evaluate(rf_result))