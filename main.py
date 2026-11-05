import pandas as pd

from source.Data_processor import CSV_PATH, load_clean_data
from source.AI_model import AImodel
from source.Demandforecast import DemandForecaster
from source.AI_sustainablity import SustainabilityScorer
from source.AI_savingadviser import SavingsAdvisor,HVACOffWhenEmpty
from source.Prediction_result import PredictionResult
from source.Model_evaluation import ModelEvaluator 


raw_df = pd.read_csv(CSV_PATH)               
clean_df = load_clean_data()                   


model = AImodel().train(clean_df)


forecaster = DemandForecaster(model, raw_df)
forecasts = forecaster.forecast_all()         

result = forecasts["BLD_OFF_01"]
print(f"\n{result.building_id} next 24 hours: {result.total_kwh} kWh, {result.total_cost_eur} €")
print(result.hourly.round(2).head(6))

print("================================")
print("\n[Sustainability Score]")
scorer = SustainabilityScorer(model, raw_df)
scores = scorer.score_all()                    

s = scores["BLD_OFF_01"]
print(f"\n{s.building_id} ({s.building_type}): {s.total_score}/100, "
      f"Current Efficiency {s.efficiency_score}/100")
print(s.details)


print("\n[Savings]")
advisor = SavingsAdvisor(model, raw_df)
tips = advisor.recommend_all()                
for tip in tips["BLD_OFF_01"]:
    print(f"\n  {tip.title}")
    print(f"  {tip.message}")
    print(f"  wasted hours: {tip.wasted_hours}, saved: {tip.saved_kwh} kWh / {tip.saved_eur} € "
          f"in data period, {tip.saved_eur_per_month} € per month ({tip.saved_pct} %)")
 

hvac = advisor.evaluate("BLD_UNI_02", HVACOffWhenEmpty())
print(f"\n  {hvac.building_id}: {hvac.message}")



print("\n[Model Evaluation]")
test_time = pd.to_datetime(raw_df.loc[model.X_test.index, "Timestamp"])   
rf_result = PredictionResult.from_series(
    "Random Forest", model.predict(model.X_test), model.y_test, timestamps=test_time)

evaluator = ModelEvaluator()
print(evaluator.evaluate(rf_result))
