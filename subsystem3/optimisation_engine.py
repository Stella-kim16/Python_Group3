import pandas as pd
from subsystem3.optimisation_scenario import OptimisationScenario
from subsystem3.optimisation_result import OptimisationResult

class OptimisationEngine:
    """Generates scenarios, evaluates them using the AI predictor, and selects the best one."""
    
    def __init__(self, predictor, scenarios: list[OptimisationScenario] | None = None):
        self.predictor = predictor  # AI model (e.g. AImodel from Subsystem 2)
        self.scenarios = scenarios or self.generate_default_scenarios()

    def generate_default_scenarios(self) -> list[OptimisationScenario]:
        return [
            OptimisationScenario(1, {"HVACUsage": 1, "LightingUsage": 1}, "Baseline (Current Operation)"),
            OptimisationScenario(2, {"HVACUsage": 0, "LightingUsage": 1}, "Switch off HVAC only"),
            OptimisationScenario(3, {"HVACUsage": 1, "LightingUsage": 0}, "Switch off Lighting only"),
            OptimisationScenario(4, {"HVACUsage": 0, "LightingUsage": 0}, "Aggressive Energy Saving (Both Off)"),
        ]

    def evaluate_scenario(self, base_X: pd.DataFrame, scenario: OptimisationScenario, price_per_kwh: float = 0.15, carbon_factor: float = 0.4) -> OptimisationResult:
        # Apply scenario parameter adjustments
        X_test = base_X.copy()
        for col, val in scenario.parameters.items():
            if col in X_test.columns:
                X_test[col] = val

        # Predict kWh using AI predictor
        pred_kwh_series = self.predictor.predict(X_test)
        total_kwh = float(pred_kwh_series.sum())
        
        total_cost = total_kwh * price_per_kwh
        total_carbon = total_kwh * carbon_factor
        
        # Scoring logic (lower energy/cost/carbon = higher score)
        score = max(0.0, 100.0 - (total_kwh / 10.0))

        return OptimisationResult(
            scenario_id=scenario.scenario_id,
            description=scenario.description,
            energy_kwh=round(total_kwh, 2),
            carbon_kg=round(total_carbon, 2),
            cost_eur=round(total_cost, 2),
            score=round(score, 2),
            recommended=False,
            summary={"parameters": scenario.parameters}
        )

    def select_best_scenario(self, base_X: pd.DataFrame) -> list[OptimisationResult]:
        results = []
        for s in self.scenarios:
            res = self.evaluate_scenario(base_X, s)
            results.append(res)
        
        # Sort by lowest cost / energy consumption
        results.sort(key=lambda r: r.cost_eur)
        if results:
            results[0].recommended = True  # Mark best scenario
        return results