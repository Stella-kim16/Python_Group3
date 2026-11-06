import pandas as pd
from typing import List, Dict, Any
from subsystem3.models import OptimisationScenario, OptimisationResult

class OptimisationEngine:
    """Generates energy-saving scenarios, evaluates them using the AI predictor, and selects the best option."""
    
    def __init__(self, ai_predictor=None, metrics_calculator=None):
        self.scenarios: List[OptimisationScenario] = []
        self.ai_predictor = ai_predictor              # Stella's AIPrediction object
        self.metrics_calculator = metrics_calculator  # Ecesu's MetricsCalculator object

    def generate_scenarios(self) -> List[OptimisationScenario]:
        """Generates a list of predefined or dynamic energy-saving scenarios to test."""
        self.scenarios = [
            OptimisationScenario(
                scenario_id=1,
                parameters={"HVACUsage": "On", "LightingUsage": "On", "Temperature": 22.0, "Humidity": 45.0, "Occupancy": 20, "SquareFootage": 1420.0, "RenewableEnergy": 20.0},
                description="Baseline Standard Operation"
            ),
            OptimisationScenario(
                scenario_id=2,
                parameters={"HVACUsage": "On", "LightingUsage": "Eco", "Temperature": 24.0, "Humidity": 45.0, "Occupancy": 20, "SquareFootage": 1420.0, "RenewableEnergy": 30.0},
                description="Eco-Friendly Mode (Higher HVAC Setpoint)"
            ),
            OptimisationScenario(
                scenario_id=3,
                parameters={"HVACUsage": "Off", "LightingUsage": "Off", "Temperature": 21.0, "Humidity": 40.0, "Occupancy": 0, "SquareFootage": 1420.0, "RenewableEnergy": 50.0},
                description="Weekend Load Shifting / Unoccupied Mode"
            )
        ]
        return self.scenarios

    def evaluate_scenario(self, scenario: OptimisationScenario, timestamp="2026-06-01 12:00:00", holiday="No") -> OptimisationResult:
        """Evaluates a single scenario by predicting energy consumption and computing cost, carbon, and score."""
        if self.ai_predictor is not None:
            # Convert scenario parameters dict into a single-row DataFrame for Stella's AI predictor
            scenario_df = pd.DataFrame([scenario.parameters])
            pred_df = self.ai_predictor.predict(scenario_df)
            energy_kwh = float(pred_df["PredictedEnergy"].iloc[0])
        else:
            # Fallback dummy value if AI predictor is not yet attached
            energy_kwh = 120.0

        # Compute cost and carbon using Ecesu's metrics calculator (or fallback logic)
        if self.metrics_calculator is not None:
            cost_eur = self.metrics_calculator.calculate_energy_cost(energy_kwh, timestamp, holiday=holiday)
            carbon_kg = self.metrics_calculator.calculate_carbon_emission(energy_kwh, scenario.parameters.get("RenewableEnergy", 20.0))
        else:
            cost_eur = round(energy_kwh * 0.25, 2)
            carbon_kg = round(energy_kwh * 0.42 * 0.8, 2)

        # Compute a composite score (0 to 100) where lower energy yields a higher efficiency score
        score = round(max(0.0, min(100.0, 100.0 * (1.0 - (energy_kwh / 200.0)))), 1)

        return OptimisationResult(
            energy_kwh=energy_kwh,
            carbon_kg=carbon_kg,
            cost_eur=cost_eur,
            score=score,
            recommended=False
        )

    def select_best_scenario(self, results: List[OptimisationResult]) -> OptimisationResult:
        """Selects the scenario with the highest optimisation score and marks it as recommended."""
        if not results:
            raise ValueError("No results provided for selection.")
        best = max(results, key=lambda r: r.score)
        best.recommended = True
        return best