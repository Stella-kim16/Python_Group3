from typing import List
from subsystem3.optimisation_scenario import OptimisationScenario
from subsystem3.optimisation_result import OptimisationResult

class OptimisationEngine:
    """Coordinates scenario generation, evaluation, and optimal selection."""
    
    def __init__(self, building_id: str):
        self._building_id = building_id
        self._scenarios: List[OptimisationScenario] = []
        self.initialize_engine()

    def initialize_engine(self) -> None:
        """Setup default baseline operational scenarios upon instantiation."""
        self._scenarios = [
            OptimisationScenario(1, {"hvac_offset_c": 0, "lighting_dimming": 0.0}, "Baseline Operation"),
            OptimisationScenario(2, {"hvac_offset_c": 2, "lighting_dimming": 0.2}, "Moderate Energy Saving"),
            OptimisationScenario(3, {"hvac_offset_c": 4, "lighting_dimming": 0.5}, "Aggressive Eco Mode")
        ]

    # Getters (Accessor Methods)
    def get_building_id(self) -> str:
        return self._building_id

    def get_scenarios(self) -> List[OptimisationScenario]:
        return self._scenarios

    # Setters (Mutator Methods)
    def set_building_id(self, building_id: str) -> None:
        self._building_id = building_id

    def set_scenarios(self, scenarios: List[OptimisationScenario]) -> None:
        self._scenarios = scenarios

    def evaluate_scenarios(self) -> List[OptimisationResult]:
        """Evaluates all managed scenarios and returns calculated results."""
        results = []
        for sc in self._scenarios:
            params = sc.get_parameters()
            reduction_factor = 1.0 - (params.get("hvac_offset_c", 0) * 0.05 + params.get("lighting_dimming", 0.0) * 0.1)
            
            energy = 1500.0 * reduction_factor
            carbon = energy * 0.4
            cost = energy * 0.25
            score = 60.0 + (1.0 - reduction_factor) * 40.0
            
            res = OptimisationResult(sc.get_scenario_id(), sc.get_description(), energy, carbon, cost, score)
            results.append(res)
        return results

    def select_best_scenario(self, results: List[OptimisationResult]) -> OptimisationResult:
        """Selects the scenario with the highest sustainability score."""
        best = max(results, key=lambda r: r.get_score())
        best.set_recommended(True)
        return best