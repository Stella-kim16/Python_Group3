from typing import List, Any
from subsystem3.optimisation_scenario import OptimisationScenario
from subsystem3.optimisation_result import OptimisationResult
from subsystem3.exceptions_subsystem3 import ScenarioGenerationError, OptimisationEvaluationError

class OptimisationEngine:
    """
    Core engine responsible for generating operational scenarios, 
    evaluating them against building metrics, and selecting the optimal result.
    """

    def __init__(self, predictor: Any = None, building_id: str = "Building_A") -> None:
        self._predictor = predictor
        self._building_id = building_id
        self._scenarios: List[OptimisationScenario] = []
        self.generate_scenarios()

    # --- Getters and Setters ---
    def get_predictor(self) -> Any:
        return self._predictor

    def set_predictor(self, predictor: Any) -> None:
        self._predictor = predictor

    def get_building_id(self) -> str:
        return self._building_id

    def set_building_id(self, building_id: str) -> None:
        self._building_id = building_id

    def get_scenarios(self) -> List[OptimisationScenario]:
        return self._scenarios

    def set_scenarios(self, scenarios: List[OptimisationScenario]) -> None:
        self._scenarios = scenarios

    # --- Core Engine Logic with Error Handling ---
    def generate_scenarios(self) -> List[OptimisationScenario]:
        """
        Generates baseline and optimised operational scenarios for the building 
        with safety guardrails and custom error handling.
        """
        try:
            # Building standard operational scenarios
            self._scenarios = [
                OptimisationScenario(1, {"hvac_offset_c": 0, "lighting_dimming": 0.0}, "Baseline Operation"),
                OptimisationScenario(2, {"hvac_offset_c": 2, "lighting_dimming": 0.2}, "Moderate Energy Saving"),
                OptimisationScenario(3, {"hvac_offset_c": 4, "lighting_dimming": 0.5}, "Aggressive Eco Mode")
            ]
            
            if not self._scenarios:
                raise ValueError("Generated scenario list resulted in an empty collection.")
                
            return self._scenarios
            
        except Exception as e:
            raise ScenarioGenerationError(
                f"Failed to generate operational scenarios for building '{self._building_id}': {e}"
            ) from e

    def evaluate_scenario(self, scenario: OptimisationScenario) -> OptimisationResult:
        """
        Evaluates a given optimisation scenario, calculating expected energy, 
        carbon emissions, cost, and a composite score safely.
        """
        try:
            if not isinstance(scenario, OptimisationScenario):
                raise TypeError("Provided object is not a valid instance of OptimisationScenario.")

            params = scenario.get_parameters()
            hvac_offset = params.get("hvac_offset_c", 0)
            lighting_dim = params.get("lighting_dimming", 0.0)

            # Calculation logic based on reduction offsets
            reduction_factor = 1.0 - (hvac_offset * 0.05 + lighting_dim * 0.1)
            energy = 1500.0 * reduction_factor

            if energy < 0:
                raise ValueError("Evaluated energy consumption cannot be negative.")

            carbon = energy * 0.4
            cost = energy * 0.25
            score = 60.0 + (1.0 - reduction_factor) * 40.0

            return OptimisationResult(
                scenario_id=scenario.get_scenario_id(),
                description=scenario.get_description(),
                energy_kwh=energy,
                carbon_kg=carbon,
                cost_eur=cost,
                score=score
            ) 

        except Exception as e:
            raise OptimisationEvaluationError(
                f"Error evaluating scenario ID {scenario.get_scenario_id()}: {e}"
            ) from e

    def select_best_scenario(self, results: List[OptimisationResult]) -> OptimisationResult:
        """
        Selects and marks the best performing result based on the calculated score.
        """
        try:
            if not results:
                raise ValueError("Cannot select best scenario from an empty results list.")

            # Find result with the highest score
            best_result = max(results, key=lambda r: r.get_score())
            best_result.set_recommended(True)
            return best_result

        except Exception as e:
            raise OptimisationEvaluationError(f"Failed to select optimal scenario: {e}") from e