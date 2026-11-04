from typing import Dict, Any

class OptimisationScenario:
    """Represents a specific energy-saving operating condition scenario."""
    
    def __init__(self, scenario_id: int, parameters: Dict[str, Any], description: str):
        self.scenario_id = scenario_id
        self.parameters = parameters
        self.description = description

    def to_dict(self) -> Dict[str, Any]:
        """Converts the scenario object into a standard Python dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "parameters": self.parameters,
            "description": self.description
        }


class OptimisationResult:
    """Stores the evaluated metrics of an optimisation scenario."""
    
    def __init__(self, energy_kwh: float, carbon_kg: float, cost_eur: float, score: float, recommended: bool = False):
        self.energy_kwh = energy_kwh
        self.carbon_kg = carbon_kg
        self.cost_eur = cost_eur
        self.score = score
        self.recommended = recommended

    def summary(self) -> Dict[str, Any]:
        """Returns a dictionary summary of all evaluated metrics."""
        return {
            "energy_kwh": self.energy_kwh,
            "carbon_kg": self.carbon_kg,
            "cost_eur": self.cost_eur,
            "score": self.score,
            "recommended": self.recommended
        }