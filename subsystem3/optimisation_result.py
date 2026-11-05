from typing import Dict, Any

class OptimisationResult:
    """Stores the quantitative results of an evaluated optimization scenario."""
    
    def __init__(self, scenario_id: int, description: str, energy_kwh: float, carbon_kg: float, cost_eur: float, score: float):
        self._scenario_id = scenario_id
        self._description = description
        self._energy_kwh = energy_kwh
        self._carbon_kg = carbon_kg
        self._cost_eur = cost_eur
        self._score = score
        self._recommended = False
        self.initialize_metrics()

    def initialize_metrics(self) -> None:
        """Setup and validation logic for evaluation metrics."""
        if self._score < 0.0:
            self._score = 0.0

    # Getters (Accessor Methods)
    def get_scenario_id(self) -> int:
        return self._scenario_id

    def get_description(self) -> str:
        return self._description

    def get_energy_kwh(self) -> float:
        return self._energy_kwh

    def get_carbon_kg(self) -> float:
        return self._carbon_kg

    def get_cost_eur(self) -> float:
        return self._cost_eur

    def get_score(self) -> float:
        return self._score

    def is_recommended(self) -> bool:
        return self._recommended

    # Setters (Mutator Methods)
    def set_description(self, description: str) -> None:
        self._description = description

    def set_energy_kwh(self, energy_kwh: float) -> None:
        self._energy_kwh = energy_kwh

    def set_carbon_kg(self, carbon_kg: float) -> None:
        self._carbon_kg = carbon_kg

    def set_cost_eur(self, cost_eur: float) -> None:
        self._cost_eur = cost_eur

    def set_score(self, score: float) -> None:
        self._score = score

    def set_recommended(self, recommended: bool) -> None:
        self._recommended = recommended

    def summary(self) -> Dict[str, Any]:
        """Returns a structured summary dictionary of the results."""
        return {
            "scenario_id": self._scenario_id,
            "description": self._description,
            "energy_kwh": self._energy_kwh,
            "carbon_kg": self._carbon_kg,
            "cost_eur": self._cost_eur,
            "score": self._score,
            "recommended": self._recommended
        }