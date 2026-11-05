from typing import Dict, Any

class OptimisationScenario:
    """Represents an individual energy-saving operational scenario."""
    
    def __init__(self, scenario_id: int, parameters: Dict[str, Any], description: str = ""):
        # Private attributes for encapsulation
        self._scenario_id = scenario_id
        self._parameters = parameters
        self._description = description
        self.initialize_scenario()

    def initialize_scenario(self) -> None:
        """Setup or configuration logic for the scenario parameters."""
        if not isinstance(self._parameters, dict):
            self._parameters = {}

    # Getters (Accessor Methods)
    def get_scenario_id(self) -> int:
        return self._scenario_id

    def get_parameters(self) -> Dict[str, Any]:
        return self._parameters

    def get_description(self) -> str:
        return self._description

    # Setters (Mutator Methods)
    def set_scenario_id(self, scenario_id: int) -> None:
        self._scenario_id = scenario_id

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        self._parameters = parameters

    def set_description(self, description: str) -> None:
        self._description = description

    def to_dict(self) -> Dict[str, Any]:
        """Converts the scenario instance into a dictionary format."""
        return {
            "scenario_id": self._scenario_id,
            "parameters": self._parameters,
            "description": self._description,
        }