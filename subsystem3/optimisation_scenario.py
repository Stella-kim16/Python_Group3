from dataclasses import dataclass, field

@dataclass
class OptimisationScenario:
    """Represents one energy-saving optimization scenario."""
    scenario_id: int
    parameters: dict = field(default_factory=dict)  # e.g., HVAC On/Off, Lighting state, setpoints
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "parameters": self.parameters,
            "description": self.description,
        }