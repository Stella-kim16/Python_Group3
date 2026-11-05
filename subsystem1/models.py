"""Domain models for Subsystem 1: Real-Time Energy Monitoring & Data Management."""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict


@dataclass
class Building:
    """Represents a monitored building (Home, Office, or University)."""

    building_id: str
    building_type: str
    area_sqft: float
    max_occupancy: int
    baseline_kwh: float

    def get_building_info(self) -> Dict[str, Any]:
        """Return building metadata as a dictionary."""
        return asdict(self)


@dataclass
class EnergyLog:
    """Represents a single validated and feature-engineered energy telemetry record."""

    timestamp: datetime
    building_id: str
    building_type: str
    area_sqft: float
    temperature: float
    humidity: float
    occupancy: int
    hvac_usage: str
    lighting_usage: str
    renewable_energy_pct: float
    day_of_week: str
    holiday: str
    energy_consumption_kwh: float
    carbon_footprint_kg: float = 0.0
    energy_cost_eur: float = 0.0
    is_anomaly: bool = False

    def validate_data_types(self) -> bool:
        """Validate domain constraints and data types for this telemetry record."""
        if not isinstance(self.timestamp, datetime):
            return False
        if not isinstance(self.occupancy, int) or self.occupancy < 0:
            return False
        if self.area_sqft <= 0:
            return False
        if not (0.0 <= self.humidity <= 100.0):
            return False
        if not (0.0 <= self.renewable_energy_pct <= 100.0):
            return False
        if self.hvac_usage not in ("On", "Off") or self.lighting_usage not in ("On", "Off"):
            return False
        if self.holiday not in ("Yes", "No"):
            return False
        if self.energy_consumption_kwh < 0:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the energy log record into a JSON-compatible dictionary."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return data
