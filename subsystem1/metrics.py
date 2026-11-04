"""Metrics Calculator for energy cost, carbon footprint, anomaly detection, and sustainability scoring."""

from datetime import datetime
from typing import Dict, Union
import numpy as np
import pandas as pd


class MetricsCalculator:
    """Computes financial costs, carbon emissions, anomaly flags, and sustainability scores."""

    def __init__(
        self,
        emission_factor: float = 0.42,  # kg CO2 per kWh of non-renewable grid electricity
        peak_tariff: float = 0.30,      # EUR per kWh during weekday peak hours (08:00 - 20:00)
        off_peak_tariff: float = 0.15,  # EUR per kWh during nights, weekends, and holidays
    ) -> None:
        self.emission_factor = emission_factor
        self.peak_tariff = peak_tariff
        self.off_peak_tariff = off_peak_tariff

    def estimate_kwh(
        self,
        hvac: str,
        lighting: str,
        sqft: float,
        temperature: float = 24.0,
        humidity: float = 45.0,
        occupancy: int = 5,
        max_occupancy: int = 35,
    ) -> float:
        """Estimate hourly energy consumption (kWh) for a single state (used by UML / stream estimator)."""
        occ_rate = min(max(occupancy / max(max_occupancy, 1), 0.0), 1.0)
        base_load = 0.026 * sqft
        thermal_load = 1.65 * (temperature - 20.0) + 0.12 * (humidity - 30.0)
        occupancy_load = 12.0 * occ_rate + 0.18 * occupancy
        hvac_load = (11.5 + 0.55 * (temperature - 20.0)) if hvac == "On" else 0.0
        lighting_load = (4.5 + 0.002 * sqft) if lighting == "On" else 0.0
        return round(max(base_load + thermal_load + occupancy_load + hvac_load + lighting_load, 25.0), 4)

    def get_tariff_rate(self, timestamp: Union[datetime, pd.Timestamp], holiday: str = "No") -> float:
        """Return dynamic tariff rate (EUR/kWh) based on time-of-use and weekday/weekend schedule."""
        ts = pd.to_datetime(timestamp)
        is_weekend = ts.weekday() >= 5
        if is_weekend or holiday == "Yes":
            return self.off_peak_tariff
        if 8 <= ts.hour < 20:
            return self.peak_tariff
        return self.off_peak_tariff

    def calculate_energy_cost(
        self,
        estimated_kwh: float,
        timestamp: Union[datetime, pd.Timestamp],
        holiday: str = "No",
    ) -> float:
        """Calculate total energy cost (EUR) for a given consumption and timestamp."""
        rate = self.get_tariff_rate(timestamp, holiday=holiday)
        return round(float(estimated_kwh) * rate, 4)

    def calculate_carbon_emission(self, estimated_kwh: float, renewable_pct: float) -> float:
        """
        Calculate carbon footprint (kg CO2) factoring in the building's renewable energy percentage.
        """
        non_renewable_share = max(0.0, min(1.0, 1.0 - (float(renewable_pct) / 100.0)))
        carbon_kg = float(estimated_kwh) * non_renewable_share * self.emission_factor
        return round(carbon_kg, 4)

    def detect_anomalies(self, df: pd.DataFrame, z_threshold: float = 2.2) -> pd.Series:
        """
        Flag anomalous telemetry records where:
        1. Consumption exceeds the building's mean baseline by > z_threshold standard deviations, OR
        2. Building is completely unoccupied (Occupancy == 0) while both HVAC and Lighting are On.
        """
        grouped = df.groupby("building_id")["EnergyConsumption"]
        b_mean = grouped.transform("mean")
        b_std = grouped.transform("std").replace(0, 1.0)
        z_scores = (df["EnergyConsumption"] - b_mean) / b_std

        unoccupied_waste = (
            (df["Occupancy"] == 0)
            & (df["HVACUsage"] == "On")
            & (df["LightingUsage"] == "On")
        )
        return (z_scores > z_threshold) | unoccupied_waste

    def enrich_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add cost_per_kwh, energy_cost_eur, carbon_footprint_kg, and is_anomaly columns to DataFrame."""
        out = df.copy()
        hours = pd.to_datetime(out["Timestamp"]).dt.hour
        is_weekend = out["DayOfWeek"].isin(["Saturday", "Sunday"])
        is_holiday = out["Holiday"] == "Yes"

        is_peak = (~is_weekend) & (~is_holiday) & (hours >= 8) & (hours < 20)
        out["cost_per_kwh"] = np.where(is_peak, self.peak_tariff, self.off_peak_tariff)
        out["energy_cost_eur"] = (out["EnergyConsumption"] * out["cost_per_kwh"]).round(4)

        non_ren_share = np.clip(1.0 - (out["RenewableEnergy"] / 100.0), 0.0, 1.0)
        out["carbon_footprint_kg"] = (
            out["EnergyConsumption"] * non_ren_share * self.emission_factor
        ).round(4)

        out["is_anomaly"] = self.detect_anomalies(out)
        return out

    def calculate_sustainability_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute a 0-100 Sustainability Score per building based on:
        - Renewable energy utilization (40 pts)
        - Off-hours equipment discipline when unoccupied (30 pts)
        - Carbon efficiency per sq ft (30 pts)
        """
        scores: Dict[str, float] = {}
        for bid, group in df.groupby("building_id"):
            ren_score = min(40.0, (group["RenewableEnergy"].mean() / 25.0) * 40.0)

            unocc = group[group["Occupancy"] == 0]
            if len(unocc) > 0:
                waste_ratio = ((unocc["HVACUsage"] == "On") | (unocc["LightingUsage"] == "On")).mean()
                discipline_score = 30.0 * (1.0 - waste_ratio)
            else:
                discipline_score = 25.0

            carbon_per_ksqft = (group["carbon_footprint_kg"] / (group["SquareFootage"] / 1000.0)).mean()
            carbon_score = max(0.0, min(30.0, 30.0 * (1.0 - (carbon_per_ksqft - 8.0) / 22.0)))

            scores[str(bid)] = round(ren_score + discipline_score + carbon_score, 1)
        return scores
