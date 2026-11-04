"""Feature Engineering and realistic building/occupancy simulation for Subsystem 1."""

from typing import Dict, List
import numpy as np
import pandas as pd
from subsystem1.models import Building


DEFAULT_BUILDING_CATALOG: List[Building] = [
    Building("BLD_HOM_01", "Home", 1100.0, max_occupancy=4, baseline_kwh=28.0),
    Building("BLD_HOM_02", "Home", 1250.0, max_occupancy=6, baseline_kwh=32.0),
    Building("BLD_OFF_01", "Office", 1420.0, max_occupancy=35, baseline_kwh=48.0),
    Building("BLD_OFF_02", "Office", 1580.0, max_occupancy=50, baseline_kwh=55.0),
    Building("BLD_UNI_01", "University", 1750.0, max_occupancy=90, baseline_kwh=65.0),
    Building("BLD_UNI_02", "University", 1920.0, max_occupancy=120, baseline_kwh=74.0),
]


class FeatureEngineer:
    """Transforms raw energy logs into multi-building, schedule-aware telemetry."""

    def __init__(self, buildings: List[Building] | None = None, random_seed: int = 42) -> None:
        self.buildings = buildings or DEFAULT_BUILDING_CATALOG
        self.building_map: Dict[str, Building] = {b.building_id: b for b in self.buildings}
        self.random_seed = random_seed

    def assign_buildings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Option A: Map existing 1,000 rows to specific buildings based on original
        SquareFootage buckets and snap SquareFootage to each building's fixed area.
        """
        out = df.copy()
        bins = [0.0, 1165.0, 1330.0, 1500.0, 1665.0, 1830.0, float("inf")]
        labels = [b.building_id for b in self.buildings]

        out["building_id"] = pd.cut(out["SquareFootage"], bins=bins, labels=labels, right=False).astype(str)
        out["building_type"] = out["building_id"].map(lambda bid: self.building_map[bid].building_type)
        out["max_occupancy"] = out["building_id"].map(lambda bid: self.building_map[bid].max_occupancy).astype(int)
        out["SquareFootage"] = out["building_id"].map(lambda bid: self.building_map[bid].area_sqft)
        return out

    def fix_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep the existing Holiday column from the initial CSV dataset and ensure
        all Sundays are also marked as Holiday = 'Yes'.
        """
        out = df.copy()
        out["Holiday"] = out["Holiday"].astype(str).str.strip().str.title()
        out.loc[out["DayOfWeek"] == "Sunday", "Holiday"] = "Yes"
        return out

    def simulate_realistic_occupancy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale integer headcount realistically by building_type, hour from Timestamp, and weekend/holiday:
        - Office: very low on weekends/nights, high on weekday work hours (08:00-17:00).
        - Home: very high in evenings (18:00-23:00), nights, and weekends; low on weekday daytime.
        - University: high on weekday daytime (08:00-17:00), moderate in evenings, low on weekends.
        """
        out = df.copy()
        rng = np.random.default_rng(self.random_seed)
        hours = pd.to_datetime(out["Timestamp"]).dt.hour.to_numpy()

        # Use the original 0-9 Occupancy column as a subtle within-band variation signal (0.0 to 1.0)
        orig_norm = np.clip(out["Occupancy"].to_numpy(dtype=float) / 9.0, 0.0, 1.0)
        jitter = rng.uniform(-0.08, 0.08, size=len(out))
        signal = np.clip(0.7 * orig_norm + 0.3 * rng.random(len(out)) + jitter, 0.0, 1.0)

        new_occupancy = np.zeros(len(out), dtype=int)

        for idx, row in out.iterrows():
            btype = row["building_type"]
            max_occ = int(row["max_occupancy"])
            hour = int(hours[idx])
            is_weekend = row["DayOfWeek"] in ("Saturday", "Sunday")
            is_holiday = row["Holiday"] == "Yes"
            off_day = bool(is_weekend or is_holiday)
            s = float(signal[idx])

            if btype == "Office":
                if off_day:
                    # Very low on weekends/holidays at office buildings (0 to ~5% capacity)
                    frac = 0.0 if s < 0.65 else 0.05 * s
                else:
                    if 9 <= hour <= 16:
                        frac = 0.65 + 0.30 * s  # Peak office hours: 65% - 95%
                    elif hour in (8, 17):
                        frac = 0.40 + 0.30 * s  # Arrival / departure: 40% - 70%
                    elif hour in (7, 18, 19):
                        frac = 0.08 + 0.17 * s  # Early / late workers: 8% - 25%
                    else:
                        frac = 0.0 if s < 0.75 else 0.03  # Night security: 0 - 1 person

            elif btype == "Home":
                if off_day:
                    if 18 <= hour <= 23 or 0 <= hour <= 8:
                        frac = 0.80 + 0.20 * s  # Weekend evenings/nights: 80% - 100%
                    else:
                        frac = 0.50 + 0.40 * s  # Weekend daytime: 50% - 90%
                else:
                    if 18 <= hour <= 23:
                        # Very high on the evenings at homes (75% - 100%)
                        frac = 0.75 + 0.25 * s
                    elif 0 <= hour <= 6:
                        frac = 0.85 + 0.15 * s  # Sleeping hours: 85% - 100%
                    elif hour in (7, 17):
                        frac = 0.45 + 0.35 * s  # Morning/evening transition: 45% - 80%
                    else:
                        frac = 0.05 + 0.25 * s  # Weekday daytime: 5% - 30%

            else:  # University
                if off_day:
                    if 9 <= hour <= 18:
                        frac = 0.04 + 0.11 * s  # Weekend library/labs: 4% - 15%
                    else:
                        frac = 0.0 if s < 0.80 else 0.02
                else:
                    if 9 <= hour <= 16:
                        frac = 0.60 + 0.35 * s  # Main lecture/lab hours: 60% - 95%
                    elif hour in (8, 17):
                        frac = 0.35 + 0.30 * s  # 35% - 65%
                    elif 18 <= hour <= 21:
                        frac = 0.18 + 0.25 * s  # Evening classes / study halls: 18% - 43%
                    else:
                        frac = 0.0 if s < 0.75 else 0.02

            new_occupancy[idx] = int(np.clip(round(frac * max_occ), 0, max_occ))

        out["Occupancy"] = new_occupancy.astype(int)
        return out

    def align_hvac_and_lighting(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Align HVACUsage and LightingUsage with occupancy, building schedule, and weather,
        leaving a small fraction (~5%) of unoccupied active equipment as realistic waste anomalies.
        """
        out = df.copy()
        rng = np.random.default_rng(self.random_seed + 7)
        hours = pd.to_datetime(out["Timestamp"]).dt.hour.to_numpy()

        hvac_list: List[str] = []
        light_list: List[str] = []

        for idx, row in out.iterrows():
            occ = int(row["Occupancy"])
            max_occ = max(int(row["max_occupancy"]), 1)
            occ_rate = occ / max_occ
            temp = float(row["Temperature"])
            hum = float(row["Humidity"])
            hour = int(hours[idx])
            btype = row["building_type"]

            if occ == 0:
                light = "On" if rng.random() < 0.06 else "Off"
                hvac = "On" if (temp > 28.5 or rng.random() < 0.06) else "Off"
            else:
                if btype == "Home":
                    if 17 <= hour <= 23 or hour in (6, 7):
                        light = "On"
                    elif 0 <= hour <= 5:
                        light = "Off"
                    else:
                        light = "On" if rng.random() < 0.25 else "Off"
                else:
                    light = "On" if (occ_rate >= 0.10 or 17 <= hour <= 21) else ("On" if rng.random() < 0.3 else "Off")

                if temp >= 24.0 or hum >= 52.0 or occ_rate >= 0.45:
                    hvac = "On"
                else:
                    hvac = "On" if rng.random() < 0.30 else "Off"

            hvac_list.append(hvac)
            light_list.append(light)

        out["HVACUsage"] = hvac_list
        out["LightingUsage"] = light_list
        return out

    def recalibrate_energy_consumption(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recalculate EnergyConsumption (kWh) based on physical building characteristics,
        weather, occupancy, and appliance states so ML models and optimisation scenarios
        capture realistic causal relationships.
        """
        out = df.copy()
        rng = np.random.default_rng(self.random_seed + 99)

        sqft = out["SquareFootage"].to_numpy(dtype=float)
        temp = out["Temperature"].to_numpy(dtype=float)
        hum = out["Humidity"].to_numpy(dtype=float)
        occ = out["Occupancy"].to_numpy(dtype=float)
        max_occ = np.maximum(out["max_occupancy"].to_numpy(dtype=float), 1.0)
        occ_rate = occ / max_occ
        hvac_on = (out["HVACUsage"] == "On").to_numpy(dtype=float)
        light_on = (out["LightingUsage"] == "On").to_numpy(dtype=float)

        base_load = 0.026 * sqft
        thermal_load = 1.65 * (temp - 20.0) + 0.12 * (hum - 30.0)
        occupancy_load = 12.0 * occ_rate + 0.18 * occ
        hvac_load = hvac_on * (11.5 + 0.55 * (temp - 20.0))
        lighting_load = light_on * (4.5 + 0.002 * sqft)
        noise = rng.normal(0.0, 1.25, size=len(out))

        energy_kwh = base_load + thermal_load + occupancy_load + hvac_load + lighting_load + noise
        out["EnergyConsumption"] = np.clip(energy_kwh, 25.0, None).round(4)

        ordered_cols = [
            "Timestamp",
            "building_id",
            "building_type",
            "SquareFootage",
            "max_occupancy",
            "Temperature",
            "Humidity",
            "Occupancy",
            "HVACUsage",
            "LightingUsage",
            "RenewableEnergy",
            "DayOfWeek",
            "Holiday",
            "EnergyConsumption",
        ]
        return out[ordered_cols]

    def run_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute all feature engineering and data manipulation steps in sequence."""
        step1 = self.assign_buildings(df)
        step2 = self.fix_calendar_features(step1)
        step3 = self.simulate_realistic_occupancy(step2)
        step4 = self.align_hvac_and_lighting(step3)
        step5 = self.recalibrate_energy_consumption(step4)
        return step5
