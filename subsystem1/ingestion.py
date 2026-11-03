"""Data Ingestion Engine for batch CSV files and real-time sensor streams."""

from pathlib import Path
from typing import Any, Dict, Union
import pandas as pd


REQUIRED_RAW_COLUMNS = [
    "Timestamp",
    "Temperature",
    "Humidity",
    "SquareFootage",
    "Occupancy",
    "HVACUsage",
    "LightingUsage",
    "RenewableEnergy",
    "DayOfWeek",
    "Holiday",
    "EnergyConsumption",
]


class DataIngestionEngine:
    """Ingests, validates, deduplicates, and cleans raw energy telemetry data."""

    def __init__(self, raw_file_path: Union[str, Path, None] = None) -> None:
        self.raw_file_path = Path(raw_file_path) if raw_file_path else None

    def import_csv_batch(self, file_path: Union[str, Path, None] = None) -> pd.DataFrame:
        """Load a batch CSV file, validate schema, deduplicate, and clean invalid rows."""
        target_path = Path(file_path) if file_path else self.raw_file_path
        if target_path is None or not target_path.exists():
            raise FileNotFoundError(f"CSV file not found: {target_path}")

        raw_df = pd.read_csv(target_path)
        missing_cols = [col for col in REQUIRED_RAW_COLUMNS if col not in raw_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")

        cleaned_df = self.clean_missing_data(raw_df)
        deduped_df = self.deduplicate_timestamps(cleaned_df)
        return deduped_df

    def ingest_sensor_stream(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """Validate and convert a single real-time JSON/dict sensor payload into a 1-row DataFrame."""
        missing_cols = [col for col in REQUIRED_RAW_COLUMNS if col not in payload]
        if missing_cols:
            raise ValueError(f"Sensor payload missing required keys: {missing_cols}")

        df = pd.DataFrame([payload])
        cleaned_df = self.clean_missing_data(df)
        if cleaned_df.empty:
            raise ValueError("Sensor payload failed data validation checks.")
        return cleaned_df

    def clean_missing_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Filter or impute missing, negative, or out-of-range telemetry values."""
        df = raw_data.copy()

        # Parse Timestamp and drop rows with invalid timestamps
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        df = df.dropna(subset=["Timestamp"])

        # Coerce numeric columns
        numeric_cols = [
            "Temperature",
            "Humidity",
            "SquareFootage",
            "Occupancy",
            "RenewableEnergy",
            "EnergyConsumption",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Interpolate or fill minor missing numeric sensor readings if any exist
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear").bfill().ffill()

        # Filter out physically invalid / negative values
        valid_mask = (
            (df["SquareFootage"] > 0)
            & (df["Occupancy"] >= 0)
            & (df["Humidity"] >= 0.0)
            & (df["Humidity"] <= 100.0)
            & (df["RenewableEnergy"] >= 0.0)
            & (df["RenewableEnergy"] <= 100.0)
            & (df["EnergyConsumption"] >= 0.0)
        )
        df = df.loc[valid_mask].copy()

        # Normalize categorical usage strings
        df["HVACUsage"] = df["HVACUsage"].astype(str).str.strip().str.title()
        df["LightingUsage"] = df["LightingUsage"].astype(str).str.strip().str.title()
        df = df[df["HVACUsage"].isin(["On", "Off"]) & df["LightingUsage"].isin(["On", "Off"])].copy()

        df["Occupancy"] = df["Occupancy"].round().astype(int)
        return df.reset_index(drop=True)

    def deduplicate_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove redundant telemetry records based on timestamps and sort chronologically."""
        out = df.sort_values("Timestamp").drop_duplicates(subset=["Timestamp"], keep="first")
        return out.reset_index(drop=True)

