"""Data Ingestion Engine, Smart Meter Stream Iterator, and Sub-Threshold Filter."""

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Union
import pandas as pd
from subsystem1.models import EnergyLog, StreamProcessor
from subsystem1.exceptions_subsystem1 import DataIngestionError
from subsystem1.exceptions_subsystem1 import DataIngestionError, DataProcessingError
from subsystem1.exceptions_subsystem1 import DataProcessingError


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


class SmartMeterStream:
    """
    Custom Iterator and Generator for consuming
    high-frequency telemetry readings from smart meters across buildings on the fly.
    Uses only the exact columns present in the CSV dataset.
    """

    def __init__(self, source: Union[str, Path, pd.DataFrame, Iterable[Dict[str, Any]]]) -> None:
        try:
            if isinstance(source, (str, Path)):
                path = Path(source)
                if not path.exists():
                    raise FileNotFoundError(f"Stream source file not found: {path}")
                df = pd.read_csv(path)
                self._records: List[Dict[str, Any]] = df.to_dict(orient="records")
            elif isinstance(source, pd.DataFrame):
                self._records = source.to_dict(orient="records")
            else:
                self._records = list(source)
            
            if not self._records:
                raise ValueError("Stream source resulted in an empty record collection.")
                
            self._index: int = 0

        except Exception as e:
            raise DataIngestionError(f"Failed to initialise SmartMeterStream from source: {e}") from e

    def __iter__(self) -> "SmartMeterStream":
        """Reset stream cursor and return the iterator instance."""
        try:
            self._index = 0
            return self
        except Exception as e:
            raise DataIngestionError(f"Error resetting stream iterator cursor: {e}") from e

    def __next__(self) -> Dict[str, Any]:
        """Return the next smart meter reading or raise StopIteration."""
        try:
            if self._index >= len(self._records):
                raise StopIteration
            raw = dict(self._records[self._index])
            self._index += 1
            return raw
        except StopIteration:
            raise
        except Exception as e:
            raise DataIngestionError(f"Error reading next telemetry record from stream: {e}") from e

    def event_generator(self) -> Iterator[Dict[str, Any]]:
        """Generator function (`yield`) emitting smart meter events one by one."""
        try:
            for record in self:
                yield record
        except Exception as e:
            raise DataIngestionError(f"Error during smart meter event generation: {e}") from e


class SubThresholdMeterFilter(StreamProcessor):
    """
    Specialized StreamProcessor that filters out inactive or sub-threshold
    smart meter readings based on `EnergyConsumption` (and optional HVAC/Lighting
    activity) using the existing CSV/DB columns (Pipeline Goal 1).
    """

    def __init__(
        self,
        min_power_kwh: float = 25.0,
        require_active_systems: bool = False,
    ) -> None:
        try:
            self.min_power_kwh = float(min_power_kwh)
            self.require_active_systems = require_active_systems
        except Exception as e:
            raise DataProcessingError(f"Failed to initialize SubThresholdMeterFilter: {e}") from e

    def process(
        self, data: Union[Dict[str, Any], EnergyLog, pd.DataFrame, None]
    ) -> Union[Dict[str, Any], EnergyLog, pd.DataFrame, None]:
        """
        Polymorphic `process` implementation with custom exception safety.
        """
        try:
            if data is None:
                return None

            if isinstance(data, pd.DataFrame):
                df = data.copy()
                if "EnergyConsumption" in df.columns:
                    df = df[pd.to_numeric(df["EnergyConsumption"], errors="coerce") >= self.min_power_kwh]
                if self.require_active_systems and {"HVACUsage", "LightingUsage"}.issubset(df.columns):
                    df = df[(df["HVACUsage"] == "On") | (df["LightingUsage"] == "On")]
                return df.reset_index(drop=True)

            if isinstance(data, EnergyLog):
                if data.energy_consumption_kwh < self.min_power_kwh or not data.validate_data_types():
                    return None
                if (
                    self.require_active_systems
                    and data.hvac_usage == "Off"
                    and data.lighting_usage == "Off"
                ):
                    return None
                return data

            if isinstance(data, dict):
                try:
                    power_kwh = float(data.get("EnergyConsumption", data.get("energy_consumption_kwh", -1.0)))
                    sqft = float(data.get("SquareFootage", data.get("area_sqft", 1.0)))
                    hum = float(data.get("Humidity", data.get("humidity", 50.0)))
                    ren = float(data.get("RenewableEnergy", data.get("renewable_energy_pct", 0.0)))
                except (TypeError, ValueError):
                    return None

                if (
                    power_kwh < self.min_power_kwh
                    or sqft <= 0
                    or not (0.0 <= hum <= 100.0)
                    or not (0.0 <= ren <= 100.0)
                ):
                    return None

                if self.require_active_systems:
                    hvac = str(data.get("HVACUsage", data.get("hvac_usage", "Off"))).strip().title()
                    light = str(data.get("LightingUsage", data.get("lighting_usage", "Off"))).strip().title()
                    if hvac == "Off" and light == "Off":
                        return None

                return data

            raise TypeError(f"Unsupported data type for SubThresholdMeterFilter: {type(data)}")

        except Exception as e:
            if isinstance(e, DataProcessingError):
                raise
            raise DataProcessingError(f"Error filtering records in SubThresholdMeterFilter: {e}") from e


class DataIngestionEngine:
    """
    Ingests, validates, filters, and streams raw energy telemetry data 
    with robust custom exception handling.
    """

    def __init__(
        self,
        raw_file_path: Union[str, Path, None] = None,
        min_power_threshold: float = 0.0,
    ) -> None:
        try:
            self.raw_file_path = Path(raw_file_path) if raw_file_path else None
            self.min_power_threshold = float(min_power_threshold)
            self._stream_records: List[Dict[str, Any]] = []
            self._stream_index: int = 0
        except Exception as e:
            raise DataIngestionError(f"Failed to initialise DataIngestionEngine: {e}") from e

    def __iter__(self) -> "DataIngestionEngine":
        """Initialize iterator over the cleaned telemetry records (`__iter__`)."""
        try:
            if self.raw_file_path is None:
                raise ValueError("Cannot iterate DataIngestionEngine without a `raw_file_path`.")
            df = self.import_csv_batch(self.raw_file_path)
            self._stream_records = df.to_dict(orient="records")
            self._stream_index = 0
            return self
        except Exception as e:
            raise DataIngestionError(f"Error resetting or starting iterator stream: {e}") from e

    def __next__(self) -> Dict[str, Any]:
        """Yield the next telemetry event from the stream (`__next__`)."""
        try:
            if self._stream_index >= len(self._stream_records):
                raise StopIteration
            event = dict(self._stream_records[self._stream_index])
            self._stream_index += 1
            return event
        except StopIteration:
            raise
        except Exception as e:
            raise DataIngestionError(f"Error retrieving next stream event: {e}") from e

    def stream_events(
        self, file_path: Union[str, Path, None] = None
    ) -> Iterator[Dict[str, Any]]:
        """Generator (`yield`) that streams validated meter readings on the fly."""
        try:
            target = Path(file_path) if file_path else self.raw_file_path
            if target is None:
                raise ValueError("File path is required for streaming events.")
            df = self.import_csv_batch(target)
            for row in df.to_dict(orient="records"):
                yield row
        except Exception as e:
            raise DataIngestionError(f"Error during streaming events generator: {e}") from e

    def process(
        self, data: Union[Dict[str, Any], pd.DataFrame, str, Path, None]
    ) -> Union[Dict[str, Any], pd.DataFrame, None]:
        """
        Uniform `StreamProcessor.process` implementation with custom error encapsulation.
        """
        try:
            if data is None:
                return None

            if isinstance(data, (str, Path)):
                return self.import_csv_batch(data)

            if isinstance(data, pd.DataFrame):
                cleaned = self.clean_missing_data(data)
                return self.deduplicate_timestamps(cleaned)

            if isinstance(data, dict):
                missing_cols = [col for col in REQUIRED_RAW_COLUMNS if col not in data]
                if missing_cols:
                    return None
                df = pd.DataFrame([data])
                cleaned_df = self.clean_missing_data(df)
                if cleaned_df.empty:
                    return None
                return cleaned_df.iloc[0].to_dict()

            raise TypeError(f"Unsupported input type for DataIngestionEngine.process: {type(data)}")

        except Exception as e:
            if isinstance(e, (DataIngestionError, DataProcessingError)):
                raise
            raise DataProcessingError(f"Error processing data input in DataIngestionEngine: {e}") from e

    def import_csv_batch(self, file_path: Union[str, Path, None] = None) -> pd.DataFrame:
        """Load a batch CSV file, validate schema, deduplicate, and clean invalid rows."""
        try:
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

        except Exception as e:
            raise DataIngestionError(f"Failed to ingest CSV batch from '{file_path}': {e}") from e

    def ingest_sensor_stream(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """Validate and convert a single real-time JSON/dict sensor payload into a 1-row DataFrame."""
        try:
            missing_cols = [col for col in REQUIRED_RAW_COLUMNS if col not in payload]
            if missing_cols:
                raise ValueError(f"Sensor payload missing required keys: {missing_cols}")

            df = pd.DataFrame([payload])
            cleaned_df = self.clean_missing_data(df)
            if cleaned_df.empty:
                raise ValueError("Sensor payload failed data validation checks.")
            return cleaned_df

        except Exception as e:
            raise DataIngestionError(f"Failed to ingest sensor stream payload: {e}") from e

    def clean_missing_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Filter or impute missing, negative, or out-of-range telemetry values safely."""
        try:
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

            # Filter out physically invalid / negative or sub-threshold values
            valid_mask = (
                (df["SquareFootage"] > 0)
                & (df["Occupancy"] >= 0)
                & (df["Humidity"] >= 0.0)
                & (df["Humidity"] <= 100.0)
                & (df["RenewableEnergy"] >= 0.0)
                & (df["RenewableEnergy"] <= 100.0)
                & (df["EnergyConsumption"] >= self.min_power_threshold)
            )
            df = df.loc[valid_mask].copy()

            # Normalize categorical usage strings
            df["HVACUsage"] = df["HVACUsage"].astype(str).str.strip().str.title()
            df["LightingUsage"] = df["LightingUsage"].astype(str).str.strip().str.title()
            df = df[df["HVACUsage"].isin(["On", "Off"]) & df["LightingUsage"].isin(["On", "Off"])].copy()

            df["Occupancy"] = df["Occupancy"].round().astype(int)
            return df.reset_index(drop=True)

        except Exception as e:
            raise DataProcessingError(f"Error cleaning missing or malformed telemetry data: {e}") from e

    def deduplicate_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove redundant telemetry records based on timestamps and sort chronologically."""
        try:
            out = df.sort_values("Timestamp").drop_duplicates(subset=["Timestamp"], keep="first")
            return out.reset_index(drop=True)
        except Exception as e:
            raise DataProcessingError(f"Error deduplicating timestamps: {e}") from e
