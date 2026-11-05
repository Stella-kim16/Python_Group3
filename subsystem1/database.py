"""Database Manager for SQLite persistence, time-series queries, and CSV/JSON exports."""

import json
from pathlib import Path
import sqlite3
from typing import List, Optional, Union
import pandas as pd
from subsystem1.models import Building, EnergyLog
from subsystem1.exceptions_subsystem1 import DataIngestionError, DataProcessingError


class DatabaseManager:
    """Manages SQLite relational storage and file exports for buildings and energy telemetry with custom exception safety."""

    def __init__(self, connection_string: Union[str, Path] = "data/processed/smart_energy.db") -> None:
        try:
            self.connection_string = str(connection_string)
            self._memory_conn: Optional[sqlite3.Connection] = None
            if self.connection_string == ":memory:":
                self._memory_conn = sqlite3.connect(":memory:")
                self._memory_conn.execute("PRAGMA foreign_keys = ON;")
            else:
                Path(self.connection_string).parent.mkdir(parents=True, exist_ok=True)
            self._init_schema()
        except Exception as e:
            raise DataIngestionError(f"Failed to initialize DatabaseManager: {e}") from e

    def _get_connection(self) -> sqlite3.Connection:
        try:
            if self._memory_conn is not None:
                return self._memory_conn
            conn = sqlite3.connect(self.connection_string)
            conn.execute("PRAGMA foreign_keys = ON;")
            return conn
        except Exception as e:
            raise DataIngestionError(f"Failed to establish SQLite database connection: {e}") from e

    def _init_schema(self) -> None:
        """Create buildings and energy_logs tables with foreign keys and time-series indexes."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS buildings (
                        building_id TEXT PRIMARY KEY,
                        building_type TEXT NOT NULL,
                        area_sqft REAL NOT NULL,
                        max_occupancy INTEGER NOT NULL,
                        baseline_kwh REAL NOT NULL
                    );
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS energy_logs (
                        Timestamp TEXT NOT NULL,
                        building_id TEXT NOT NULL,
                        building_type TEXT NOT NULL,
                        SquareFootage REAL NOT NULL,
                        max_occupancy INTEGER NOT NULL,
                        Temperature REAL NOT NULL,
                        Humidity REAL NOT NULL,
                        Occupancy INTEGER NOT NULL,
                        HVACUsage TEXT NOT NULL,
                        LightingUsage TEXT NOT NULL,
                        RenewableEnergy REAL NOT NULL,
                        DayOfWeek TEXT NOT NULL,
                        Holiday TEXT NOT NULL,
                        EnergyConsumption REAL NOT NULL,
                        cost_per_kwh REAL NOT NULL,
                        energy_cost_eur REAL NOT NULL,
                        carbon_footprint_kg REAL NOT NULL,
                        is_anomaly INTEGER NOT NULL,
                        PRIMARY KEY (Timestamp, building_id),
                        FOREIGN KEY (building_id) REFERENCES buildings (building_id)
                    );
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_logs_building_time ON energy_logs (building_id, Timestamp);"
                )
        except Exception as e:
            raise DataIngestionError(f"Failed to initialize database schema: {e}") from e

    def save_building_record(self, building: Building) -> None:
        """Insert or update a single Building record in SQLite."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO buildings
                    (building_id, building_type, area_sqft, max_occupancy, baseline_kwh)
                    VALUES (?, ?, ?, ?, ?);
                    """,
                    (
                        building.building_id,
                        building.building_type,
                        building.area_sqft,
                        building.max_occupancy,
                        building.baseline_kwh,
                    ),
                )
        except Exception as e:
            raise DataIngestionError(f"Failed to save building record '{building.building_id}': {e}") from e

    def save_buildings(self, buildings: List[Building]) -> None:
        """Batch save a list of Building instances."""
        try:
            for b in buildings:
                self.save_building_record(b)
        except Exception as e:
            raise DataIngestionError(f"Failed during batch building records save: {e}") from e

    def save_energy_log(self, log: EnergyLog) -> None:
        """Validate and save a single EnergyLog domain object into SQLite."""
        try:
            if not log.validate_data_types():
                raise ValueError(f"Invalid EnergyLog record at {log.timestamp}")

            cost_per_kwh = round(log.energy_cost_eur / max(log.energy_consumption_kwh, 1e-6), 4)

            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO energy_logs (
                        Timestamp, building_id, building_type, SquareFootage, max_occupancy,
                        Temperature, Humidity, Occupancy, HVACUsage, LightingUsage,
                        RenewableEnergy, DayOfWeek, Holiday, EnergyConsumption, cost_per_kwh,
                        energy_cost_eur, carbon_footprint_kg, is_anomaly
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        log.building_id,
                        log.building_type,
                        log.area_sqft,
                        max(log.occupancy, 1),
                        log.temperature,
                        log.humidity,
                        int(log.occupancy),
                        log.hvac_usage,
                        log.lighting_usage,
                        log.renewable_energy_pct,
                        log.day_of_week,
                        log.holiday,
                        log.energy_consumption_kwh,
                        cost_per_kwh,
                        log.energy_cost_eur,
                        log.carbon_footprint_kg,
                        int(log.is_anomaly),
                    ),
                )
        except Exception as e:
            if isinstance(e, DataIngestionError):
                raise
            raise DataIngestionError(f"Failed to save energy log record: {e}") from e

    def save_energy_logs_df(self, df: pd.DataFrame) -> None:
        """Persist an entire feature-engineered and enriched DataFrame into SQLite."""
        try:
            to_save = df.copy()
            to_save["Timestamp"] = pd.to_datetime(to_save["Timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
            to_save["Occupancy"] = to_save["Occupancy"].astype(int)
            to_save["max_occupancy"] = to_save["max_occupancy"].astype(int)
            to_save["is_anomaly"] = to_save["is_anomaly"].astype(int)

            with self._get_connection() as conn:
                conn.execute("DROP TABLE IF EXISTS energy_logs;")
                self._init_schema()
                to_save.to_sql("energy_logs", conn, if_exists="append", index=False)
        except Exception as e:
            raise DataProcessingError(f"Failed to persist energy logs dataframe to SQLite: {e}") from e

    def fetch_time_series_data(
        self,
        building_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Query historical energy logs filtered by optional building_id and date range."""
        try:
            query = "SELECT * FROM energy_logs WHERE 1=1"
            params: List[str] = []

            if building_id:
                query += " AND building_id = ?"
                params.append(building_id)
            if start_date:
                query += " AND Timestamp >= ?"
                params.append(start_date)
            if end_date:
                query += " AND Timestamp <= ?"
                params.append(end_date)

            query += " ORDER BY Timestamp ASC;"

            with self._get_connection() as conn:
                df = pd.read_sql_query(query, conn, params=params)

            if not df.empty:
                df["Timestamp"] = pd.to_datetime(df["Timestamp"])
                df["Occupancy"] = df["Occupancy"].astype(int)
                df["max_occupancy"] = df["max_occupancy"].astype(int)
                df["is_anomaly"] = df["is_anomaly"].astype(bool)
            return df
        except Exception as e:
            raise DataProcessingError(f"Failed to fetch time-series data from database: {e}") from e

    def export_to_csv(
        self, file_path: Union[str, Path] = "data/processed/enriched_energy_data.csv"
    ) -> Path:
        """Export all processed energy logs from SQLite to a CSV file."""
        try:
            target = Path(file_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            df = self.fetch_time_series_data()
            if not df.empty:
                df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            df.to_csv(target, index=False)
            return target
        except Exception as e:
            raise DataIngestionError(f"Failed to export energy logs to CSV: {e}") from e

    def export_to_json(
        self, file_path: Union[str, Path] = "data/processed/enriched_energy_data.json"
    ) -> Path:
        """Export all processed energy logs from SQLite to a standard JSON file."""
        try:
            target = Path(file_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            df = self.fetch_time_series_data()
            if not df.empty:
                df["Timestamp"] = df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            records = df.to_dict(orient="records")
            target.write_text(json.dumps(records, indent=2), encoding="utf-8")
            return target
        except Exception as e:
            raise DataIngestionError(f"Failed to export energy logs to JSON: {e}") from e