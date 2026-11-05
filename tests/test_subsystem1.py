"""Unit tests for Subsystem 1: Real-Time Energy Monitoring & Data Management."""

from datetime import datetime
from pathlib import Path
import unittest
import pandas as pd
from subsystem1 import (
    DEFAULT_BUILDING_CATALOG,
    DatabaseManager,
    DataIngestionEngine,
    EnergyLog,
    FeatureEngineer,
    MetricsCalculator,
)


class TestSubsystem1(unittest.TestCase):
    """Test suite for Subsystem 1 ingestion, feature engineering, metrics, and database."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.raw_csv = Path("data/raw/Energy_consumption.csv")
        cls.ingestion = DataIngestionEngine(cls.raw_csv)
        cls.clean_df = cls.ingestion.import_csv_batch()
        cls.engineer = FeatureEngineer(buildings=DEFAULT_BUILDING_CATALOG, random_seed=42)
        cls.engineered_df = cls.engineer.run_pipeline(cls.clean_df)
        cls.calculator = MetricsCalculator()
        cls.enriched_df = cls.calculator.enrich_metrics(cls.engineered_df)

    def test_option_a_row_count_and_buildings(self) -> None:
        """Verify Option A preserves 1,000 rows, integer occupancy, and clean columns."""
        self.assertEqual(len(self.enriched_df), 1000)
        self.assertEqual(
            set(self.enriched_df["building_type"].unique()),
            {"Home", "Office", "University"},
        )
        self.assertEqual(len(self.enriched_df["building_id"].unique()), 6)
        self.assertTrue(pd.api.types.is_integer_dtype(self.enriched_df["Occupancy"]))
        self.assertNotIn("occupancy_rate", self.enriched_df.columns)
        self.assertNotIn("is_holiday", self.enriched_df.columns)
        self.assertNotIn("hour_of_day", self.enriched_df.columns)

    def test_calendar_consistency(self) -> None:
        """Verify all Sundays are Holiday='Yes' and existing non-Sunday holidays from CSV are kept."""
        sundays = self.enriched_df[self.enriched_df["DayOfWeek"] == "Sunday"]
        self.assertTrue((sundays["Holiday"] == "Yes").all())

        non_sunday_holidays = self.enriched_df[
            (self.enriched_df["DayOfWeek"] != "Sunday") & (self.enriched_df["Holiday"] == "Yes")
        ]
        self.assertGreater(len(non_sunday_holidays), 100)

    def test_occupancy_patterns_by_building_type(self) -> None:
        """Verify weekend office occupancy is very low and evening home occupancy is high."""
        df = self.enriched_df
        hours = pd.to_datetime(df["Timestamp"]).dt.hour
        is_weekend = df["DayOfWeek"].isin(["Saturday", "Sunday"])
        is_holiday = df["Holiday"] == "Yes"

        office = df[df["building_type"] == "Office"]
        home = df[df["building_type"] == "Home"]

        office_weekend_mean = office[is_weekend.loc[office.index]]["Occupancy"].mean()
        office_weekday_work_mean = office[
            (~is_weekend.loc[office.index])
            & (~is_holiday.loc[office.index])
            & (hours.loc[office.index].between(9, 16))
        ]["Occupancy"].mean()

        home_evening_mean = home[hours.loc[home.index].between(18, 23)]["Occupancy"].mean()
        home_weekday_day_mean = home[
            (~is_weekend.loc[home.index])
            & (~is_holiday.loc[home.index])
            & (hours.loc[home.index].between(9, 16))
        ]["Occupancy"].mean()

        self.assertLess(office_weekend_mean, 1.5)
        self.assertGreater(office_weekday_work_mean, 25.0)
        self.assertGreater(home_evening_mean, 3.8)
        self.assertLess(home_weekday_day_mean, 2.0)

    def test_metrics_and_database_roundtrip(self) -> None:
        """Verify cost, carbon, sustainability scores, and SQLite query roundtrip."""
        scores = self.calculator.calculate_sustainability_scores(self.enriched_df)
        self.assertEqual(len(scores), 6)
        for score in scores.values():
            self.assertTrue(0.0 <= score <= 100.0)

        db = DatabaseManager(":memory:")
        db.save_buildings(DEFAULT_BUILDING_CATALOG)
        log = EnergyLog(
            timestamp=datetime(2022, 1, 5, 10, 0, 0),
            building_id="BLD_OFF_01",
            building_type="Office",
            area_sqft=1420.0,
            temperature=25.0,
            humidity=45.0,
            occupancy=30,
            hvac_usage="On",
            lighting_usage="On",
            renewable_energy_pct=15.0,
            day_of_week="Wednesday",
            holiday="No",
            energy_consumption_kwh=78.5,
            carbon_footprint_kg=28.02,
            energy_cost_eur=23.55,
            is_anomaly=False,
        )
        db.save_energy_log(log)
        queried = db.fetch_time_series_data(building_id="BLD_OFF_01")
        self.assertEqual(len(queried), 1)
        self.assertEqual(queried.iloc[0]["Occupancy"], 30)


if __name__ == "__main__":
    unittest.main()
