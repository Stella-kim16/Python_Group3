"""Main entry point for Subsystem 1: Real-Time Energy Monitoring & Data Management."""

from pathlib import Path
import pandas as pd
from subsystem1 import (
    DEFAULT_BUILDING_CATALOG,
    DatabaseManager,
    DataIngestionEngine,
    FeatureEngineer,
    MetricsCalculator,
)


def run_subsystem1_pipeline(
    raw_csv_path: str | Path = "data/raw/Energy_consumption.csv",
    db_path: str | Path = "data/processed/smart_energy.db",
    output_csv_path: str | Path = "data/processed/enriched_energy_data.csv",
    output_json_path: str | Path = "data/processed/enriched_energy_data.json",
) -> pd.DataFrame:
    """Run the full Subsystem 1 ingestion, feature engineering, metrics, and storage pipeline."""
    # 1. Ingest and validate raw Kaggle CSV
    ingestion = DataIngestionEngine(raw_csv_path)
    clean_df = ingestion.import_csv_batch()

    # 2. Feature engineering (Building assignment, calendar fix, occupancy rules, energy recalibration)
    engineer = FeatureEngineer(buildings=DEFAULT_BUILDING_CATALOG, random_seed=42)
    engineered_df = engineer.run_pipeline(clean_df)

    # 3. Calculate financial, carbon, and anomaly metrics
    calculator = MetricsCalculator()
    enriched_df = calculator.enrich_metrics(engineered_df)

    # 4. Persist to SQLite and export processed CSV & JSON for Subsystems 2 and 3
    db = DatabaseManager(connection_string=db_path)
    db.save_buildings(DEFAULT_BUILDING_CATALOG)
    db.save_energy_logs_df(enriched_df)
    db.export_to_csv(output_csv_path)
    db.export_to_json(output_json_path)

    return enriched_df


def print_summary_report(df: pd.DataFrame) -> None:
    """Print a diagnostic summary verifying the feature engineering rules."""
    calculator = MetricsCalculator()
    scores = calculator.calculate_sustainability_scores(df)

    print("=" * 72)
    print("SUBSYSTEM 1: REAL-TIME ENERGY MONITORING & DATA MANAGEMENT SUMMARY")
    print("=" * 72)
    print(f"Total Validated & Engineered Records : {len(df):,}")
    print(f"Date Range                           : {df['Timestamp'].min()} -> {df['Timestamp'].max()}")
    print(f"Total Energy Consumption             : {df['EnergyConsumption'].sum():,.2f} kWh")
    print(f"Total Energy Cost                    : EUR {df['energy_cost_eur'].sum():,.2f}")
    print(f"Total Carbon Footprint               : {df['carbon_footprint_kg'].sum():,.2f} kg CO2")
    print(f"Anomalies Flagged                    : {int(df['is_anomaly'].sum())} records")
    print("-" * 72)

    print("\n1. BUILDING CATALOG & SUSTAINABILITY SCORES:")
    b_summary = (
        df.groupby(["building_id", "building_type", "SquareFootage", "max_occupancy"])
        .agg(
            records=("Timestamp", "count"),
            avg_occ=("Occupancy", lambda s: int(round(s.mean()))),
            avg_kwh=("EnergyConsumption", "mean"),
            avg_cost=("energy_cost_eur", "mean"),
            avg_carbon=("carbon_footprint_kg", "mean"),
        )
        .reset_index()
    )
    b_summary["sustainability_score"] = b_summary["building_id"].map(scores)
    print(b_summary.to_string(index=False))

    print("\n2. OCCUPANCY VERIFICATION BY BUILDING TYPE & SCHEDULE (Rounded Mean Headcount):")
    hours = pd.to_datetime(df["Timestamp"]).dt.hour
    is_weekend = df["DayOfWeek"].isin(["Saturday", "Sunday"])

    office_df = df[df["building_type"] == "Office"]
    home_df = df[df["building_type"] == "Home"]
    uni_df = df[df["building_type"] == "University"]

    off_wkday_day = int(round(office_df[(~is_weekend.loc[office_df.index]) & (hours.loc[office_df.index].between(9, 16))]["Occupancy"].mean()))
    off_wkend = int(round(office_df[is_weekend.loc[office_df.index]]["Occupancy"].mean()))

    home_wkday_day = int(round(home_df[(~is_weekend.loc[home_df.index]) & (hours.loc[home_df.index].between(9, 16))]["Occupancy"].mean()))
    home_evening = int(round(home_df[hours.loc[home_df.index].between(18, 23)]["Occupancy"].mean()))

    uni_wkday_day = int(round(uni_df[(~is_weekend.loc[uni_df.index]) & (hours.loc[uni_df.index].between(9, 16))]["Occupancy"].mean()))
    uni_wkend = int(round(uni_df[is_weekend.loc[uni_df.index]]["Occupancy"].mean()))

    print(f"  - Office     | Weekday Work Hours (09-16): {off_wkday_day:3d} people | Weekends: {off_wkend:3d} people")
    print(f"  - Home       | Weekday Daytime    (09-16): {home_wkday_day:3d} people | Evenings: {home_evening:3d} people")
    print(f"  - University | Weekday Lectures   (09-16): {uni_wkday_day:3d} people | Weekends: {uni_wkend:3d} people")

    print("\n3. HOLIDAY vs. NON-HOLIDAY OCCUPANCY (Rounded Mean Headcount):")
    hol_summary = (
        df.groupby(["building_type", "Holiday"])["Occupancy"]
        .mean()
        .round()
        .astype(int)
        .unstack()
    )
    print(hol_summary.to_string())
    print("=" * 72)


if __name__ == "__main__":
    final_df = run_subsystem1_pipeline()
    print_summary_report(final_df)
