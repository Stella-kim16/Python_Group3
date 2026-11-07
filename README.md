# AI-Powered Smart Energy Management System — Event-Driven Stream Processing (Homework Assignment 02)

## 1. Author
- **Team**: Python Group 6
- **Members**:
  - Ecesu Yildiran `[ESE 7226935]`
  - Stella (Sung Eun Kim) `[ESE 7226046]`
  - Devashish Deshpande `[ESE 7225975]`

---

## 2. Project Description
The aim of this work is to design and implement an **event-driven stream processing framework** and real-time data pipeline for an **AI-Powered Smart Energy Management System** using core Object-Oriented Programming (OOP) abstractions in Python.

The system ingests high-frequency readings from smart meters across 6 smart buildings (`Home`, `Office`, and `University`), filters out inactive or sub-threshold meter readings, and transforms valid energy telemetry into updated load profiles, financial tariffs (peak/off-peak), carbon equivalents, and waste anomaly flags.

---

## 3. Technologies
- **Language**: Python 3.10+
- **Core Libraries**:
  - `abc`, `dataclasses`, `typing`, `pathlib`, `datetime`, `json` (Python Standard Library for OOP & stream abstractions)
  - `pandas` & `numpy` (Time-series data processing, feature engineering, and statistical z-score anomaly detection)
  - `sqlite3` (Relational persistence with foreign-key constraints and time-series indexing in `DatabaseManager`)

---

## 4. How to Run

### 1. Install Dependencies
```bash
pip install pandas numpy
```

### 2. Run the Main Subsystem 1 Pipeline
Executes data ingestion, feature engineering, metrics calculation, SQLite database persistence, and prints the summary report:
```bash
python main.py
```

---

## 5. Data

### Raw Input Data (`data/raw/Energy_consumption.csv`)
Contains 1,000 hourly smart meter telemetry records with 11 raw attributes:
- `Timestamp`, `Temperature`, `Humidity`, `SquareFootage`, `Occupancy`, `HVACUsage`, `LightingUsage`, `RenewableEnergy`, `DayOfWeek`, `Holiday`, `EnergyConsumption`

### Processed Outputs (`data/processed/`)
Generated automatically by `DatabaseManager` after running the pipeline:
- **`data/processed/smart_energy.db`**: SQLite database containing the `buildings` table (6 buildings) and `energy_logs` table.
- **`data/processed/enriched_energy_data.csv`** & **`data/processed/enriched_energy_data.json`**: Enriched 18-column dataset containing:
  - Building metadata: `building_id`, `building_type`, `SquareFootage`, `max_occupancy`
  - Environmental & equipment telemetry: `Timestamp`, `Temperature`, `Humidity`, `Occupancy`, `HVACUsage`, `LightingUsage`, `RenewableEnergy`, `DayOfWeek`, `Holiday`
  - Transformed energy, tariff & carbon metrics: `EnergyConsumption`, `cost_per_kwh`, `energy_cost_eur`, `carbon_footprint_kg`, `is_anomaly`

---

## 6. Video link
- https://drive.google.com/file/d/1CXTnm7quibNwHVZBLlYZxje3-YZb-Zqz/view?usp=sharing

