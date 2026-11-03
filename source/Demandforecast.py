from dataclasses import dataclass

import pandas as pd

# 미래 24시간의 입력을 만들어서 계산기(AIforecast)에 물어보는 파트
# AIforecast : "이 조건이면 몇 kWh" 를 계산
# DemandForecaster : "다음 24시간은 보통 이런 조건" 을 만들어서 AIforecast 에 넣고 비용까지 계산

# ---------------------------------------------------------------
# 설정값 — enriched_energy_data.csv / Data_processor 기준
# ---------------------------------------------------------------
DAY_PREFIX = "Day_"          # one-hot 요일 컬럼 (Day_Monday, Day_Sunday ...)
BUILDING_PREFIX = "Bld_"     # one-hot 건물 컬럼 (Bld_BLD_OFF_01 ...)
HOLIDAY_COL = "Holiday"      # 1 = 휴일
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
META_COLS = ["Timestamp", "building_id", "cost_per_kwh", "EnergyConsumption"]


@dataclass
class ForecastResult:
    """Everything one building's Demand Forecast needs to show."""
    building_id: str
    hourly: pd.DataFrame          # 시간별 predicted_kWh, price_per_kWh, predicted_cost_eur
    total_kwh: float              # 24시간 합계 kWh
    total_cost_eur: float         # 24시간 합계 €
    recent_actual: pd.DataFrame   # 그래프 앞부분(실선)용 최근 실측값


class DemandForecaster:
    """Predict the next 24 hours of energy use for each building."""

    def __init__(self, model, raw_df: pd.DataFrame, min_rows: int = 3):
        # model    : train() 이 끝난 AIforecast 객체
        # raw_df   : pd.read_csv() 로 읽은 원본 CSV (Timestamp, building_id, 단가를 가져오기 위해)
        # min_rows : 대표값을 만들 때 최소로 필요한 과거 행 수
        if not model.is_trained:
            raise RuntimeError("Train the AIforecast model before creating DemandForecaster.")

        self.model = model
        self.min_rows = min_rows

        # 1) 모델이 학습에 본 한 달 반 전체 입력 (train + test)
        self.history_X = model.history_X()

        # 2) 같은 행 번호로 원본에서 시각 / 빌딩 이름 / 단가 / 실측값 가져오기
        self.meta = self._build_meta(raw_df)

        # 3) 자주 쓰는 컬럼 목록
        self.day_cols = [c for c in self.history_X.columns if c.startswith(DAY_PREFIX)]
        self.building_cols = [c for c in self.history_X.columns if c.startswith(BUILDING_PREFIX)]
        if not self.building_cols:
            raise RuntimeError(
                "No 'Bld_' columns found. In Data_processor.encode_categoricals(), "
                "one-hot encode 'building_id' with prefix='Bld' and drop_first=False.")
        # 값이 0/1 뿐인 컬럼 — 중앙값이 0.5 가 되면 반올림하기 위해
        self.binary_cols = [c for c in self.history_X.columns
                            if set(self.history_X[c].dropna().unique()) <= {0, 1}]

        # 4) 시간대별 평균 단가 (0~23시) — kWh 를 € 로 바꿀 때 사용
        self.hourly_price = (self.meta.groupby(self.meta["Timestamp"].dt.hour)["cost_per_kwh"]
                                      .mean())

    # -----------------------------------------------------------
    # 내부 도우미 함수
    # -----------------------------------------------------------
    def _build_meta(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Side information for the same rows as history_X (anomaly rows are already gone)."""
        missing = [c for c in META_COLS if c not in raw_df.columns]
        if missing:
            raise KeyError(f"Columns missing in raw CSV: {missing}")
        meta = raw_df.loc[self.history_X.index, META_COLS].copy()
        meta["Timestamp"] = pd.to_datetime(meta["Timestamp"], format=TIME_FORMAT)
        return meta

    def _is_day(self, X: pd.DataFrame, day_name: str) -> pd.Series:
        """True for rows on day_name. The day dropped by drop_first has all Day_ columns = 0."""
        col = f"{DAY_PREFIX}{day_name}"
        if col in X.columns:
            return X[col] == 1
        return (X[self.day_cols] == 0).all(axis=1)

    def _typical_row(self, hist: pd.DataFrame, ts: pd.Timestamp) -> pd.Series:
        """
        Typical input values for this building at time ts.
        Search widens step by step until at least min_rows past rows are found:
        same weekday + hour → same hour → hour ± 1 → whole building history.
        """
        candidates = [
            (hist["Hour"] == ts.hour) & self._is_day(hist, ts.day_name()),
            hist["Hour"] == ts.hour,
            hist["Hour"].isin([(ts.hour - 1) % 24, ts.hour, (ts.hour + 1) % 24]),
        ]
        rows = hist
        for mask in candidates:
            if mask.sum() >= self.min_rows:
                rows = hist[mask]
                break

        typical = rows.median()
        typical[self.binary_cols] = typical[self.binary_cols].round()
        return typical

    # -----------------------------------------------------------
    # 빌딩 목록
    # -----------------------------------------------------------
    def buildings(self) -> list[str]:
        """Building IDs that can be forecast, e.g. ['BLD_HOM_01', ...]."""
        return sorted(c[len(BUILDING_PREFIX):] for c in self.building_cols)

    # -----------------------------------------------------------
    # 빌딩 하나 예측
    # -----------------------------------------------------------
    def forecast_building(self, building_id: str, start: pd.Timestamp | None = None,
                          holidays=(), hours: int = 24) -> ForecastResult:
        # 1) 이 빌딩의 과거 데이터만 고르기
        bld_col = f"{BUILDING_PREFIX}{building_id}"
        if bld_col not in self.building_cols:
            raise ValueError(f"Unknown building '{building_id}'. Available: {self.buildings()}")
        hist = self.history_X[self.history_X[bld_col] == 1]

        # 2) 시작 시각: 지정하지 않으면 데이터의 마지막 시각 + 1시간
        if start is None:
            start = self.meta["Timestamp"].max() + pd.Timedelta(hours=1)

        # 3) 미래 24개 시각 (1시간 간격)
        future_hours = pd.date_range(start=pd.Timestamp(start).floor("h"),
                                     periods=hours, freq="h")
        holiday_dates = {pd.Timestamp(d).date() for d in holidays}

        # 4) 각 시각의 입력 한 줄 만들기
        rows = []
        for ts in future_hours:
            row = self._typical_row(hist, ts)          # 모르는 값 + 빌딩 고정값

            # 달력으로 아는 값은 실제 날짜로 덮어쓰기
            row["Hour"] = ts.hour
            if "Month" in row.index:
                row["Month"] = ts.month
            row[self.day_cols] = 0
            if f"{DAY_PREFIX}{ts.day_name()}" in row.index:
                row[f"{DAY_PREFIX}{ts.day_name()}"] = 1
            if HOLIDAY_COL in row.index:
                row[HOLIDAY_COL] = int(ts.date() in holiday_dates)
            rows.append(row)

        X_next = pd.DataFrame(rows, index=future_hours)

        # 5) AIforecast 로 예측 → 6) 단가를 곱해서 비용 계산
        kwh = self.model.predict(X_next).values
        price = self.hourly_price.reindex(future_hours.hour).values
        hourly = pd.DataFrame({
            "predicted_kWh": kwh,
            "price_per_kWh": price,
            "predicted_cost_eur": kwh * price,
        }, index=future_hours)

        # 7) 그래프 앞부분용: 이 빌딩의 최근 실측값 (마지막 2일)
        bld_meta = self.meta[self.meta["building_id"] == building_id]
        recent = bld_meta[bld_meta["Timestamp"] > future_hours[0] - pd.Timedelta(days=2)]
        recent_actual = recent.set_index("Timestamp")[["EnergyConsumption"]]

        return ForecastResult(
            building_id=building_id,
            hourly=hourly,
            total_kwh=round(float(kwh.sum()), 2),
            total_cost_eur=round(float((kwh * price).sum()), 2),
            recent_actual=recent_actual,
        )

    # -----------------------------------------------------------
    # 빌딩 전체 예측
    # -----------------------------------------------------------
    def forecast_all(self, start: pd.Timestamp | None = None,
                     holidays=()) -> dict[str, ForecastResult]:
        """{building_id: ForecastResult} for every building."""
        results = {}
        for building_id in self.buildings():
            results[building_id] = self.forecast_building(building_id, start, holidays)
            r = results[building_id]
            print(f"  {building_id}: {r.total_kwh:.1f} kWh, {r.total_cost_eur:.2f} €")
        return results