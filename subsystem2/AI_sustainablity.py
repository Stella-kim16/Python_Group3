from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# 빌딩별 지속가능성 점수 (0~100, 높을수록 좋음)
# 세부 점수 4개를 가중 평균해서 총점을 만든다
#   waste      : 빈 방에서 HVAC / 조명이 켜져 낭비된 에너지 비율 (모델로 추정)
#   renewable  : 사용량 중 재생에너지 비율
#   peak       : 비싼(피크) 요금 시간대에 쓴 에너지 비율
#   efficiency : 면적당 사용량을 같은 유형 빌딩과 비교 (= Current Efficiency Score)

# ---------------------------------------------------------------
# 설정값 — 팀에서 바꿀 수 있는 기준
# ---------------------------------------------------------------
WEIGHTS = {"waste": 0.4, "renewable": 0.3, "peak": 0.2, "efficiency": 0.1}  # 합 = 1
WASTE_LIMIT = 0.10        # 낭비 비율이 10% 이상이면 waste 점수 0점
RENEWABLE_TARGET = 0.50   # 재생에너지 비율 50% 이상이면 renewable 점수 100점
PEAK_LIMIT = 0.50         # 피크 시간 사용 비율이 50% 이상이면 peak 점수 0점

BUILDING_PREFIX = "Bld_"
OCCUPANCY_COL = "Occupancy"
SWITCH_OFF_COLS = ["HVACUsage", "LightingUsage"]   # 빈 방이면 꺼야 하는 기기
RAW_COLS = ["building_id", "building_type", "EnergyConsumption", "RenewableEnergy",
            "cost_per_kwh", "SquareFootage", "carbon_footprint_kg", "Timestamp"]


@dataclass
class ScoreResult:
    """Sustainability score of one building with the numbers behind it."""
    building_id: str
    building_type: str
    total_score: float                 # 0~100 (인터페이스의 큰 숫자, 예: 74/100)
    efficiency_score: float            # 0~100 (Current Efficiency 게이지)
    sub_scores: dict = field(default_factory=dict)   # waste / renewable / peak / efficiency 점수
    details: dict = field(default_factory=dict)      # 점수 계산에 쓴 실제 수치


def _clip(value: float) -> float:
    """Keep a score between 0 and 100."""
    return float(np.clip(value, 0, 100))


class SustainabilityScorer:
    """Score how sustainably each building uses energy."""

    def __init__(self, model, raw_df: pd.DataFrame, weights: dict | None = None):
        # model   : train() 이 끝난 AIforecast 객체 (낭비 kWh 추정에 사용)
        # raw_df  : pd.read_csv() 로 읽은 원본 CSV (실측 사용량 / 재생에너지 / 단가 / 면적)
        # weights : 세부 점수 가중치 (없으면 WEIGHTS 사용)
        if not model.is_trained:
            raise RuntimeError("Train the AIforecast model before creating SustainabilityScorer.")
        missing = [c for c in RAW_COLS if c not in raw_df.columns]
        if missing:
            raise KeyError(f"Columns missing in raw CSV: {missing}")

        self.model = model
        self.raw = raw_df
        self.weights = weights or WEIGHTS
        if abs(sum(self.weights.values()) - 1) > 1e-6:
            raise ValueError("Weights must add up to 1.")

        self.history_X = model.history_X()     # 모델이 본 입력 (이상치 제외)
        self.building_cols = [c for c in self.history_X.columns if c.startswith(BUILDING_PREFIX)]
        if not self.building_cols:
            raise RuntimeError("No 'Bld_' columns found — one-hot encode 'building_id' first.")

        # 피크 요금 = 가장 싼 단가보다 비싼 시간 (이 데이터: 0.15 € = 기본, 0.30 € = 피크)
        self.offpeak_price = self.raw["cost_per_kwh"].min()

        # 같은 유형(Home / Office / University) 빌딩의 면적당 사용량 중 가장 좋은 값
        intensity = (self.raw["EnergyConsumption"] / self.raw["SquareFootage"])
        per_building = intensity.groupby(self.raw["building_id"]).mean()
        types = self.raw.groupby("building_id")["building_type"].first()
        self.intensity = per_building                               # 빌딩별 kWh / sqft
        self.best_in_type = per_building.groupby(types).min()       # 유형별 최솟값
        self.types = types

    # -----------------------------------------------------------
    # 빌딩 목록
    # -----------------------------------------------------------
    def buildings(self) -> list[str]:
        return sorted(c[len(BUILDING_PREFIX):] for c in self.building_cols)

    # -----------------------------------------------------------
    # 세부 점수 계산
    # -----------------------------------------------------------
    def _waste(self, building_id: str) -> tuple[float, dict]:
        """Model-estimated share of energy that would be saved by switching devices off in empty rooms."""
        hist = self.history_X[self.history_X[f"{BUILDING_PREFIX}{building_id}"] == 1]
        base_kwh = self.model.predict(hist)                    # 지금 상태 그대로

        no_waste = hist.copy()                                 # 빈 방이면 HVAC / 조명 Off
        empty = no_waste[OCCUPANCY_COL] == 0
        cols = [c for c in SWITCH_OFF_COLS if c in no_waste.columns]
        wasted_hours = int((empty & (no_waste[cols] == 1).any(axis=1)).sum())
        no_waste.loc[empty, cols] = 0
        rule_kwh = self.model.predict(no_waste)

        waste_kwh = float((base_kwh - rule_kwh).clip(lower=0).sum())
        ratio = waste_kwh / float(base_kwh.sum())
        score = _clip(100 * (1 - ratio / WASTE_LIMIT))
        return score, {"wasted_kWh": round(waste_kwh, 2),
                       "waste_ratio": round(ratio, 4),
                       "wasted_hours": wasted_hours}

    def _renewable(self, rows: pd.DataFrame) -> tuple[float, dict]:
        share = rows["RenewableEnergy"].sum() / rows["EnergyConsumption"].sum()
        return _clip(100 * share / RENEWABLE_TARGET), {"renewable_share": round(float(share), 4)}

    def _peak(self, rows: pd.DataFrame) -> tuple[float, dict]:
        is_peak = rows["cost_per_kwh"] > self.offpeak_price
        share = rows.loc[is_peak, "EnergyConsumption"].sum() / rows["EnergyConsumption"].sum()
        return _clip(100 * (1 - share / PEAK_LIMIT)), {"peak_share": round(float(share), 4)}

    def _efficiency(self, building_id: str) -> tuple[float, dict]:
        own = self.intensity[building_id]
        best = self.best_in_type[self.types[building_id]]
        return _clip(100 * best / own), {"kWh_per_1000_sqft": round(float(own * 1000), 2)}

    # -----------------------------------------------------------
    # 빌딩 하나 점수
    # -----------------------------------------------------------
    def score_building(self, building_id: str) -> ScoreResult:
        if f"{BUILDING_PREFIX}{building_id}" not in self.building_cols:
            raise ValueError(f"Unknown building '{building_id}'. Available: {self.buildings()}")

        rows = self.raw[self.raw["building_id"] == building_id]   # 이 빌딩의 실측 데이터

        waste, d1 = self._waste(building_id)
        renewable, d2 = self._renewable(rows)
        peak, d3 = self._peak(rows)
        efficiency, d4 = self._efficiency(building_id)
        sub_scores = {"waste": round(waste, 1), "renewable": round(renewable, 1),
                      "peak": round(peak, 1), "efficiency": round(efficiency, 1)}

        total = sum(self.weights[k] * v for k, v in sub_scores.items())

        n_days = pd.to_datetime(rows["Timestamp"]).dt.date.nunique()
        details = {**d1, **d2, **d3, **d4,
                   "carbon_kg_per_day": round(float(rows["carbon_footprint_kg"].sum() / n_days), 2)}

        return ScoreResult(
            building_id=building_id,
            building_type=self.types[building_id],
            total_score=round(total, 1),
            efficiency_score=round(efficiency, 1),
            sub_scores=sub_scores,
            details=details,
        )

    # -----------------------------------------------------------
    # 빌딩 전체 점수
    # -----------------------------------------------------------
    def score_all(self) -> dict[str, ScoreResult]:
        """{building_id: ScoreResult} for every building."""
        results = {}
        for building_id in self.buildings():
            results[building_id] = self.score_building(building_id)
            r = results[building_id]
            print(f"  {building_id}: {r.total_score}/100  {r.sub_scores}")
        return results