from dataclasses import dataclass

import pandas as pd

# "이렇게 바꾸면 얼마를 아낄 수 있다" 를 계산하는 파트
# 계산 원리:
#   1) 과거 데이터에서 낭비 조건에 맞는 시간을 찾는다 (예: 사람 0명 + HVAC On)
#   2) AIforecast 에 두 번 묻는다 — 그대로 / 그 시간만 기기를 끈 상태
#   3) 차이 = 아낄 수 있는 kWh → × 그 시간 단가 = €
#   4) 데이터 기간의 절약액을 하루 평균으로 바꾼 뒤 한 달로 환산

BUILDING_PREFIX = "Bld_"
DAYS_PER_MONTH = 30
META_COLS = ["Timestamp", "building_id", "cost_per_kwh"]
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass
class Recommendation:
    """One saving tip for one building, with the money behind it."""
    building_id: str
    title: str                    # 예: "Switch off HVAC in empty rooms"
    message: str                  # 사용자에게 보여줄 문장
    wasted_hours: int             # 데이터 기간 중 낭비 조건에 해당한 시간 수
    saved_kwh: float              # 데이터 기간 동안 아낄 수 있었던 kWh
    saved_eur: float              # 데이터 기간 동안 아낄 수 있었던 €
    saved_eur_per_month: float    # 한 달(30일) 환산 절약액
    saved_pct: float              # 이 빌딩 전체 비용 대비 %


# ---------------------------------------------------------------
# 시나리오: 부모 클래스 — 모든 절약 방법이 같은 형태를 가짐
# ---------------------------------------------------------------
class SavingScenario:
    title = "Base scenario"
    switch_off: list[str] = []          # 끌 기기 컬럼

    def waste_mask(self, X: pd.DataFrame) -> pd.Series:
        """True for hours where energy is wasted. Child classes define the condition."""
        raise NotImplementedError

    def apply(self, X: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Copy of X with the devices switched off in wasted hours."""
        mask = self.waste_mask(X)
        X_new = X.copy()
        X_new.loc[mask, self.switch_off] = 0
        return X_new, mask

    def message(self, saved_eur_per_month: float, wasted_hours: int) -> str:
        return f"{self.title}: save about {saved_eur_per_month:.2f} € per month."


# ---------------------------------------------------------------
# 시나리오: 자식 클래스 — 조건만 다르게
# ---------------------------------------------------------------
class HVACOffWhenEmpty(SavingScenario):
    """Nobody in the room but the HVAC is on → switch the HVAC off."""
    title = "Switch off HVAC in empty rooms"
    switch_off = ["HVACUsage"]

    def waste_mask(self, X: pd.DataFrame) -> pd.Series:
        return (X["Occupancy"] == 0) & (X["HVACUsage"] == 1)

    def message(self, saved_eur_per_month: float, wasted_hours: int) -> str:
        return (f"The HVAC was running in {wasted_hours} hours with nobody inside. "
                f"Switching it off saves about {saved_eur_per_month:.2f} € per month.")


class LightsOffWhenEmpty(SavingScenario):
    """Nobody in the room but the lights are on → switch the lights off."""
    title = "Switch off lights in empty rooms"
    switch_off = ["LightingUsage"]

    def waste_mask(self, X: pd.DataFrame) -> pd.Series:
        return (X["Occupancy"] == 0) & (X["LightingUsage"] == 1)

    def message(self, saved_eur_per_month: float, wasted_hours: int) -> str:
        return (f"The lights were on in {wasted_hours} hours with nobody inside. "
                f"Switching them off saves about {saved_eur_per_month:.2f} € per month.")


# ---------------------------------------------------------------
# 절약액 계산기
# ---------------------------------------------------------------
class SavingsAdvisor:
    """Estimate how much money each saving scenario would save per building."""

    def __init__(self, model, raw_df: pd.DataFrame, scenarios: list[SavingScenario] | None = None):
        # model     : train() 이 끝난 AIforecast 객체
        # raw_df    : pd.read_csv() 로 읽은 원본 CSV (시각 / 빌딩 이름 / 단가)
        # scenarios : 계산할 절약 방법 목록 (없으면 HVAC + 조명)
        if not model.is_trained:
            raise RuntimeError("Train the AIforecast model before creating SavingsAdvisor.")

        self.model = model
        self.scenarios = scenarios or [HVACOffWhenEmpty(), LightsOffWhenEmpty()]

        self.history_X = model.history_X()                     # 모델이 본 입력 (이상치 제외)
        meta = raw_df.loc[self.history_X.index, META_COLS].copy()
        meta["Timestamp"] = pd.to_datetime(meta["Timestamp"], format=TIME_FORMAT)
        self.price = meta["cost_per_kwh"]                       # 행마다 실제 단가
        self.meta = meta

        self.building_cols = [c for c in self.history_X.columns if c.startswith(BUILDING_PREFIX)]
        if not self.building_cols:
            raise RuntimeError("No 'Bld_' columns found — one-hot encode 'building_id' first.")
        # 데이터 기간 (일) — 한 달 환산에 사용
        self.n_days = (meta["Timestamp"].max() - meta["Timestamp"].min()) / pd.Timedelta(days=1)

    def buildings(self) -> list[str]:
        return sorted(c[len(BUILDING_PREFIX):] for c in self.building_cols)

    # -----------------------------------------------------------
    # 시나리오 하나 × 빌딩 하나
    # -----------------------------------------------------------
    def evaluate(self, building_id: str, scenario: SavingScenario) -> Recommendation:
        bld_col = f"{BUILDING_PREFIX}{building_id}"
        if bld_col not in self.building_cols:
            raise ValueError(f"Unknown building '{building_id}'. Available: {self.buildings()}")

        # 1) 이 빌딩의 과거 입력
        hist = self.history_X[self.history_X[bld_col] == 1]
        price = self.price.loc[hist.index]

        # 2) 그대로 / 기기를 끈 상태 — 두 번 예측
        base_kwh = self.model.predict(hist)
        changed_X, mask = scenario.apply(hist)
        new_kwh = self.model.predict(changed_X)

        # 3) 차이 = 아낄 수 있는 kWh → 단가를 곱해서 €
        saved_kwh = (base_kwh - new_kwh).clip(lower=0)     # 오히려 늘어나는 경우는 0 으로
        saved_eur = float((saved_kwh * price).sum())
        base_eur = float((base_kwh * price).sum())

        # 4) 한 달 환산
        per_month = saved_eur / self.n_days * DAYS_PER_MONTH
        wasted_hours = int(mask.sum())

        return Recommendation(
            building_id=building_id,
            title=scenario.title,
            message=scenario.message(per_month, wasted_hours),
            wasted_hours=wasted_hours,
            saved_kwh=round(float(saved_kwh.sum()), 2),
            saved_eur=round(saved_eur, 2),
            saved_eur_per_month=round(per_month, 2),
            saved_pct=round(100 * saved_eur / base_eur, 2) if base_eur else 0.0,
        )

    # -----------------------------------------------------------
    # 빌딩 하나의 추천 목록 (절약액 큰 순서, 0 € 인 것은 제외)
    # -----------------------------------------------------------
    def recommend(self, building_id: str) -> list[Recommendation]:
        recs = [self.evaluate(building_id, s) for s in self.scenarios]
        recs = [r for r in recs if r.wasted_hours > 0]
        return sorted(recs, key=lambda r: r.saved_eur_per_month, reverse=True)

    # -----------------------------------------------------------
    # 빌딩 전체
    # -----------------------------------------------------------
    def recommend_all(self) -> dict[str, list[Recommendation]]:
        """{building_id: [Recommendation, ...]} for every building."""
        results = {}
        for building_id in self.buildings():
            results[building_id] = self.recommend(building_id)
            total = sum(r.saved_eur_per_month for r in results[building_id])
            print(f"  {building_id}: {len(results[building_id])} tips, "
                  f"up to {total:.2f} € / month")
        return results