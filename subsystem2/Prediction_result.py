from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

# 예측 결과를 담는 상자 — 인터페이스에서 쓰기 쉽게 Python 리스트로 저장


@dataclass
class PredictionResult:
    modelName: str                       # 예: "Random Forest"
    timestamps: list[str]                # 각 값의 시각 (문자열)
    predictions: list[float]             # 예측 kWh
    actualValues: list[float | None]     # 실제 kWh (모르면 None)
    createdAt: datetime = field(default_factory=datetime.now)   # 결과를 만든 시각

    @classmethod
    def from_series(cls, modelName: str, predicted: pd.Series,
                    actual: pd.Series | None = None, timestamps=None) -> "PredictionResult":
        """
        predicted  : model.predict(...) 결과
        actual     : 실제값 (없으면 None)
        timestamps : 각 행의 시각 (없으면 predicted 의 index 사용)
        """
        times = predicted.index if timestamps is None else timestamps
        preds = [round(float(v), 3) for v in predicted]
        if actual is None:
            actuals = [None] * len(preds)
        else:
            actuals = [round(float(v), 3) for v in actual.loc[predicted.index]]
        return cls(modelName, [str(t) for t in times], preds, actuals)

    # UML getter
    def getPredictions(self) -> list[float]:
        return self.predictions

    def getActualValues(self) -> list[float | None]:
        return self.actualValues

    def getModelName(self) -> str:
        return self.modelName

    def getTimestamp(self) -> datetime:
        return self.createdAt

    # 6번 피드백: 실측값이 들어오면 앞에서부터 채우기
    def addActualValues(self, values: list[float]) -> None:
        for i, v in enumerate(values):
            self.actualValues[i] = round(float(v), 3)

    # 인터페이스용
    def to_dict(self) -> dict:
        return {
            "modelName": self.modelName,
            "timestamps": self.timestamps,
            "predictions": self.predictions,
            "actualValues": self.actualValues,
        }