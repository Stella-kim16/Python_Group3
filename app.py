from flask import Flask, render_template   # 웹서버 + HTML 렌더링
import pandas as pd                        # 표 다루기
from db import engine                      # db.py의 MySQL 연결

app = Flask(__name__)


INPUT_TABLE = "raw_input"           # load_to_mysql.py에서 만든 이전 데이터 테이블
OUTPUT_TABLE = "smartenergy_logs"   # load_to_mysql.py에서 만든 이후 데이터 테이블
KEY_COL = "Timestamp"               # 이전/이후를 같은 행끼리 맞출 기준 열
TARGET_COL = "EnergyConsumption"    # 비교할 값




@app.route("/")
def index():
    # MySQL에서 읽기
    raw_df = pd.read_sql(f"SELECT * FROM {INPUT_TABLE}", engine)      # 이전(input)
    logs_df = pd.read_sql(f"SELECT * FROM {OUTPUT_TABLE}", engine)   # 이후(output)

    # ── 1. 열 목록 ──
    raw_cols = raw_df.columns.tolist()      # input의 열 이름 리스트
    logs_cols = logs_df.columns.tolist()    # output의 열 이름 리스트
    added_cols = [c for c in logs_cols if c not in raw_cols]     # output에만 있는 열 = 새로 추가된 열
    removed_cols = [c for c in raw_cols if c not in logs_cols]   # input에만 있던 열

    # ── 2. EnergyConsumption 비교 ──
    # 두 데이터의 Timestamp 형식이 달라도 맞도록 날짜형으로 통일
    raw_df[KEY_COL] = pd.to_datetime(raw_df[KEY_COL])
    logs_df[KEY_COL] = pd.to_datetime(logs_df[KEY_COL])

    # merge: Timestamp가 같은 행끼리 옆으로 붙임
    # suffixes: 같은 이름의 열에 _before / _after를 붙여 구분
    compare_df = pd.merge(
        raw_df[[KEY_COL, TARGET_COL]],
        logs_df[[KEY_COL, "building_id", TARGET_COL]],
        on=KEY_COL,
        suffixes=("_before", "_after"),
    )
    compare_df["difference"] = (compare_df[f"{TARGET_COL}_after"]
                                - compare_df[f"{TARGET_COL}_before"]).round(4)   # 이후 − 이전


    return render_template(
        "index.html",
        raw_cols=raw_cols,
        logs_cols=logs_cols,
        added_cols=added_cols,
        removed_cols=removed_cols,
        compare_table=compare_df.head(20).to_html(index=False, na_rep="NULL"),
        changed_count=int((compare_df["difference"] != 0).sum()),   # 값이 바뀐 행 수
        matched_count=len(compare_df),                              # Timestamp로 짝지어진 행 수
    )


if __name__ == "__main__":
    app.run(debug=True)