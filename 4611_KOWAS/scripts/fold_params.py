"""평가 창별로 fit이 고른 하이퍼파라미터와 경보 분류기 계수를 기록한다 → runs/fold_params.json

evaluation.py와 같은 규칙(학습 자료 기준 주 이후 행 제외, 이후가 목표주인 라벨 가림)으로 자료를 만들어 fit만 다시 실행한다.
"""
import csv
import json
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "KOWAS-EPI"
sys.path.insert(0, str(ROOT / "submit"))
import model as M  # noqa: E402

FEATURES = ["예측의 경보선 대비 여유", "현재 농도의 기준 대비 위치", "자기 1주 변화",
            "공통파 추세", "공통파 대비 위치", "처리장 수(log)"]


def main():
    warnings.filterwarnings("ignore")
    with open(DATA / "kowas_epi_panel.csv", encoding="utf-8-sig") as h:
        rows = list(csv.DictReader(h))
    with open(DATA / "folds.csv", encoding="utf-8-sig") as h:
        folds = list(csv.DictReader(h))
    out = {"features": FEATURES, "folds": {}}
    for f in folds:
        tu = f["train_until"]
        known = [dict(r, target_conc_t2="", target_alert_t2="") if r["target_week_t2"] > tu else r
                 for r in rows if r["date_week"] <= tu]
        m = M.Model()
        m.fit(known, [])
        out["folds"][f["fold_id"]] = {
            "params": {k: m.params[k] for k in ("w", "phi", "rho", "k")},
            "cut": m.cut, "coef": [float(c) for c in m.logit[2][1:]],
        }
        print(f["fold_id"], out["folds"][f["fold_id"]]["params"], flush=True)
    (ROOT / "runs" / "fold_params.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
