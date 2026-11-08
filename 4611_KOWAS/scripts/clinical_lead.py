"""공통 유행파의 임상 선행성 사후 검증.

각 주 t에 그 주까지의 자료만으로 공통파 F_t(실시간 값)와 2주 앞 예측값을 만들고,
전국 코로나19 표본감시 신고수(log10)와 시차 k별 상관을 구한다. 비교 대상은 KOWAS 전국 집계 농도.
결과: runs/clinical_lead.json
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "KOWAS-EPI"
sys.path.insert(0, str(ROOT / "submit"))
from model import Panel, decompose, _last  # noqa: E402

LAGS = range(-2, 5)


def read(name):
    with open(DATA / name, encoding="utf-8-sig") as h:
        return list(csv.DictReader(h))


def lag_corr(x, y, k):
    """x[t] 와 y[t+k] 의 상관 (둘 다 있는 주만)."""
    a, b = [], []
    for t in range(len(x)):
        if 0 <= t + k < len(y) and not np.isnan(x[t]) and not np.isnan(y[t + k]):
            a.append(x[t]); b.append(y[t + k])
    return (float(np.corrcoef(a, b)[0, 1]), len(a)) if len(a) > 10 else (np.nan, len(a))


def main():
    P = Panel(read("kowas_epi_panel.csv"))
    nat = {r["date_week"]: r for r in read("national_weekly.csv")}
    T = len(P.weeks)

    realtime, ahead = np.full(T, np.nan), np.full(T, np.nan)
    for o in range(8, T):
        F = decompose(P.L, o, P.N)
        if np.isnan(F[o]):
            continue
        realtime[o] = F[o]
        j2, F2 = _last(F, o - 2)
        slope = (F[o] - F2) / (o - j2) if j2 is not None else 0.0
        ahead[o] = F[o] + 0.5 * 2 * slope          # 2주 앞 공통파 (phi = 0.5)

    def logcol(col):
        return np.array([np.log10(float(nat[w][col])) if w in nat and nat[w][col] not in ("", "0") else np.nan
                         for w in P.weeks])

    clinical = logcol("covid_cases_national")
    kowas = logcol("conc_mean_national")

    out = {}
    for name, x in [("KOWAS 전국 집계 농도", kowas), ("공통파 실시간 F_t", realtime), ("공통파 2주 앞 예측", ahead)]:
        out[name] = {str(k): dict(zip(("r", "n"), lag_corr(x, clinical, k))) for k in LAGS}
    # 2주 앞 예측은 t+2주 하수를 가리키므로, 임상 t+k와의 상관에서 k가 곧 '발표 시점 기준 선행 주수'
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / "clinical_lead.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for name, d in out.items():
        best = max(d, key=lambda k: d[k]["r"] if d[k]["r"] == d[k]["r"] else -9)
        print(f"{name:14s} " + "  ".join(f"k={k}:{d[k]['r']:.3f}" for k in d) + f"   최대 k={best}")


if __name__ == "__main__":
    main()
