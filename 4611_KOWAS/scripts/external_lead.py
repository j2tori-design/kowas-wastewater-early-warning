"""'바다 건너 오는 파도' 분석: 미국 하수 지수의 최근 변화가 한국 공통파의 다음 2주 변화를 앞서는가.

입력: external/us_wastewater_weekly.csv (scripts/fetch_us_wastewater.py), 배포 national_weekly.csv,
      submit/predictions.csv, runs/external.json (scripts/external_experiment.py)
출력: runs/external_lead.json (상관·설명력 등 파생 수치만), figures/fig8_해외파도.png
python scripts/external_lead.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "KOWAS-EPI"
sys.path.insert(0, str(ROOT / "scripts"))
import make_figures as mf  # noqa: E402  (그림 스타일 공유)


def r2(X, y):
    A = np.c_[np.ones(len(X)), X]
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(1 - (y - A @ b).var() / y.var())


def main():
    kr = pd.read_csv(DATA / "national_weekly.csv", parse_dates=["week_start_date"]).set_index("week_start_date")
    kr = np.log10(kr["conc_mean_national"])
    us = pd.read_csv(ROOT / "external" / "us_wastewater_weekly.csv", parse_dates=["week_start_date"])
    us = us.set_index("week_start_date")["us_ww_log10_index"]
    us = us.reindex(pd.date_range(us.index.min(), us.index.max(), freq="7D")).reindex(kr.index)
    y = kr.shift(-2) - kr                                    # 한국 전국 다음 2주 변화
    own = kr - kr.shift(2)
    out = {"lag_corr": {}, "note": "lag = 원점 주보다 몇 주 앞선 미국 값까지 쓰는지(발표 지연)"}
    for L in range(0, 5):
        u = us.shift(L)
        out["lag_corr"][L] = round(float(y.corr(u - u.shift(2))), 3)
    u = us.shift(1)
    X = pd.DataFrame({"own": own, "us": u - u.shift(2), "y": y}).dropna()
    out["r2_own"] = round(r2(X[["own"]].values, X.y.values), 3)
    out["r2_own_us"] = round(r2(X[["own", "us"]].values, X.y.values), 3)
    out["n_weeks"] = len(X)

    # 제출 모델이 놓친 전국 단위 오차와의 상관 (평가 창 원점)
    p = pd.read_csv(ROOT / "submit" / "predictions.csv")
    k = pd.read_csv(DATA / "kowas_epi_panel.csv", usecols=["region", "date_week", "week_start_date", "target_conc_t2"])
    m = p.merge(k, on=["region", "date_week"]).dropna(subset=["target_conc_t2"])
    m["err"] = np.log10(1 + m.target_conc_t2) - np.log10(1 + m.pred_conc)
    e = m.groupby("week_start_date").err.mean()
    e.index = pd.to_datetime(e.index)
    out["model_error_corr_lag1"] = round(float(e.corr((u - u.shift(2)).reindex(e.index))), 3)

    ext = json.loads((ROOT / "runs" / "external.json").read_text(encoding="utf-8"))
    out["backtest"] = {n: {"final": round(v["final"], 4), "delta": round(v["delta"], 4),
                           "min_window": round(v["min_window"], 4)} for n, v in ext.items()}
    (ROOT / "runs" / "external_lead.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "backtest"}, ensure_ascii=False))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.4]})
    lags = list(out["lag_corr"])
    a1.bar(lags, [out["lag_corr"][L] for L in lags], color=[mf.BLUE if L == 1 else mf.GRAY for L in lags])
    for L in lags:
        a1.text(L, out["lag_corr"][L] + 0.01, f"{out['lag_corr'][L]:.2f}", ha="center", color=mf.INK, fontsize=9)
    a1.set_xticks(lags, [f"{L}주" for L in lags])
    a1.set_xlabel("미국 자료 지연 (원점보다 앞선 주)")
    a1.set_ylabel("상관 (한국 다음 2주 변화)")
    a1.set_title("미국 하수 최근 2주 변화와의 상관 (탐색적)", loc="left", fontsize=11, color=mf.INK)
    names = [("us_ols_alert", "회귀 계수 + 경보 입력"), ("us_lag1_alert", "그리드 + 경보 입력"),
             ("us_ols", "회귀 계수"), ("us_lag1", "그리드"), ("us_alert_only", "경보 입력만"),
             ("us_ols_alert_lag2", "회귀 계수 + 경보 입력, 2주 지연"),
             ("us_ols_lag2", "회귀 계수, 2주 지연"), ("us_lag2", "그리드, 2주 지연")]
    d = [ext[n]["delta"] for n, _ in names]
    yy = np.arange(len(names))[::-1]
    a2.barh(yy, d, color=[mf.AQUA if v > 0 else mf.ORANGE for v in d])
    for yv, v in zip(yy, d):
        a2.text(v + (0.0002 if v >= 0 else -0.0002), yv, f"{v:+.4f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9, color=mf.INK)
    a2.axvline(0, color=mf.INK2, lw=0.8)
    a2.set_yticks(yy, [t for _, t in names])
    a2.set_xlim(-0.009, 0.011)
    a2.set_xlabel("제출 모델(0.3005) 대비 최종 정량점수 변화")
    a2.set_title("공식 채점기 백테스트 (탐색 실험) — 제출 모델에는 넣지 않음", loc="left", fontsize=11, color=mf.INK)
    fig.tight_layout()
    fig.savefig(mf.FIG / "fig8_해외파도.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
