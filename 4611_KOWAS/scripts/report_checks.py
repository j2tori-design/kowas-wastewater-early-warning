"""결과보고서 보강 지표.

1) 경보: 평가 창별 정밀도·재현율·급증 기저율, 확률의 Brier score·보정 곡선·주 안 순위 변별력(AUC)
   - 공식 채점기와 같은 방식(evaluation.run_fold 규칙: known_at, 창마다 새로 학습)으로 제출 모델을 돌리며
     경보 분류기의 확률을 함께 기록한다. 경보(0/1)는 submit/predictions.csv 와 같은지 확인한다.
2) 임상 선행성: 수준 상관과 함께 1주 변화량(차분) 상관을 시차별로 계산한다.
3) 자기 지역 되돌림(−0.27)의 출처: 측정 처리장 수가 적은 주와 많은 주로 나눠 상관을 비교한다.

출력: runs/report_checks.json (파생 지표만), figures/fig9_경보_보정.png
python scripts/report_checks.py
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "KOWAS-EPI"
sys.path.insert(0, str(DATA))
sys.path.insert(0, str(ROOT / "submit"))
sys.path.insert(0, str(ROOT / "scripts"))
import evaluation as E  # noqa: E402
import model as M  # noqa: E402
import make_figures as mf  # noqa: E402


# ---------------- 1) 경보 ----------------
_last_prob = {}
ALL_PROBS = []          # 평가 키 2,159개 전부의 경보 확률 (라벨 없는 키 포함)
_orig_prob = M._logit_prob


def _capture(model, X):
    p = _orig_prob(model, X)
    _last_prob["p"] = p
    return p


M._logit_prob = _capture


def auc(score, y):
    score, y = np.asarray(score, float), np.asarray(y, int)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def prf(pred, y):
    pred, y = np.asarray(pred, int), np.asarray(y, int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else np.nan
    rec = tp / (tp + fn) if tp + fn else np.nan
    return prec, rec


def alert_records():
    panel = E.read_csv("kowas_epi_panel.csv")
    national = E.read_csv("national_weekly.csv")
    sub = {(r["region"], r["date_week"]): int(r["pred_alert"])
           for r in csv.DictReader(open(ROOT / "submit" / "predictions.csv", encoding="utf-8-sig"))}
    base = {(r["region"], r["date_week"]): int(r["pred_alert"])
            for r in csv.DictReader(open(ROOT / "runs" / "baseline_pred.csv", encoding="utf-8-sig"))}
    truth = {(r["region"], r["target_week_t2"]): r["target_alert_t2"] for r in panel}
    recs, mismatch, train_rate = [], 0, {}
    for fold in E.load_folds():
        m = M.Model()
        train = E.known_at(panel, fold["train_until"])
        m.fit(train, E.national_at(national, fold["train_until"]))
        ys = [float(r["target_alert_t2"]) for r in train if r["target_alert_t2"] != ""]
        train_rate[fold["fold_id"]] = float(np.mean(ys))
        for target in E.fold_weeks(fold):
            origin = E.shift(target, -2)
            _last_prob.clear()
            out = m.predict(E.known_at(panel, origin), origin, E.national_at(national, origin))
            prob = _last_prob.get("p")
            for i, (g, (pc, a)) in enumerate(out.items()):
                mismatch += int(sub[(g, target)] != a)          # 라벨 유무와 관계없이 2,159개 키 모두 대조
                ALL_PROBS.append({"fold": fold["fold_id"], "week": target, "region": g, "alert": a, "cut": m.cut,
                                  "prob": float(prob[i]) if prob is not None else np.nan})
                t = truth.get((g, target), "")
                if t == "":
                    continue
                recs.append({"fold": fold["fold_id"], "week": target, "region": g, "y": int(float(t)),
                             "alert": a, "prob": float(prob[i]) if prob is not None else np.nan,
                             "base_alert": base[(g, target)], "cut": m.cut})
    return recs, mismatch, train_rate


def alert_summary(recs, train_rate):
    folds = sorted({r["fold"] for r in recs})
    rows = {}
    for f in folds + ["전체"]:
        R = [r for r in recs if f == "전체" or r["fold"] == f]
        y = np.array([r["y"] for r in R]); a = np.array([r["alert"] for r in R])
        b = np.array([r["base_alert"] for r in R]); p = np.array([r["prob"] for r in R])
        ok = ~np.isnan(p)
        prec, rec = prf(a, y); bprec, brec = prf(b, y)
        ref = (np.array([train_rate[r["fold"]] for r in R]))           # 학습 구간 급증 비율을 상수 확률로 쓴 기준
        brier = float(((p[ok] - y[ok]) ** 2).mean())
        brier_ref = float(((ref[ok] - y[ok]) ** 2).mean())
        weekly = []
        for w in sorted({r["week"] for r in R}):
            W = [r for r in R if r["week"] == w and not np.isnan(r["prob"])]
            v = auc([r["prob"] for r in W], [r["y"] for r in W])
            if not np.isnan(v):
                weekly.append(v)
        rows[f] = {"n": int(len(y)), "base_rate": float(y.mean()), "alert_rate": float(a.mean()),
                   "precision": prec, "recall": rec, "baseline_precision": bprec, "baseline_recall": brec,
                   "brier": brier, "brier_ref": brier_ref, "bss": 1 - brier / brier_ref,
                   "auc": auc(p[ok], y[ok]), "auc_within_week": float(np.mean(weekly)) if weekly else np.nan,
                   "n_weeks_auc": len(weekly)}
    return rows


def calibration(recs, bins=np.linspace(0, 1, 11)):
    p = np.array([r["prob"] for r in recs]); y = np.array([r["y"] for r in recs])
    ok = ~np.isnan(p); p, y = p[ok], y[ok]
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.sum():
            out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                        "mean_prob": float(p[m].mean()), "observed": float(y[m].mean())})
    ece = sum(c["n"] * abs(c["mean_prob"] - c["observed"]) for c in out) / len(p)
    return out, float(ece)


# ---------------- 2) 임상 선행성: 차분 ----------------
def clinical_diff():
    import clinical_lead as C
    P = M.Panel(C.read("kowas_epi_panel.csv"))
    nat = {r["date_week"]: r for r in C.read("national_weekly.csv")}
    T = len(P.weeks)
    rt = np.full(T, np.nan)
    for o in range(8, T):
        F = M.decompose(P.L, o, P.N)
        rt[o] = F[o]

    def logcol(col):
        return np.array([np.log10(float(nat[w][col])) if w in nat and nat[w][col] not in ("", "0") else np.nan
                         for w in P.weeks])

    clin, kow = logcol("covid_cases_national"), logcol("conc_mean_national")
    d = lambda x: np.r_[np.nan, x[1:] - x[:-1]]                        # 1주 변화량
    res = {}
    for name, x in [("KOWAS 전국 집계 농도", kow), ("공통파 실시간 F_t", rt)]:
        res[name] = {"수준": {str(k): round(C.lag_corr(x, clin, k)[0], 3) for k in C.LAGS},
                     "1주 차분": {str(k): round(C.lag_corr(d(x), d(clin), k)[0], 3) for k in C.LAGS},
                     "n_diff": C.lag_corr(d(x), d(clin), 1)[1]}
    return res, P.weeks, {"임상 신고수": clin, "KOWAS 전국 집계 농도": kow, "공통파 실시간 F_t": rt,
                          "공통파 2주 앞 예측": ahead_series(P)}


def ahead_series(P):
    """clinical_lead.py 와 같은 2주 앞 공통파 예측(phi = 0.5). 값은 그 예측을 낸 주(원점)에 둔다."""
    T = len(P.weeks)
    out = np.full(T, np.nan)
    for o in range(8, T):
        F = M.decompose(P.L, o, P.N)
        if np.isnan(F[o]):
            continue
        j2, F2 = M._last(F, o - 2)
        out[o] = F[o] + 0.5 * 2 * ((F[o] - F2) / (o - j2) if j2 is not None else 0.0)
    return out


# ---------------- 2-b) 임상 유행 전환점(상승 시작·정점)을 몇 주 앞서 잡았나 ----------------
HALF, PROM = 6, 0.2      # 전환점 = 앞뒤 6주 안 최솟값(최댓값), 그 전후 10주 안 진폭 0.2 log10(약 1.6배) 이상


def turning_points(x):
    pts = []
    for t in range(len(x)):
        if np.isnan(x[t]):
            continue
        win = x[max(0, t - HALF): t + HALF + 1]
        wide = x[max(0, t - 10): t + 11]
        for kind, hit in (("상승 시작", np.nanmin(win) == x[t] and np.nanmax(wide) - x[t] >= PROM),
                          ("정점", np.nanmax(win) == x[t] and x[t] - np.nanmin(wide) >= PROM)):
            # 같은 값이 이어진 구간 등으로 6주 안에 같은 종류가 또 잡히면 앞의 것만 둔다
            if hit and not any(k == kind and t - s <= HALF for k, s in pts):
                pts.append((kind, t))
    return pts


def wave_timing(weeks, series):
    clin = series["임상 신고수"]
    ok = np.where(~np.isnan(clin))[0]
    first, last = int(ok[0]), int(ok[-1])
    # 임상 자료 시작 직후·끝 직전은 앞뒤 6주를 볼 수 없어 판정에서 뺀다
    events = [(k, t) for k, t in turning_points(clin) if t - first >= HALF and last - t >= HALF]
    out = {"정의": "전환점 = 앞뒤 6주 안 최솟값(상승 시작)·최댓값(정점), 전후 10주 진폭 0.2 log10 이상. "
                   "선행 주수 = 임상 전환점 주 − 하수 계열의 같은 종류 전환점 주(±6주 안 가장 가까운 것), 양수면 하수가 앞섬. "
                   "2주 앞 예측은 예측을 낸 주 기준.",
           "events": []}
    for kind, t in events:
        row = {"종류": kind, "임상 주차": weeks[t]}
        for name in ("KOWAS 전국 집계 농도", "공통파 실시간 F_t", "공통파 2주 앞 예측"):
            cand = [s for k, s in turning_points(series[name]) if k == kind and abs(s - t) <= HALF]
            row[name] = (t - min(cand, key=lambda s: abs(s - t))) if cand else None
        out["events"].append(row)
    for name in ("KOWAS 전국 집계 농도", "공통파 실시간 F_t", "공통파 2주 앞 예측"):
        v = [e[name] for e in out["events"] if e[name] is not None]
        out[name] = {"median_lead": float(np.median(v)) if v else None, "mean_lead": float(np.mean(v)) if v else None,
                     "matched": len(v), "of": len(out["events"])}
    return out


# ---------------- 3) 자기 지역 되돌림과 측정 잡음 ----------------
def own_by_sites():
    P = M.Panel(E.read_csv("kowas_epi_panel.csv"))
    L, N = P.L, P.N
    G = np.full_like(L, np.nan); G[:, 2:] = L[:, 2:] - L[:, :-2]       # 최근 2주 변화 (t-2 → t)
    F2 = np.full_like(L, np.nan); F2[:, :-2] = L[:, 2:] - L[:, :-2]    # 다음 2주 변화 (t → t+2)
    # 공통 움직임을 빼면 남는 '자기 지역만의' 되돌림
    Gc = G - np.nanmean(G, 0); Fc = F2 - np.nanmean(F2, 0)
    out = {}
    for name, (lo, hi) in {"처리장 1~2곳": (1, 2), "처리장 3~6곳": (3, 6), "처리장 7곳 이상": (7, 999)}.items():
        m = (N >= lo) & (N <= hi) & ~np.isnan(G) & ~np.isnan(F2)
        out[name] = {"n": int(m.sum()), "r": round(float(np.corrcoef(G[m], F2[m])[0, 1]), 3),
                     "r_공통제거": round(float(np.corrcoef(Gc[m], Fc[m])[0, 1]), 3)}
    return out


def figure(summary, calib, ece):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9), gridspec_kw={"width_ratios": [1, 1.35]})
    a1.plot([0, 1], [0, 1], color=mf.GRAY, lw=1, ls="--")
    x = [c["mean_prob"] for c in calib]; y = [c["observed"] for c in calib]; n = [c["n"] for c in calib]
    a1.plot(x, y, color=mf.BLUE, lw=1.6)
    a1.scatter(x, y, s=[12 + v / 4 for v in n], color=mf.BLUE, zorder=3)
    a1.set_xlim(0, 1); a1.set_ylim(0, 1)
    a1.set_xlabel("예측 경보 확률 (구간 평균)"); a1.set_ylabel("실제 급증 비율")
    s = summary["전체"]
    a1.set_title(f"보정 곡선 · Brier {s['brier']:.3f} (기저율 {s['brier_ref']:.3f}) · ECE {ece:.3f}",
                 loc="left", fontsize=10, color=mf.INK)
    folds = [f for f in summary if f != "전체"]
    xx = np.arange(len(folds)); w = 0.27
    a2.bar(xx - w, [summary[f]["base_rate"] for f in folds], w, color=mf.GRAY, label="급증 기저율")
    a2.bar(xx, [summary[f]["precision"] for f in folds], w, color=mf.BLUE, label="정밀도")
    a2.bar(xx + w, [summary[f]["recall"] for f in folds], w, color=mf.AQUA, label="재현율")
    a2.set_xticks(xx, [f[2:] for f in folds], fontsize=8)
    a2.set_ylim(0, 1.05); a2.legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    a2.set_title("평가 창별 경보 정밀도·재현율과 급증 기저율", loc="left", fontsize=10, color=mf.INK)
    fig.tight_layout()
    fig.savefig(mf.FIG / "fig9_경보_보정.png")
    plt.close(fig)


def main():
    recs, mismatch, train_rate = alert_records()
    summary = alert_summary(recs, train_rate)
    calib, ece = calibration(recs)
    cdiff, weeks, series = clinical_diff()
    out = {"alert_mismatch_vs_submission": mismatch, "n_alert_rows": len(recs),
           "alert": summary, "calibration": calib, "ece": ece,
           "clinical_diff": cdiff, "wave_timing": wave_timing(weeks, series),
           "own_reversion_by_sites": own_by_sites(),
           "note": "패널 원값 없이 파생 지표만 담음. 기저율 Brier 기준 = 각 창 학습 구간의 급증 비율을 상수 확률로 쓴 경우"}
    (ROOT / "runs" / "report_checks.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1, default=lambda v: None), encoding="utf-8")
    figure(summary, calib, ece)
    s = summary["전체"]
    print(f"경보 불일치 {mismatch} / {len(ALL_PROBS)} 키 (라벨 있는 행 {len(recs)})")
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()})
    print("ECE", round(ece, 3))
    print(json.dumps(out["clinical_diff"], ensure_ascii=False))
    print(json.dumps(out["own_reversion_by_sites"], ensure_ascii=False))
    print(json.dumps(out["wave_timing"], ensure_ascii=False, indent=0))


if __name__ == "__main__":
    main()
