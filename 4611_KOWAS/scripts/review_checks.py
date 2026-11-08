"""심사 피드백 추가 검증 (모두 사후 분석이며 제출 예측에는 쓰지 않는다).

A) 지역 순위의 활용성: 평가 창의 주별(목표주) 경보 확률 상위 5곳의 정밀도·포착률, 주별 경보 수·오경보·미탐.
   비교: 제출 모델의 경보(0/1), 공식 기준 모델의 경보(현재 경보 지속), 단순 순위(원점 주 농도 ÷ conc_base_avg).
B) 실제 경보 규칙의 임상 선행성: 미래 정보 없이 매주 판단하는 규칙을 먼저 고정하고(아래 RULE_*),
   공통파 실시간 값과 2주 앞 예측에 똑같이 적용한 뒤, 사후에 정의한 임상 전환점과 맞춘다.
C) 개선 폭의 불확실성: 평가 창마다 목표주 단위(그 주의 17개 시·도를 함께) 블록 재표집 후
   공식 창 점수 산식과 10개 창 평균을 그대로 계산해 두 모델 점수 차의 분포를 구한다.

입력: 배포 패널·전국 집계·folds(외부 입력), submit/predictions.csv, runs/baseline_pred.csv,
      runs/ablation/{no_wave,no_logit}/pred.csv, 제출 모델(경보 확률 계산용)
출력: runs/review_checks.json (파생 지표만), runs/alert_probs.csv (평가 키 2,159개의 경보 확률 — 모델 예측값)
python scripts/review_checks.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import report_checks as RC  # noqa: E402  (공식 규칙으로 경보 확률을 다시 계산하는 함수 재사용)

E, M = RC.E, RC.M
TOPK = 5
RULE_RISE = 0.15          # 경보 규칙: 최근 4주 최솟값보다 0.15 log10(약 1.4배) 이상 높아지면 '상승 신호'
RULE_FALL = 0.15          #            최근 4주 최댓값보다 0.15 log10 이상 낮아지면 '하강 신호'
RULE_WIN, REFRACT = 4, 4  # 비교 구간 4주, 신호 뒤 4주 안의 같은 신호는 같은 사건으로 본다
MATCH = (8, 4)            # 임상 전환점 c 에 대해 [c−8, c+4] 주 안의 첫 신호를 짝지음
N_BOOT, SEED = 2000, 20261106


def read(path):
    with open(path, encoding="utf-8-sig") as h:
        return list(csv.DictReader(h))


def preds(path):
    return {(r["region"], r["date_week"]): (float(r["pred_conc"]), int(r["pred_alert"])) for r in read(path)}


# ---------------- A) 지역 순위 ----------------
def ranking(recs, panel):
    at = {(r["region"], r["date_week"]): r for r in panel}
    weeks = sorted({r["week"] for r in recs})
    rows = []
    for w in weeks:
        R = [r for r in recs if r["week"] == w]
        origin = E.shift(w, -2)
        for r in R:
            o = at[(r["region"], origin)]
            c, b = o["conc_mean"], o["conc_base_avg"]
            r["ratio"] = float(c) / float(b) if c and b else -np.inf      # 관측이 없으면 순위 맨 뒤
        y = np.array([r["y"] for r in R])
        res = {"week": w, "n": len(R), "pos": int(y.sum())}
        for name, key in (("모델 확률", "prob"), ("단순 비율(원점 농도÷기준선)", "ratio")):
            order = sorted(R, key=lambda r: (-r[key], r["region"]))       # 동률은 시·도 이름순
            top = order[:TOPK]
            hit = sum(r["y"] for r in top)
            res[name] = {"hit": hit, "k": len(top)}
        for name, key in (("모델 경보", "alert"), ("기준 모델 경보", "base_alert")):
            a = np.array([r[key] for r in R])
            res[name] = {"alerts": int(a.sum()), "tp": int(((a == 1) & (y == 1)).sum()),
                         "fp": int(((a == 1) & (y == 0)).sum()), "fn": int(((a == 0) & (y == 1)).sum())}
        rows.append(res)

    summ = {"weeks": len(rows), "weeks_with_pos": sum(r["pos"] > 0 for r in rows),
            "pos_total": sum(r["pos"] for r in rows), "rows": sum(r["n"] for r in rows),
            "base_rate": sum(r["pos"] for r in rows) / sum(r["n"] for r in rows)}
    for name in ("모델 확률", "단순 비율(원점 농도÷기준선)"):
        prec = [r[name]["hit"] / r[name]["k"] for r in rows]
        cap = [r[name]["hit"] / r["pos"] for r in rows if r["pos"] > 0]
        summ[name] = {"top5_precision_mean_weeks": float(np.mean(prec)),
                      "top5_precision_pooled": sum(r[name]["hit"] for r in rows) / sum(r[name]["k"] for r in rows),
                      "top5_capture_mean_weeks_with_pos": float(np.mean(cap)),
                      "top5_capture_pooled": sum(r[name]["hit"] for r in rows) / summ["pos_total"]}
    for name in ("모델 경보", "기준 모델 경보"):
        a = [r[name] for r in rows]
        summ[name] = {"alerts_per_week_mean": float(np.mean([x["alerts"] for x in a])),
                      "alerts_per_week_max": int(max(x["alerts"] for x in a)),
                      "false_alarms_total": sum(x["fp"] for x in a), "misses_total": sum(x["fn"] for x in a),
                      "true_alerts_total": sum(x["tp"] for x in a),
                      "weeks_all17_alert": sum(x["alerts"] >= 17 for x in a),
                      "weeks_zero_alert_with_pos": sum(x["alerts"] == 0 and r["pos"] > 0 for x, r in zip(a, rows))}
    return summ, rows


# ---------------- B) 경보 규칙의 임상 선행성 ----------------
def signals(x, kind):
    out, last = [], -99
    for t in range(RULE_WIN, len(x)):
        win = x[t - RULE_WIN: t + 1]
        if np.isnan(x[t]) or np.all(np.isnan(win)):
            continue
        hit = (x[t] - np.nanmin(win) >= RULE_RISE) if kind == "상승 시작" else (np.nanmax(win) - x[t] >= RULE_FALL)
        if hit and t - last > REFRACT:
            out.append(t)
        if hit:
            last = t
    return out


def clinical_rule():
    cdiff, weeks, series = RC.clinical_diff()
    clin = series["임상 신고수"]
    ok = np.where(~np.isnan(clin))[0]
    first, last = int(ok[0]), int(ok[-1])
    events = [(k, t) for k, t in RC.turning_points(clin) if t - first >= RC.HALF and last - t >= RC.HALF]
    out = {"rule": f"상승 신호: x[t] − min(x[t−{RULE_WIN}..t]) ≥ {RULE_RISE} log10, 하강 신호: max(x[t−{RULE_WIN}..t]) − x[t] ≥ {RULE_FALL}. "
                   f"신호 뒤 {REFRACT}주 안 재발은 같은 신호. 임상 전환점 c 와 [c−{MATCH[0]}, c+{MATCH[1]}]주 안 첫 신호를 짝지음(양수 = 앞섬). "
                   "규칙과 기준값은 결과를 보기 전에 한 번 정하고 바꾸지 않음. 임상 전환점은 앞뒤 6주 자료로 정한 사후 정의.",
           "events": [], "summary": {}}
    for name in ("공통파 실시간 F_t", "공통파 2주 앞 예측"):
        x = series[name]
        res = {}
        for kind in ("상승 시작", "정점"):
            sig = [t for t in signals(x, "상승 시작" if kind == "상승 시작" else "정점") if first <= t <= last]
            ev = [t for k, t in events if k == kind]
            used, leads, miss = set(), [], 0
            for c in ev:
                cand = [t for t in sig if c - MATCH[0] <= t <= c + MATCH[1] and t not in used]
                if cand:
                    used.add(cand[0]); leads.append(c - cand[0])
                    out["events"].append({"계열": name, "종류": kind, "임상 주차": weeks[c], "신호 주차": weeks[cand[0]],
                                          "선행 주": c - cand[0]})
                else:
                    miss += 1
                    out["events"].append({"계열": name, "종류": kind, "임상 주차": weeks[c], "신호 주차": None,
                                          "선행 주": None})
            res[kind] = {"clinical_events": len(ev), "signals": len(sig), "matched": len(leads), "missed": miss,
                         "false_signals": len(sig) - len(used),
                         "lead_median": float(np.median(leads)) if leads else None,
                         "lead_mean": float(np.mean(leads)) if leads else None,
                         "lead_min": int(min(leads)) if leads else None, "lead_max": int(max(leads)) if leads else None}
        out["summary"][name] = res
    return out


# ---------------- C) 블록 재표집 ----------------
def term(p, y):
    s = max(p, y)
    return 0.0 if s == 0 else 2 * abs(p - y) / (p + y)


def week_stats(pred, panel, fold):
    """목표주별 합계: 회귀 sMAPE 항(모델·기준선)과 행 수, 경보 혼동 행렬 성분."""
    weeks = E.fold_weeks(fold)
    stats = np.zeros((len(weeks), 9))   # n_reg, sum_term_model, sum_term_base, tp1, fp1, fn1, tp0, fp0, fn0
    for j, w in enumerate(weeks):
        for r in panel:
            if r["target_week_t2"] != w:
                continue
            p, a = pred[(r["region"], w)]
            if r["target_conc_t2"] != "" and r["conc_3wk_avg"] != "":
                y = float(r["target_conc_t2"])
                stats[j, 0] += 1; stats[j, 1] += term(p, y); stats[j, 2] += term(float(r["conc_3wk_avg"]), y)
            if r["target_alert_t2"] != "":
                t = int(r["target_alert_t2"])
                stats[j, 3] += (a == 1) & (t == 1); stats[j, 4] += (a == 1) & (t == 0); stats[j, 5] += (a == 0) & (t == 1)
                stats[j, 6] += (a == 0) & (t == 0); stats[j, 7] += (a == 0) & (t == 1); stats[j, 8] += (a == 1) & (t == 0)
    return stats


def window_score(S):
    """S: (..., 9) 합계 → 공식 창 점수 (Macro-F1 의 '실제·예측 모두 0건이면 1' 규칙 포함)."""
    sm, sb = 100 * S[..., 1] / S[..., 0], 100 * S[..., 2] / S[..., 0]
    f1s = []
    for tp, fp, fn in ((S[..., 3], S[..., 4], S[..., 5]), (S[..., 6], S[..., 7], S[..., 8])):
        den = 2 * tp + fp + fn
        f1s.append(np.where(den == 0, 1.0, 2 * tp / np.where(den == 0, 1, den)))
    return 0.6 * (1 - sm / sb) + 0.4 * (f1s[0] + f1s[1]) / 2


def bootstrap(panel, folds):
    models = {"최종 모델": preds(ROOT / "submit" / "predictions.csv"),
              "공식 기준 모델": preds(ROOT / "runs" / "baseline_pred.csv"),
              "공통파 제거": preds(ROOT / "runs" / "ablation" / "no_wave" / "pred.csv"),
              "비율 경보(경보 분류기 제거)": preds(ROOT / "runs" / "ablation" / "no_logit" / "pred.csv")}
    stats = {m: [week_stats(p, panel, f) for f in folds] for m, p in models.items()}
    point = {m: float(np.mean([window_score(s.sum(0)) for s in stats[m]])) for m in models}
    rng = np.random.default_rng(SEED)
    boot = {m: np.zeros(N_BOOT) for m in models}
    for fi, f in enumerate(folds):
        nw = stats["최종 모델"][fi].shape[0]
        W = rng.multinomial(nw, np.ones(nw) / nw, size=N_BOOT)            # 창 안 목표주 재표집 (모든 모델 공통)
        for m in models:
            boot[m] += window_score(W @ stats[m][fi]) / len(folds)
    out = {"method": f"평가 창마다 목표주를 복원 추출(그 주의 모든 시·도를 함께, {N_BOOT}회, seed {SEED})하고 "
                     "공식 창 점수와 10개 창 단순 평균을 그대로 계산. 모든 모델에 같은 재표집을 적용해 차이를 구함. "
                     "모델 선택에 이미 쓴 평가 창이라 미래 일반화 오차를 대신하지 않음.",
           "point": point, "diff": {}}
    for other in ("공식 기준 모델", "공통파 제거", "비율 경보(경보 분류기 제거)"):
        d = boot["최종 모델"] - boot[other]
        out["diff"][f"최종 모델 − {other}"] = {
            "point": point["최종 모델"] - point[other], "boot_mean": float(d.mean()),
            "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "share_positive": float((d > 0).mean())}
    return out


def main():
    panel = E.read_csv("kowas_epi_panel.csv")
    folds = E.load_folds()
    recs, mismatch, _ = RC.alert_records()
    assert mismatch == 0, "경보 재계산이 제출 예측과 다름"
    with open(ROOT / "runs" / "alert_probs.csv", "w", encoding="utf-8", newline="") as h:
        wr = csv.writer(h)
        wr.writerow(["region", "date_week", "fold_id", "alert_prob", "pred_alert", "threshold"])
        for r in RC.ALL_PROBS:
            wr.writerow([r["region"], r["week"], r["fold"], f"{r['prob']:.4f}", r["alert"], f"{r['cut']:.2f}"])
    rank, _ = ranking(recs, panel)
    out = {"ranking": rank, "clinical_rule": clinical_rule(), "bootstrap": bootstrap(panel, folds),
           "note": "모두 평가 창(공식 백테스트) 안의 사후 분석. 패널 원값은 담지 않음."}
    (ROOT / "runs" / "review_checks.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out["ranking"].items()}, ensure_ascii=False, indent=1))
    print(json.dumps(out["clinical_rule"]["summary"], ensure_ascii=False, indent=1))
    print(json.dumps(out["bootstrap"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
