"""외부자료 결합 실험: 미국 CDC 하수 감시(NWSS) 지수의 최근 2주 변화를 공통파 예측에 더한다.

가설: 변이 유행은 국경을 넘어 퍼지므로, 해외 하수의 최근 상승은 한국 공통파의 다음 2주 상승을 앞서 알린다.
외부 지수는 원점 주 시작일보다 US_LAG주 앞선 주까지만 쓴다(발표 지연 감안, 원점 이후 값 미사용).
gam(외부 신호 계수)은 다른 하이퍼파라미터와 함께 fit 안 롤링 검증으로 고르며, 0도 후보라서 쓸모없으면 꺼진다.

결과: runs/external.json, runs/external/<변형>/
python scripts/external_experiment.py
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "KOWAS-EPI"
SRC = (ROOT / "submit" / "model.py").read_text(encoding="utf-8")
EXT = ROOT / "external" / "us_wastewater_weekly.csv"

LOADER = '''

US_LAG = {lag}
US_FILE = Path(__file__).with_name("us_wastewater_weekly.csv")


def us_signal(P):
    """주차별 미국 하수 지수의 2주 변화. 원점 주 시작일보다 US_LAG주 앞선 주까지의 값만 쓴다."""
    import datetime as dt
    g = np.zeros(len(P.weeks))
    if not US_FILE.exists():
        return g
    with open(US_FILE, encoding="utf-8") as h:
        idx = {{r["week_start_date"]: float(r["us_ww_log10_index"]) for r in csv.DictReader(h)}}
    for j, w in enumerate(P.weeks):
        d = dt.date.fromisoformat(P.ws[w])
        a = (d - dt.timedelta(days=7 * US_LAG)).isoformat()
        b = (d - dt.timedelta(days=7 * (US_LAG + 2))).isoformat()
        if a in idx and b in idx:
            g[j] = idx[a] - idx[b]
    return g


def _wmean('''

COMMON = [
    ("\ndef _wmean(", None),  # LOADER로 대체
    ('            i, j = ri[r["region"]], wi[r["date_week"]]\n',
     '            i, j = ri[r["region"]], wi[r["date_week"]]\n            self.ws[r["date_week"]] = r["week_start_date"]\n'),
    ("        shape = (len(self.regions), len(self.weeks))\n",
     "        shape = (len(self.regions), len(self.weeks))\n        self.ws = {}\n"),
    ('        out[i] = a + Ft + p["phi"] * 2 * slope + p["rho"] * d_now\n',
     '        out[i] = a + Ft + p["phi"] * 2 * slope + p["rho"] * d_now + p.get("gam", 0.0) * P.us[o]\n'),
    ('    "m": [0.7, 0.8, 0.9, 1.0, 1.1],\n', '    "m": [0.7, 0.8, 0.9, 1.0, 1.1],\n    "gam": {gam},\n'),
    ('        base_keys = list(itertools.product(GRID["w"], GRID["phi"], GRID["rho"], GRID["k"]))\n',
     '        base_keys = list(itertools.product(GRID["w"], GRID["phi"], GRID["rho"], GRID["k"], GRID["gam"]))\n'),
    ('        for w, phi, rho, k in base_keys:\n            p = {"w": w, "phi": phi, "rho": rho, "k": k}\n',
     '        for w, phi, rho, k, gam in base_keys:\n            p = {"w": w, "phi": phi, "rho": rho, "k": k, "gam": gam}\n'),
    ("        P = Panel(rows)\n        self._prep_weather(P, aux, fit=True)\n",
     "        P = Panel(rows)\n        P.us = us_signal(P)\n        self._prep_weather(P, aux, fit=True)\n"),
    ("        P = Panel(rows)\n        self._prep_weather(P, aux, fit=False)\n",
     "        P = Panel(rows)\n        P.us = us_signal(P)\n        self._prep_weather(P, aux, fit=False)\n"),
]
OLS = [
    ("        if best is not None:\n            self.params = best\n",
     "        if best is not None:\n            self.params = best\n        self.params[\"gam\"] = self._fit_gam(P, T)\n"),
    ("    def _fit_alert(self, P, T):\n",
     '''    def _fit_gam(self, P, T):
        """미국 신호 계수: 학습 구간 전체 원점에서 (실제 2주 뒤 log 농도 − gam 없는 예측)의 시·도 평균을
        미국 2주 변화에 원점을 지나는 직선으로 회귀한다. 음수면 0."""
        num = den = 0.0
        p0 = dict(self.params, gam=0.0)
        for o in range(20, T - 2):
            if P.us[o] == 0:
                continue
            lv = forecast_level(P, o, decompose(P.L, o, P.N), p0)
            r = P.L[:, o + 2] - lv
            r = r[~np.isnan(r)]
            if len(r) >= MIN_REGIONS:
                num += r.mean() * P.us[o]; den += P.us[o] ** 2
        return max(0.0, num / den) if den > 0 else 0.0

    def _fit_alert(self, P, T):
'''),
]
ALERT = [("    return X\n\n\ndef _dilution_beta",
          "    X = np.hstack([X, np.full((R, 1), P.us[o])])                       # 미국 하수 2주 변화\n    return X\n\n\ndef _dilution_beta")]

# (이름, 설명, US_LAG, gam 후보, 경보 입력 추가 여부)
VARIANTS = [
    ("us_lag1", "공통파 예측 + 미국 하수 2주 변화 (1주 지연)", 1, "[0.0, 0.3, 0.6, 0.9]", False),
    ("us_lag2", "공통파 예측 + 미국 하수 2주 변화 (2주 지연, 보수적)", 2, "[0.0, 0.3, 0.6, 0.9]", False),
    ("us_lag1_alert", "us_lag1 + 경보 분류기 입력에도 추가", 1, "[0.0, 0.3, 0.6, 0.9]", True),
    ("us_alert_only", "경보 분류기 입력에만 추가 (1주 지연)", 1, "[0.0]", True),
    ("us_ols", "계수를 학습 구간 전체 예측 오차 회귀로 추정 (1주 지연)", 1, "[0.0]", "ols"),
    ("us_ols_lag2", "us_ols, 2주 지연 (보수적)", 2, "[0.0]", "ols"),
    ("us_ols_alert", "us_ols + 경보 분류기 입력에도 추가", 1, "[0.0]", "ols_alert"),
    # us_ols_alert 와 같은 설정에서 지연만 2주로 바꾼 비교 (보고서의 '같은 설정' 비교를 바로잡기 위해 추가)
    ("us_ols_alert_lag2", "us_ols_alert, 2주 지연 (보수적)", 2, "[0.0]", "ols_alert"),
]


def build(lag, gam, alert):
    src = SRC
    extra = (OLS if alert in ("ols", "ols_alert") else []) + (ALERT if alert in (True, "ols_alert") else [])
    for a, b in COMMON + extra:
        b = LOADER.format(lag=lag) if b is None else b.format(gam=gam) if "{gam}" in b else b
        assert src.count(a) == 1, a
        src = src.replace(a, b, 1)
    return src


def main():
    base = json.loads((ROOT / "submit" / "score.json").read_text(encoding="utf-8"))
    bw = {f["fold_id"]: f["score"] for f in base["folds"]}
    out = {"submitted": {"desc": "제출 모델", "final": base["final"], "windows": bw,
                         "min_window": min(bw.values()), "delta": 0.0, "windows_better": 0}}
    names = sys.argv[1:] or [v[0] for v in VARIANTS]
    for name, desc, lag, gam, alert in VARIANTS:
        if name not in names:
            continue
        d = ROOT / "runs" / "external" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.py").write_text(build(lag, gam, alert), encoding="utf-8")
        shutil.copy(EXT, d / EXT.name)
        r = subprocess.run([sys.executable, "-W", "ignore", "evaluation.py", "backtest", "--model", str(d / "model.py"),
                            "--out", str(d / "pred.csv"), "--json", str(d / "score.json")],
                           cwd=DATA, env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-2000:]); raise SystemExit(f"{name} 실패")
        s = json.loads((d / "score.json").read_text(encoding="utf-8"))
        w = {f["fold_id"]: f["score"] for f in s["folds"]}
        out[name] = {"desc": desc, "final": s["final"], "windows": w, "min_window": min(w.values()),
                     "smape": sum(f["smape"] for f in s["folds"]) / 10,
                     "macro_f1": sum(f["macro_f1"] for f in s["folds"]) / 10,
                     "delta": s["final"] - base["final"],
                     "windows_better": sum(w[k] > bw[k] + 1e-12 for k in w)}
        print(f"{name:14s} {s['final']:.4f} ({s['final'] - base['final']:+.4f}, 좋아진 창 {out[name]['windows_better']}/10)  {desc}",
              flush=True)
    p = ROOT / "runs" / "external.json"
    old = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    old.update(out)
    p.write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
