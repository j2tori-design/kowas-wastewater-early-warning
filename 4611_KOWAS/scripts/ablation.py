"""성분별 제거 실험: 최종 모델에서 성분을 하나씩 빼고 공식 채점기로 백테스트한다.

각 변형은 submit/model.py 사본의 설정 한 줄만 바꾼다. 결과: runs/ablation.json, runs/ablation/<변형>/
python scripts/ablation.py
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "KOWAS-EPI"
SRC = (ROOT / "submit" / "model.py").read_text(encoding="utf-8")

# (이름, 설명, [(바꿀 원문, 바꿀 내용)])
VARIANTS = [
    ("full", "최종 모델", []),
    ("no_wave", "공통파 제거 (공통파 = 0, 자기 수준·자기 편차만)",
     [("    X = L[:, : o + 1]\n", "    if True:  # 제거 실험: 공통파를 0으로\n        return np.zeros(o + 1)\n    X = L[:, : o + 1]\n")]),
    ("no_trend", "공통파 추세 제거 (phi = 0)", [('"phi": [0.0, 0.25, 0.5, 0.75]', '"phi": [0.0]')]),
    ("no_ripple", "현재 지역 편차 미반영 (rho = 0)", [('"rho": [0.0, 0.2, 0.4, 0.6]', '"rho": [0.0]')]),
    ("no_nsites", "처리장 수 가중 제거 (k = 0)", [('"k": [0.0, 2.0]', '"k": [0.0]')]),
    ("no_logit", "경보 분류기 제거 (예측 ≥ m·1.3·기준)", [('ALERT_MODE = "logit"', 'ALERT_MODE = "ratio"')]),
]


def main():
    out = {}
    for name, desc, edits in VARIANTS:
        src = SRC
        for a, b in edits:
            assert src.count(a) >= 1, (name, a)
            src = src.replace(a, b, 1)
        d = ROOT / "runs" / "ablation" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.py").write_text(src, encoding="utf-8")
        r = subprocess.run([sys.executable, "-W", "ignore", "evaluation.py", "backtest", "--model", str(d / "model.py"),
                            "--out", str(d / "pred.csv"), "--json", str(d / "score.json")],
                           cwd=DATA, env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-2000:]); raise SystemExit(f"{name} 실패")
        s = json.loads((d / "score.json").read_text(encoding="utf-8"))
        folds = s["folds"]
        out[name] = {"desc": desc, "final": s["final"],
                     "min_window": min(f["score"] for f in folds),
                     "smape": sum(f["smape"] for f in folds) / len(folds),
                     "macro_f1": sum(f["macro_f1"] for f in folds) / len(folds),
                     "windows": {f["fold_id"]: f["score"] for f in folds}}
        print(f"{name:10s} {s['final']:.4f}  {desc}", flush=True)
    full = out["full"]["final"]
    for v in out.values():
        v["delta"] = v["final"] - full
        v["windows_worse"] = sum(v["windows"][k] < out["full"]["windows"][k] - 1e-12 for k in v["windows"])
    (ROOT / "runs" / "ablation.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
