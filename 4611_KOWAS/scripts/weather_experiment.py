"""ASOS 기상 결합 실험: WEATHER 설정 4가지를 공식 채점기로 백테스트한다.

off(기상 없음)는 최종 모델과 예측이 같아야 한다(aux 인자 추가가 결과를 바꾸지 않는지 확인).
결과: runs/weather.json, runs/weather/<변형>/
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "KOWAS-EPI"
SRC = (ROOT / "submit" / "model.py").read_text(encoding="utf-8")
VARIANTS = {
    "off": "기상 없음 (최종 모델)",
    "dilution": "강수로 설명되는 편차를 일시적 희석으로 보고 2주 뒤로 넘기지 않음",
    "alert": "경보 분류기에 강수 이상치·4주 기온 변화 추가",
    "both": "희석 보정 + 경보 기상 입력",
}


def main():
    assert 'WEATHER = "off"' in SRC
    out = {}
    for name, desc in VARIANTS.items():
        d = ROOT / "runs" / "weather" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.py").write_text(SRC.replace('WEATHER = "off"', f'WEATHER = "{name}"', 1), encoding="utf-8")
        r = subprocess.run([sys.executable, "-W", "ignore", "evaluation.py", "backtest", "--model", str(d / "model.py"),
                            "--out", str(d / "pred.csv"), "--json", str(d / "score.json")],
                           cwd=DATA, env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                           capture_output=True, text=True, encoding="utf-8")
        if r.returncode:
            print(r.stdout[-3000:], r.stderr[-3000:]); raise SystemExit(f"{name} 실패")
        s = json.loads((d / "score.json").read_text(encoding="utf-8"))
        out[name] = {"desc": desc, "final": s["final"], "min_window": min(f["score"] for f in s["folds"]),
                     "smape": sum(f["smape"] for f in s["folds"]) / 10,
                     "macro_f1": sum(f["macro_f1"] for f in s["folds"]) / 10,
                     "windows": {f["fold_id"]: f["score"] for f in s["folds"]},
                     "status": sorted({f["status"] for f in s["folds"]})}
        print(f"{name:9s} {s['final']:.4f}  min {out[name]['min_window']:.4f}  {desc}", flush=True)
    base = out["off"]
    for v in out.values():
        v["delta"] = v["final"] - base["final"]
        v["windows_better"] = sum(v["windows"][k] > base["windows"][k] + 1e-12 for k in v["windows"])
    (ROOT / "runs" / "weather.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
