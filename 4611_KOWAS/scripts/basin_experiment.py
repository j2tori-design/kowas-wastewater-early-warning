"""수계 권역 실험 (DATAON-02 집수지역 매핑 활용). 사전에 정한 설정 2개만 시험한다.

1) external/plant_catchment.csv(scripts/catchment_map.py)의 대권역을 WAMIS 수계 5곳(한강·금강·낙동강·영산강섬진강·제주)으로 묶고,
   시·도마다 처리장 수가 가장 많은 수계를 그 시·도의 권역으로 정해 external/region_basin.csv 로 저장한다(시·도 이름 ↔ 권역 이름뿐, 수치 없음).
2) 제출 model.py 사본에 다음을 더해 공식 채점기로 백테스트한다.
   basin       : 2주 뒤 예측 += psi × (같은 권역 다른 시·도들의 현재 편차 평균), psi ∈ {0, 0.2, 0.4} 를 fit 안 롤링 검증으로 선택
   basin_alert : basin + 경보 입력에 '같은 권역 다른 시·도의 공통파 대비 위치 평균' 추가
채택 기준(결과를 보기 전에 고정): 최종 점수 상승, 최저 창 ≥ 0.1952, 좋아진 창 ≥ 6/10.
결과: runs/basin.json, runs/basin/<설정>/
"""
import collections
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "KOWAS-EPI"
SRC = (ROOT / "submit" / "model.py").read_text(encoding="utf-8")
SYSTEM = {"한 강": "한강", "한강서해": "한강", "한강동해": "한강", "안성천": "한강",
          "금 강": "금강", "금강서해": "금강", "삽교천": "금강", "만경동진": "금강",
          "낙동강": "낙동강", "형산강": "낙동강", "태화강": "낙동강", "회야·수영": "낙동강",
          "영산강": "영산강섬진강", "섬진강": "영산강섬진강", "섬진강남해": "영산강섬진강", "제주도": "제주"}

LOADER = '''
_BASIN_FILE = Path(__file__).with_name("region_basin.csv")
BASIN = ({r["region"]: r["basin"] for r in csv.DictReader(open(_BASIN_FILE, encoding="utf-8"))}
         if _BASIN_FILE.exists() else {})
'''
COMMON = [
    ('DEFAULT = {', LOADER + '\nDEFAULT = {'),
    ('"m": [0.7, 0.8, 0.9, 1.0, 1.1],\n}', '"m": [0.7, 0.8, 0.9, 1.0, 1.1],\n    "psi": [0.0, 0.2, 0.4],\n}'),
    ('itertools.product(GRID["w"], GRID["phi"], GRID["rho"], GRID["k"])',
     'itertools.product(GRID["w"], GRID["phi"], GRID["rho"], GRID["k"], GRID["psi"])'),
    ('for w, phi, rho, k in base_keys:\n            p = {"w": w, "phi": phi, "rho": rho, "k": k}',
     'for w, phi, rho, k, psi in base_keys:\n            p = {"w": w, "phi": phi, "rho": rho, "k": k, "psi": psi}'),
    ('    out = np.full(R, np.nan)\n    lo = max(0, o + 1 - p["w"])',
     '    out = np.full(R, np.nan)\n    RAW = np.full(R, np.nan)\n    lo = max(0, o + 1 - p["w"])'),
    ('        out[i] = a + Ft + p["phi"] * 2 * slope + p["rho"] * d_now\n    return out',
     '        out[i] = a + Ft + p["phi"] * 2 * slope + p["rho"] * d_now\n'
     '        if not (np.isnan(P.L[i, o]) or np.isnan(F[o])):\n'
     '            RAW[i] = P.L[i, o] - F[o] - a\n'
     '    g = p.get("psi", 0.0)\n'
     '    if g:\n'
     '        for i in range(R):\n'
     '            js = [j for j in range(R) if j != i and not np.isnan(RAW[j]) and BASIN.get(P.regions[j]) is not None\n'
     '                  and BASIN.get(P.regions[j]) == BASIN.get(P.regions[i])]\n'
     '            if js and not np.isnan(out[i]):\n'
     '                out[i] += g * float(np.mean(RAW[js]))\n'
     '    return out'),
]
ALERT = [
    ('    if WEATHER in ("alert", "both") and getattr(P, "rain", None) is not None:',
     '    gx = np.array([np.mean([X[j, 4] for j in range(R) if j != i and BASIN.get(P.regions[j]) is not None\n'
     '                            and BASIN.get(P.regions[j]) == BASIN.get(P.regions[i])] or [0.0]) for i in range(R)])\n'
     '    X = np.hstack([X, gx[:, None]])\n'
     '    if WEATHER in ("alert", "both") and getattr(P, "rain", None) is not None:'),
]
VARIANTS = [("basin", "수계 권역 편차 항 (psi 선택)", COMMON),
            ("basin_alert", "수계 권역 편차 항 + 경보 입력", COMMON + ALERT)]


def region_basin():
    c = collections.defaultdict(collections.Counter)
    for r in csv.DictReader(open(ROOT / "external" / "plant_catchment.csv", encoding="utf-8")):
        if r["BBSNNM"]:
            c[r["region"]][SYSTEM[r["BBSNNM"]]] += 1
    out = ROOT / "external" / "region_basin.csv"
    with open(out, "w", encoding="utf-8", newline="") as h:
        w = csv.writer(h); w.writerow(["region", "basin"])
        for g in sorted(c):
            w.writerow([g, c[g].most_common(1)[0][0]])
    return out


def main():
    rb = region_basin()
    print(open(rb, encoding="utf-8").read())
    sub = json.loads((ROOT / "submit" / "score.json").read_text(encoding="utf-8"))
    base = {f["fold_id"]: f["score"] for f in sub["folds"]}
    res = {"submitted": {"desc": "제출 모델", "final": sub["final"], "windows": base, "min_window": min(base.values())}}
    names = sys.argv[1:] or [v[0] for v in VARIANTS]
    for name, desc, edits in VARIANTS:
        if name not in names:
            continue
        src = SRC
        for a, b in edits:
            assert src.count(a) == 1, (name, a[:60])
            src = src.replace(a, b)
        d = ROOT / "runs" / "basin" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "model.py").write_text(src, encoding="utf-8")
        shutil.copy(rb, d / "region_basin.csv")
        r = subprocess.run([sys.executable, "-W", "ignore", "evaluation.py", "backtest", "--model", str(d / "model.py"),
                            "--out", str(d / "pred.csv"), "--json", str(d / "score.json")],
                           cwd=DATA, env=dict(os.environ, PYTHONIOENCODING="utf-8"), capture_output=True, text=True, encoding="utf-8")
        if r.returncode:
            print(r.stdout[-2000:], r.stderr[-2000:]); raise SystemExit(name)
        s = json.loads((d / "score.json").read_text(encoding="utf-8"))
        w = {f["fold_id"]: f["score"] for f in s["folds"]}
        res[name] = {"desc": desc, "final": s["final"], "windows": w, "min_window": min(w.values()),
                     "delta": s["final"] - sub["final"], "windows_better": sum(w[k] > base[k] + 1e-12 for k in w)}
        ok = res[name]["delta"] > 0 and res[name]["min_window"] >= 0.1952 and res[name]["windows_better"] >= 6
        res[name]["adopt"] = ok
        print(f"{name}: {s['final']:.4f} ({res[name]['delta']:+.4f}) 최저 창 {res[name]['min_window']:.4f} "
              f"좋아진 창 {res[name]['windows_better']}/10 → {'채택 기준 충족' if ok else '기각'}", flush=True)
    (ROOT / "runs" / "basin.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
