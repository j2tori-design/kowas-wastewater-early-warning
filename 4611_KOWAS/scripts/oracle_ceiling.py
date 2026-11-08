"""회귀 항의 상한 추정 (사후 진단, 제출 예측과 무관).

미래 전국 공통파 F(t+2)를 '알고 있다'고 가정한 오라클 예측(시·도 수준 a는 최근 8주, rho 0.4)의 평가 창별
sMAPE 개선율을 구한다. 이 모델 구조에서 전국 파도를 완벽히 맞혀도 회귀 항이 어디까지 오를 수 있는지 보는 용도다.
출력: runs/oracle_ceiling.json
python scripts/oracle_ceiling.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "submit"))
sys.path.insert(0, str(ROOT / "data" / "KOWAS-EPI"))
import evaluation as E  # noqa: E402
import model as M  # noqa: E402


def smape(p, y):
    p, y = np.array(p), np.array(y)
    return 100 * np.mean(2 * np.abs(p - y) / (p + y))


def main():
    P = M.Panel(E.read_csv("kowas_epi_panel.csv"))
    wi = {w: j for j, w in enumerate(P.weeks)}
    out = {}
    for f in E.load_folds():
        pr, yy, bb = [], [], []
        for w in E.fold_weeks(f):
            o = wi[E.shift(w, -2)]
            t = o + 2
            F = M.decompose(P.L, t, P.N)              # 오라클: t+2 주까지 본 공통파
            for i in range(P.L.shape[0]):
                y, b = P.y[i, o], P.c3[i, o]
                if np.isnan(y) or np.isnan(b) or np.isnan(F[t]):
                    continue
                dev = P.L[i, o - 7: o + 1] - F[o - 7: o + 1]
                dev = dev[~np.isnan(dev)]
                if not len(dev):
                    continue
                a = dev.mean()
                d = P.L[i, o] - F[o] - a if not np.isnan(P.L[i, o]) else 0.0
                pr.append(10 ** (a + F[t] + 0.4 * d) - 1); yy.append(y); bb.append(b)
        out[f["fold_id"]] = 1 - smape(pr, yy) / smape(bb, yy)
    res = {"oracle_smape_improvement": out, "mean": float(np.mean(list(out.values()))),
           "windows_ge_20pct": int(sum(v >= 0.2 for v in out.values()))}
    (ROOT / "runs" / "oracle_ceiling.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
