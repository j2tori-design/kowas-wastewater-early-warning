"""신호 출처 분석: 자기 지역 vs 타 지역 vs 공통파, 처리장 수별 측정 잡음.

python src/analysis_signal.py [--sample-zip 경로]
결과는 runs/signal_analysis.json 에 저장한다.
"""
import argparse
import csv
import io
import json
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "KOWAS-EPI"


def load_panel(text):
    rows = list(csv.DictReader(io.StringIO(text)))
    weeks = sorted({r["date_week"] for r in rows}, key=lambda k: tuple(map(int, k.split("-W"))))
    regions = sorted({r["region"] for r in rows})
    wi = {w: i for i, w in enumerate(weeks)}
    ri = {g: i for i, g in enumerate(regions)}
    L = np.full((len(regions), len(weeks)), np.nan)
    N = np.zeros_like(L)
    for r in rows:
        if r["conc_mean"] != "":
            L[ri[r["region"]], wi[r["date_week"]]] = np.log10(1 + float(r["conc_mean"]))
        N[ri[r["region"]], wi[r["date_week"]]] = float(r["n_sites"] or 0)
    return rows, weeks, regions, L, N


def corr(a, b):
    m = ~np.isnan(a) & ~np.isnan(b)
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def analyse(L, N, regions):
    R, T = L.shape
    past = L[:, 2:] - L[:, :-2]                       # t-2 -> t
    G = np.full_like(L, np.nan); G[:, 2:] = past
    F2 = np.full_like(L, np.nan); F2[:, :-2] = L[:, 2:] - L[:, :-2]   # t -> t+2

    own, others = {}, {}
    for i, g in enumerate(regions):
        own[g] = corr(G[i], F2[i])[0]
        oth = np.nanmean(np.delete(G, i, axis=0), axis=0)
        others[g] = corr(oth, F2[i])[0]

    # 공통파: 8곳 이상 관측된 주의 시·도 평균
    cnt = (~np.isnan(L)).sum(0)
    F = np.where(cnt >= 8, np.nanmean(L, axis=0), np.nan)

    # 지역 편차(직전 8주 평균 오프셋 대비)와 2주 뒤 지속 정도, 처리장 수 구간별
    buckets = {"1곳": (1, 1), "2-3곳": (2, 3), "4-6곳": (4, 6), "7곳+": (7, 99)}
    pairs = {k: [] for k in buckets}
    for i in range(R):
        for t in range(8, T - 2):
            if np.isnan(L[i, t]) or np.isnan(L[i, t + 2]) or np.isnan(F[t]) or np.isnan(F[t + 2]):
                continue
            off = L[i, t - 8:t] - F[t - 8:t]
            off = off[~np.isnan(off)]
            if len(off) < 4:
                continue
            a = off.mean()
            d0, d2 = L[i, t] - F[t] - a, L[i, t + 2] - F[t + 2] - a
            for k, (lo, hi) in buckets.items():
                if lo <= N[i, t] <= hi:
                    pairs[k].append((d0, d2))
    noise = {}
    for k, p in pairs.items():
        p = np.array(p)
        noise[k] = {"n": len(p), "dev_sd": float(np.sqrt((p[:, 0] ** 2).mean())),
                    "persistence": float((p[:, 0] * p[:, 1]).sum() / (p[:, 0] ** 2).sum())}

    dx = L[:, 2:] - L[:, :-2]
    df = (F[2:] - F[:-2])[None, :]
    m = ~np.isnan(dx) & ~np.isnan(df)
    not_common = float((((dx - df) ** 2)[m]).sum() / ((dx ** 2)[m]).sum())

    return {
        "own_mean": float(np.mean(list(own.values()))),
        "others_mean": float(np.mean(list(others.values()))),
        "others_all_positive": all(v > 0 for v in others.values()),
        "own": own, "others": others,
        "noise_by_nsites": noise,
        "share_not_common": not_common,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-zip", default=str(Path.home() / "Downloads" / "260914_KOWAS-EPI_참가자배포.zip"))
    args = ap.parse_args()

    rows, weeks, regions, L, N = load_panel((DATA / "kowas_epi_panel.csv").read_text(encoding="utf-8-sig"))
    out = {"v2.5": analyse(L, N, regions)}

    sample = Path(args.sample_zip)
    if sample.exists():
        with zipfile.ZipFile(sample) as z:
            name = next(n for n in z.namelist() if n.endswith("kowas_epi_panel.csv"))
            srows, _, _, SL, _ = load_panel(z.read(name).decode("utf-8-sig"))
        same = np.array_equal(np.nan_to_num(SL, nan=-1), np.nan_to_num(L, nan=-1))
        changed_cols = sorted({k for a, b in zip(srows, rows) for k in a if a[k] != b[k]})
        out["sample_vs_v2.5"] = {"conc_mean_identical": bool(same), "changed_columns": changed_cols}

    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / "signal_analysis.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    v = out["v2.5"]
    print(json.dumps({k: v[k] for k in ("own_mean", "others_mean", "others_all_positive", "share_not_common", "noise_by_nsites")},
                     ensure_ascii=False, indent=1))
    print(json.dumps(out.get("sample_vs_v2.5"), ensure_ascii=False))


if __name__ == "__main__":
    main()
