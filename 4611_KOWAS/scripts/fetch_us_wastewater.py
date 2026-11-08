"""미국 CDC 하수 감시(NWSS) 표본 자료로 주간 '해외 파도' 지수를 만든다 → external/us_wastewater_weekly.csv

원천: CDC Wastewater Data for SARS-CoV-2 (data.cdc.gov, 데이터셋 j9g8-acpt), Public Domain U.S. Government.
지수: 처리장마다 '이번 주 평균 log10 농도 − 자기 지난주 값'을 구하고, 매주 그 변화의 10% 절사평균을 누적한다.
      처리장 자기 과거와만 비교하므로 미래 자료가 지수에 섞이지 않는다. 주는 KOWAS와 같은 일요일 시작.
네트워크가 필요하므로 모델 실행과는 별개로 한 번만 실행한다(결과 CSV만 실험에 쓴다).
python scripts/fetch_us_wastewater.py
"""
import csv
import io
import urllib.request
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "external" / "us_wastewater_weekly.csv"
API = ("https://data.cdc.gov/resource/j9g8-acpt.csv?$select=site,sample_collect_date,pcr_target_avg_conc_lin"
       "&$where=sample_collect_date>='2023-03-01'&$order=record_id&$limit=50000&$offset={}")


def fetch():
    rows, off = [], 0
    while True:
        with urllib.request.urlopen(API.format(off).replace(" ", "%20").replace("'", "%27"), timeout=300) as r:
            page = list(csv.DictReader(io.StringIO(r.read().decode("utf-8"))))
        rows += page
        print(f"{off + len(page):,}행", flush=True)
        if len(page) < 50000:
            return rows
        off += 50000


def trim_mean(x, cut=0.1):
    x = np.sort(np.asarray(x))
    k = int(len(x) * cut)
    return x[k: len(x) - k].mean()


def main():
    acc = defaultdict(list)                                   # (site, 일요일) -> log10 농도들
    for r in fetch():
        c = r["pcr_target_avg_conc_lin"]
        if not c or float(c) <= 0:
            continue
        d = date.fromisoformat(r["sample_collect_date"][:10])
        sun = d - timedelta(days=(d.weekday() + 1) % 7)
        acc[(r["site"], sun)].append(np.log10(float(c)))
    site_week = {k: float(np.mean(v)) for k, v in acc.items()}
    weeks = sorted({w for _, w in site_week})
    steps = defaultdict(list)
    for (s, w), v in site_week.items():
        prev = site_week.get((s, w - timedelta(days=7)))
        if prev is not None:
            steps[w].append(v - prev)
    idx, out = 0.0, []
    for w in weeks:
        st = steps.get(w, [])
        if len(st) > 20:
            idx += trim_mean(st)
        if w >= date(2023, 3, 5):
            out.append((w.isoformat(), round(idx, 5), len(st)))
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as h:
        wr = csv.writer(h)
        wr.writerow(["week_start_date", "us_ww_log10_index", "n_sites_pairs"])
        wr.writerows(out)
    print("written", OUT, len(out), "주")


if __name__ == "__main__":
    main()
