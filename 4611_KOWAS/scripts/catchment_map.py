"""처리장 → 집수지역(DATAON-02) 매핑표 만들기. 외부 라이브러리 없이 GeoPackage WKB를 직접 읽어 점-다각형 판정을 한다.

입력: DATAON-02 epsg4326_catchment_ver1.0.gpkg (CC-BY-NC, 국가연구데이터플랫폼 김석현·강원대, DOI 10.22711/idr/1112),
      배포 plant_meta.csv (처리장 위경도, 허용된 정적 파일)
출력: external/plant_catchment.csv (plant_id, region, str_ID, 대·중·소권역 코드·이름, cat_area) — 위경도·처리인구 등 원값은 담지 않음
python scripts/catchment_map.py <gpkg 경로>
"""
import csv
import sqlite3
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_gpkg_geom(b):
    flags = b[3]
    env = (flags >> 1) & 7
    off = 8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env]
    bbox = struct.unpack("<4d", b[8:40]) if env else None      # minx, maxx, miny, maxy
    return bbox, b[off:]


def read_wkb(w, i=0):
    """WKB → 다각형 목록(각 다각형은 링 목록, 링은 (x, y) 목록)."""
    bo = "<" if w[i] == 1 else ">"
    t = struct.unpack(bo + "I", w[i + 1:i + 5])[0] % 1000
    i += 5
    if t == 3:
        (nr,) = struct.unpack(bo + "I", w[i:i + 4]); i += 4
        rings = []
        for _ in range(nr):
            (n,) = struct.unpack(bo + "I", w[i:i + 4]); i += 4
            pts = struct.unpack(bo + f"{2 * n}d", w[i:i + 16 * n]); i += 16 * n
            rings.append(list(zip(pts[0::2], pts[1::2])))
        return [rings], i
    if t == 6:
        (n,) = struct.unpack(bo + "I", w[i:i + 4]); i += 4
        polys = []
        for _ in range(n):
            p, i = read_wkb(w, i)
            polys += p
        return polys, i
    raise ValueError(f"geometry type {t}")


def inside(x, y, ring):
    c = False
    for k in range(len(ring)):
        x1, y1 = ring[k - 1]
        x2, y2 = ring[k]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            c = not c
    return c


def main():
    gpkg = sys.argv[1]
    plants = [p for p in csv.DictReader(open(ROOT / "data" / "KOWAS-EPI" / "plant_meta.csv", encoding="utf-8-sig"))]
    pts = [(p, float(p["longitude"]), float(p["latitude"])) for p in plants if p["latitude"] and p["longitude"]]
    con = sqlite3.connect(f"file:{gpkg}?mode=ro", uri=True)
    rows = con.execute('select geom, str_ID, BBSNCD, BBSNNM, MBSNCD, MBSNNM, SBSNCD, SBSNNM, cat_area '
                       'from "epsg4326_catchment_ver1.0"')
    hit = {}
    for geom, *attr in rows:
        bbox, w = parse_gpkg_geom(geom)
        cand = [(p, x, y) for p, x, y in pts if p["plant_id"] not in hit and
                (bbox is None or (bbox[0] <= x <= bbox[1] and bbox[2] <= y <= bbox[3]))]
        if not cand:
            continue
        polys, _ = read_wkb(w)
        for p, x, y in cand:
            for rings in polys:
                if inside(x, y, rings[0]) and not any(inside(x, y, r) for r in rings[1:]):
                    hit[p["plant_id"]] = attr
                    break
    out = ROOT / "external" / "plant_catchment.csv"
    with open(out, "w", encoding="utf-8", newline="") as h:
        wr = csv.writer(h)
        wr.writerow(["plant_id", "region", "str_ID", "BBSNCD", "BBSNNM", "MBSNCD", "MBSNNM", "SBSNCD", "SBSNNM", "cat_area"])
        for p in plants:
            a = hit.get(p["plant_id"])
            wr.writerow([p["plant_id"], p["region"]] + (list(a) if a else [""] * 8))
    print(f"처리장 {len(plants)}곳 중 좌표 {len(pts)}곳, 집수지역 매칭 {len(hit)}곳 → {out}")


if __name__ == "__main__":
    main()
