"""파도는 함께, 물결은 따로 — 공통 유행파·지역 편차·측정 잡음 분해 모델 (KOWAS-EPI 제출용).

log10 농도를 다음처럼 본다.
    L[i, t] = a_i + F_t + d_{i,t} + e_{i,t}
    F_t     : 전국 공통 유행파(파도) — 시·도 수준(a_i)을 뺀 뒤의 주별 평균
    a_i     : 시·도 수준 — 최근 w주 동안 공통파 대비 평균 위치
    d_{i,t} : 지역 편차(물결) — 2주 뒤에는 일부(rho)만 남는다
    e_{i,t} : 측정 잡음 — 처리장 수(n_sites)가 적을수록 크다

t+2주 예측:  a_i + F_t + phi·(F_t − F_{t−2}) + rho·s(n)·d_{i,t}
    s(n) = n / (n + k)  : 처리장이 적은 주의 편차는 잡음이 많으므로 덜 믿는다
경보:       로지스틱 분류기 확률 ≥ 임계값 (ALERT_MODE = "ratio"이면 예측 농도 ≥ m · 1.3 · conc_base_avg(t))

하이퍼파라미터(w, phi, rho, k, m)는 fit 안에서 학습 구간 마지막 52주를 원점으로 한
롤링 검증(공식 창 점수와 같은 산식)으로 고른다. 평가 창·주차를 가리키는 값은 쓰지 않는다.

WEATHER가 켜져 있으면 evaluation.py가 넘기는 aux(ASOS 일자료, 기준 주 이하)와 허용된 정적 파일
(KOWAS_STATIC_DIR의 plant_meta.csv·asos_stations.csv)로 시·도별 주간 강수·기온을 만든다.
    "dilution": 강수로 설명되는 편차는 일시적인 희석으로 보고 2주 뒤로 넘기지 않는다
    "alert"   : 경보 분류기에 강수 이상치·4주 기온 변화를 더한다
외부 라이브러리는 numpy 하나.
"""
import csv
import itertools
import math
import os
import warnings
from pathlib import Path

import numpy as np

DECOMP_WEEKS = 26          # 공통파 분해에 쓰는 시·도 수준 추정 구간
MIN_REGIONS = 3            # 공통파를 계산할 최소 관측 시·도 수
CV_ORIGINS = 52            # fit 안 롤링 검증에 쓰는 최근 원점 수
WEIGHT_WAVE = False        # True: 공통파를 처리장 수 가중 평균(n/(n+2))으로 구한다
ALERT_MODE = "logit"       # "ratio": 예측 ≥ m·1.3·기준 / "logit": 로지스틱 경보 분류기
WEATHER = "alert"            # "off" / "dilution" / "alert" / "both" — ASOS 기상 결합 방식

GRID = {
    "w": [4, 8, 12],
    "phi": [0.0, 0.25, 0.5, 0.75],
    "rho": [0.0, 0.2, 0.4, 0.6],
    "k": [0.0, 2.0],
    "m": [0.7, 0.8, 0.9, 1.0, 1.1],
}
DEFAULT = {"w": 8, "phi": 0.5, "rho": 0.3, "k": 0.0, "m": 0.9}


def _week_key(k):
    y, w = k.split("-W")
    return int(y), int(w)


def _f(x):
    return float(x) if x != "" else np.nan


class Panel:
    """문자열 행 → 시·도 × 주 행렬."""

    def __init__(self, rows):
        self.weeks = sorted({r["date_week"] for r in rows}, key=_week_key)
        self.regions = sorted({r["region"] for r in rows})
        wi = {w: j for j, w in enumerate(self.weeks)}
        ri = {g: i for i, g in enumerate(self.regions)}
        shape = (len(self.regions), len(self.weeks))
        self.L = np.full(shape, np.nan)        # log10(1 + conc_mean)
        self.N = np.zeros(shape)               # n_sites
        self.base = np.full(shape, np.nan)     # conc_base_avg (경보 기준)
        self.c3 = np.full(shape, np.nan)       # conc_3wk_avg (회귀 기준선, 검증용)
        self.y = np.full(shape, np.nan)        # target_conc_t2 (검증용)
        self.ya = np.full(shape, np.nan)       # target_alert_t2 (검증용)
        for r in rows:
            i, j = ri[r["region"]], wi[r["date_week"]]
            c = _f(r["conc_mean"])
            if not np.isnan(c):
                self.L[i, j] = np.log10(1 + c)
            self.N[i, j] = _f(r["n_sites"]) if r["n_sites"] != "" else 0.0
            self.base[i, j] = _f(r["conc_base_avg"])
            self.c3[i, j] = _f(r["conc_3wk_avg"])
            self.y[i, j] = _f(r.get("target_conc_t2", ""))
            self.ya[i, j] = _f(r.get("target_alert_t2", ""))


def _read_static(name):
    d = os.environ.get("KOWAS_STATIC_DIR")
    if not d or not (Path(d) / name).exists():
        return None
    with open(Path(d) / name, encoding="utf-8-sig") as h:
        return list(csv.DictReader(h))


def _km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (*a, *b))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


def attach_weather(P, aux):
    """P.rain(log1p 주간 강수, 시·도 평균 대비 이상치)·P.temp(주 평균기온)를 붙인다. 자료가 없으면 False.

    시·도 기상 = 그 주까지 채취를 시작한 처리장(first_sample_week ≤ 주)마다 가장 가까운 ASOS 관측소의 평균.
    강수 빈칸은 0으로 가정한다(안내서 §4-6의 집계 방식과 같음).
    """
    plants, stations = _read_static("plant_meta.csv"), _read_static("asos_stations.csv")
    daily = (aux or {}).get("weather_asos_daily")
    if not plants or not stations or not daily:
        return False
    st = [(s["stn_id"], (float(s["latitude"]), float(s["longitude"]))) for s in stations]
    nearest = []                                       # (region, first_sample_week, stn_id)
    for p in plants:
        if p["latitude"] and p["longitude"] and p["first_sample_week"]:
            ll = (float(p["latitude"]), float(p["longitude"]))
            nearest.append((p["region"], p["first_sample_week"], min(st, key=lambda s: _km(ll, s[1]))[0]))
    wk = {}                                            # (stn, week) -> [강수 합, 기온 합, 기온 수]
    for r in daily:
        a = wk.setdefault((r["stn_id"], r["date_week"]), [0.0, 0.0, 0])
        a[0] += float(r["precip_mm"]) if r["precip_mm"] != "" else 0.0
        if r["temp_avg"] != "":
            a[1] += float(r["temp_avg"]); a[2] += 1
    R, T = P.L.shape
    rain, temp = np.full((R, T), np.nan), np.full((R, T), np.nan)
    ri = {g: i for i, g in enumerate(P.regions)}
    for j, w in enumerate(P.weeks):
        acc = {}
        for g, first, s in nearest:
            if g in ri and _week_key(first) <= _week_key(w) and (s, w) in wk:
                acc.setdefault(ri[g], {})[s] = wk[(s, w)]   # 같은 관측소는 한 번만
        for i, ss in acc.items():
            v = list(ss.values())
            rain[i, j] = np.mean([np.log1p(x[0]) for x in v])
            t = [x[1] / x[2] for x in v if x[2] > 0]
            temp[i, j] = np.mean(t) if t else np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        P.rain = rain - np.nanmean(rain, 1, keepdims=True)
    P.temp = temp
    return True


def _wmean(X, W):
    """열 방향 가중 평균 (nan은 제외)."""
    obs = ~np.isnan(X)
    w = np.where(obs, W, 0.0)
    s = w.sum(0)
    return np.where(s > 0, np.nansum(np.where(obs, X, 0.0) * w, 0) / np.where(s > 0, s, 1), np.nan)


def decompose(L, o, N=None):
    """원점 o까지의 자료로 공통파 F[:o+1]을 구한다 (시·도 수준을 뺀 주별 평균)."""
    X = L[:, : o + 1]
    lo = max(0, o + 1 - DECOMP_WEEKS)
    obs = ~np.isnan(X)
    if WEIGHT_WAVE and N is not None:
        Nn = N[:, : o + 1]
        W = Nn / (Nn + 2.0)
    else:
        W = np.ones_like(X)
    enough = obs.sum(0) >= MIN_REGIONS
    # 관측이 없는 시·도·주의 nanmean 경고는 의도된 결측이므로 숨긴다 (값은 아래에서 대체)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        F = np.where(enough, _wmean(X, W), np.nan)
        for _ in range(4):
            a = np.nanmean(X[:, lo:] - F[None, lo:], 1)
            a = np.where(np.isnan(a), np.nanmean(X - F[None, :], 1), a)
            a = np.where(np.isnan(a), 0.0, a)
            F = np.where(enough, _wmean(X - a[:, None], W), np.nan)
    return F


def _last(v, o):
    for j in range(o, -1, -1):
        if not np.isnan(v[j]):
            return j, v[j]
    return None, np.nan


def forecast_level(P, o, F, p):
    """원점 o에서 시·도별 t+2 log10 농도 예측 (관측이 전혀 없으면 nan)."""
    R = P.L.shape[0]
    jF, Ft = _last(F, o)
    if jF is None:
        return np.full(R, np.nan)
    j2, F2 = _last(F, jF - 2)
    slope = (Ft - F2) / (jF - j2) if j2 is not None else 0.0
    out = np.full(R, np.nan)
    lo = max(0, o + 1 - p["w"])
    for i in range(R):
        dev = P.L[i, lo : o + 1] - F[lo : o + 1]
        dev = dev[~np.isnan(dev)]
        if len(dev) == 0:
            dev = P.L[i, : o + 1] - F[: o + 1]
            dev = dev[~np.isnan(dev)]
            if len(dev) == 0:
                continue
        a = dev.mean()
        d_now = 0.0
        if not np.isnan(P.L[i, o]) and not np.isnan(F[o]):
            n = P.N[i, o]
            s = n / (n + p["k"]) if p["k"] > 0 else 1.0
            dil = P.dil[i, o] if getattr(P, "dil", None) is not None else 0.0   # 강수로 설명되는 일시적 희석
            d_now = s * (P.L[i, o] - F[o] - a - dil)
        out[i] = a + Ft + p["phi"] * 2 * slope + p["rho"] * d_now
    return out


def alert_features(P, o, F, lv):
    """경보 분류기 입력 (시·도 × 6). 모두 원점 o까지의 자료로만 만든다."""
    R = P.L.shape[0]
    jF, Ft = _last(F, o)
    j2, F2 = _last(F, (jF if jF is not None else o) - 2)
    slope = (Ft - F2) / (jF - j2) if jF is not None and j2 is not None else 0.0
    X = np.zeros((R, 6))
    for i in range(R):
        b = P.base[i, o]
        lb = np.log10(1 + b) if not np.isnan(b) else np.nan
        cur = P.L[i, o]
        prev = P.L[i, o - 1] if o >= 1 else np.nan
        X[i, 0] = lv[i] - np.log10(1 + 1.3 * b) if not np.isnan(b) else 0.0   # 예측의 경보선 대비 여유
        X[i, 1] = cur - lb if not (np.isnan(cur) or np.isnan(lb)) else 0.0    # 현재 농도의 기준 대비 위치
        X[i, 2] = cur - prev if not (np.isnan(cur) or np.isnan(prev)) else 0.0  # 자기 1주 변화
        X[i, 3] = slope                                                        # 공통파 추세
        X[i, 4] = (cur - F[o]) if not (np.isnan(cur) or np.isnan(F[o])) else 0.0  # 공통파 대비 위치
        X[i, 5] = np.log1p(P.N[i, o])                                          # 처리장 수
    if WEATHER in ("alert", "both") and getattr(P, "rain", None) is not None:
        rain = np.nan_to_num(P.rain[:, o])                                     # 이번 주 강수 이상치
        dt = P.temp[:, o] - P.temp[:, o - 4] if o >= 4 else np.zeros(R)        # 4주 기온 변화
        X = np.hstack([X, rain[:, None], np.nan_to_num(dt)[:, None]])
    return X


def _dilution_beta(P):
    """공통파·시·도 수준을 뺀 편차를 강수 이상치에 회귀한 기울기 (학습 자료 전체, 한 번)."""
    o = P.L.shape[1] - 1
    F = decompose(P.L, o, P.N)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        D = P.L - F[None, :]
        D = D - np.nanmean(D, 1, keepdims=True)
    m = ~np.isnan(D) & ~np.isnan(P.rain)
    den = float((P.rain[m] ** 2).sum())
    return float((D[m] * P.rain[m]).sum() / den) if den > 0 else 0.0


def _fit_logit(X, y, l2=1.0, iters=800, lr=0.3):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        g = Z.T @ (p - y) / len(y) + l2 * np.r_[0, w[1:]] / len(y)
        w -= lr * g
    return mu, sd, w


def _logit_prob(model, X):
    mu, sd, w = model
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    return 1 / (1 + np.exp(-Z @ w))


def _smape(p, y):
    d = np.abs(y) + np.abs(p)
    t = np.where(d == 0, 0.0, 2 * np.abs(p - y) / np.where(d == 0, 1, d))
    return 100 * t.mean()


def _macro_f1(p, y):
    s = []
    for c in (0, 1):
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        s.append(1.0 if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return (s[0] + s[1]) / 2


class Model:
    def __init__(self):
        self.params = dict(DEFAULT)
        self.beta = 0.0

    def _prep_weather(self, P, aux, fit):
        """WEATHER 설정에 따라 P에 기상(rain, temp)과 희석 보정(dil)을 붙인다."""
        P.rain = P.temp = P.dil = None
        if WEATHER == "off" or not attach_weather(P, aux):
            return
        if WEATHER in ("dilution", "both"):
            if fit:
                self.beta = _dilution_beta(P)
            P.dil = self.beta * np.nan_to_num(P.rain)

    def fit(self, rows, national, aux=None):
        P = Panel(rows)
        self._prep_weather(P, aux, fit=True)
        T = len(P.weeks)
        origins = [o for o in range(max(20, T - CV_ORIGINS - 2), T - 2)]
        if len(origins) < 8:
            return
        Fs = {o: decompose(P.L, o, P.N) for o in origins}

        # (w, phi, rho, k) 조합별 회귀 예측을 한 번씩 만든 뒤 m은 경보 단계에서만 바꾼다
        base_keys = list(itertools.product(GRID["w"], GRID["phi"], GRID["rho"], GRID["k"]))
        best, best_score = None, -np.inf
        for w, phi, rho, k in base_keys:
            p = {"w": w, "phi": phi, "rho": rho, "k": k}
            reg_p, reg_y, reg_b, al_c, al_b, al_y = [], [], [], [], [], []
            for o in origins:
                lv = forecast_level(P, o, Fs[o], p)
                pc = np.where(np.isnan(lv), np.nan, 10 ** lv - 1)
                for i in range(len(P.regions)):
                    if np.isnan(pc[i]):
                        continue
                    y, c3 = P.y[i, o], P.c3[i, o]
                    if not np.isnan(y) and not np.isnan(c3):
                        reg_p.append(max(pc[i], 0)); reg_y.append(y); reg_b.append(c3)
                    if not np.isnan(P.ya[i, o]) and not np.isnan(P.base[i, o]):
                        al_c.append(pc[i]); al_b.append(P.base[i, o]); al_y.append(P.ya[i, o])
            if len(reg_y) < 30 or len(al_y) < 30:
                continue
            reg_p, reg_y, reg_b = map(np.array, (reg_p, reg_y, reg_b))
            reg_term = 1 - _smape(reg_p, reg_y) / _smape(reg_b, reg_y)
            al_c, al_b, al_y = map(np.array, (al_c, al_b, al_y))
            for m in GRID["m"]:
                f1 = _macro_f1((al_c >= m * 1.3 * al_b).astype(int), al_y.astype(int))
                score = 0.6 * reg_term + 0.4 * f1
                if score > best_score:
                    best_score, best = score, dict(p, m=m)
        if best is not None:
            self.params = best
        self.cv_score = best_score
        self.logit = None
        if ALERT_MODE == "logit":
            self._fit_alert(P, T)

    def _fit_alert(self, P, T):
        """선택된 회귀 설정으로 학습 구간 전체 원점의 경보 분류기를 학습하고, Macro-F1 최대 임계값을 고른다."""
        Xs, ys = [], []
        for o in range(20, T - 2):
            F = decompose(P.L, o, P.N)
            lv = forecast_level(P, o, F, self.params)
            X = alert_features(P, o, F, lv)
            for i in range(len(P.regions)):
                if not np.isnan(P.ya[i, o]) and not np.isnan(P.base[i, o]) and not np.isnan(lv[i]):
                    Xs.append(X[i]); ys.append(P.ya[i, o])
        if len(ys) < 60 or len(set(ys)) < 2:
            return
        X, y = np.array(Xs), np.array(ys)
        self.logit = _fit_logit(X, y)
        p = _logit_prob(self.logit, X)
        cuts = np.arange(0.15, 0.71, 0.05)
        f1s = [_macro_f1((p >= c).astype(int), y.astype(int)) for c in cuts]
        self.cut = float(cuts[int(np.argmax(f1s))])

    def predict(self, rows, origin_week, national, aux=None):
        P = Panel(rows)
        self._prep_weather(P, aux, fit=False)
        o = P.weeks.index(origin_week)
        F = decompose(P.L, o, P.N)
        lv = forecast_level(P, o, F, self.params)
        prob = None
        if ALERT_MODE == "logit" and getattr(self, "logit", None) is not None:
            prob = _logit_prob(self.logit, alert_features(P, o, F, lv))
        out = {}
        for i, g in enumerate(P.regions):
            if np.isnan(lv[i]):
                # 관측이 전혀 없는 시·도: 공식 기준선과 같은 대체값
                c3 = P.c3[i, : o + 1]
                c3 = c3[~np.isnan(c3)]
                pc = float(c3[-1]) if len(c3) else 0.0
            else:
                pc = float(max(10 ** lv[i] - 1, 0.0))
            b = P.base[i, o]
            if prob is not None and not np.isnan(lv[i]) and not np.isnan(b):
                alert = int(prob[i] >= self.cut)
            else:
                alert = int(not np.isnan(b) and pc >= self.params["m"] * 1.3 * b)
            out[g] = (pc, alert)
        return out
