"""보고서·발표용 그림 생성 → figures/*.png

먼저 src/analysis_signal.py, scripts/clinical_lead.py, 공식 backtest(submit/score.json, runs/baseline_score.json)를 실행해 둔다.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "KOWAS-EPI"
FIG = ROOT / "figures"
sys.path.insert(0, str(ROOT / "submit"))
from model import Panel, decompose  # noqa: E402

# 참조 팔레트(dataviz reference) 1~3번 슬롯 + 기준선용 중립 회색
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9890"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"

plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150, "savefig.bbox": "tight",
})


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def fig_signal_sources():
    s = load(ROOT / "runs" / "signal_analysis.json")["v2.5"]
    regions = sorted(s["own"], key=lambda g: s["others"][g])
    y = np.arange(len(regions))
    fig, ax = plt.subplots(figsize=(7, 5.6))
    ax.barh(y + 0.2, [s["others"][g] for g in regions], height=0.36, color=BLUE, label="다른 시·도들의 최근 상승")
    ax.barh(y - 0.2, [s["own"][g] for g in regions], height=0.36, color=ORANGE, label="자기 시·도의 최근 상승")
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_yticks(y, regions)
    ax.set_xlabel("다음 2주 변화와의 상관 (log10 농도, t-2→t 상승 vs t→t+2 변화)")
    ax.set_title("내 지역의 상승은 되돌아가고, 이웃의 상승은 이어진다", loc="left", color=INK, fontsize=12, pad=28)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, borderaxespad=0.2)
    ax.text(0.0, -0.17, f"평균: 자기 {s['own_mean']:+.2f} · 타 지역 {s['others_mean']:+.2f} (17개 시·도 모두 양)",
            transform=ax.transAxes, color=INK2, fontsize=9)
    fig.savefig(FIG / "fig1_신호출처.png"); plt.close(fig)


def fig_noise():
    s = load(ROOT / "runs" / "signal_analysis.json")["v2.5"]["noise_by_nsites"]
    keys = list(s)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(keys, [s[k]["dev_sd"] for k in keys], color=BLUE, width=0.55)
    for b, k in zip(bars, keys):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                f"{s[k]['dev_sd']:.2f}\n(n={s[k]['n']})", ha="center", va="bottom", color=INK2, fontsize=9)
    ax.set_ylim(0, max(v["dev_sd"] for v in s.values()) * 1.3)
    ax.set_xlabel("그 주에 측정한 처리장 수")
    ax.set_ylabel("공통파 대비 지역 편차 크기 (log10 RMS)")
    ax.set_title("처리장이 적은 주일수록 편차가 크다", loc="left", color=INK, fontsize=12)
    fig.savefig(FIG / "fig2_처리장수별_잡음.png"); plt.close(fig)


def fig_window_scores():
    base = {f["fold_id"]: f["score"] for f in load(ROOT / "runs" / "baseline_score.json")["folds"]}
    final = load(ROOT / "submit" / "score.json")
    ours = {f["fold_id"]: f["score"] for f in final["folds"]}
    folds = list(base)
    x = np.arange(len(folds))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(x - 0.19, [base[f] for f in folds], width=0.36, color=GRAY, label=f"공식 기준 모델 (평균 0.2212)")
    ax.bar(x + 0.19, [ours[f] for f in folds], width=0.36, color=BLUE, label=f"파도와 물결 (평균 {final['final']:.4f})")
    ax.set_xticks(x, folds, rotation=0, fontsize=8.5)
    ax.set_ylabel("평가 창 점수")
    ax.set_ylim(0, 0.5)
    ax.set_title("10개 평가 창 모두 기준 모델보다 높다", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.savefig(FIG / "fig3_평가창별_점수.png"); plt.close(fig)


def fig_clinical_lag():
    c = load(ROOT / "runs" / "clinical_lead.json")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for (name, col) in zip(c, (GRAY, BLUE, AQUA)):
        ks = sorted(c[name], key=int)
        ax.plot([int(k) for k in ks], [c[name][k]["r"] for k in ks], marker="o", ms=5, lw=2, color=col, label=name)
    ax.set_xlabel("시차 k (하수 t주 vs 임상 t+k주, 양수 = 하수가 앞섬)")
    ax.set_ylabel("상관계수 r (log10)")
    ax.set_title("공통 유행파는 임상 신고수와 더 강하게 연동한다", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, loc="lower center")
    fig.savefig(FIG / "fig4_임상_선행성.png"); plt.close(fig)


def fig_wave_vs_clinical():
    with open(DATA / "kowas_epi_panel.csv", encoding="utf-8-sig") as h:
        P = Panel(list(csv.DictReader(h)))
    with open(DATA / "national_weekly.csv", encoding="utf-8-sig") as h:
        nat = {r["date_week"]: r for r in csv.DictReader(h)}
    T = len(P.weeks)
    F = np.full(T, np.nan)
    for o in range(8, T):                      # 각 주의 실시간 공통파 (그 주까지의 자료만)
        F[o] = decompose(P.L, o, P.N)[o]
    clin = np.array([np.log10(float(nat[w]["covid_cases_national"])) if nat.get(w, {}).get("covid_cases_national") else np.nan
                     for w in P.weeks])
    z = lambda v: (v - np.nanmean(v)) / np.nanstd(v)
    m = ~np.isnan(clin)
    first = int(np.argmax(m))
    x = np.arange(T)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(x[first:], z(F[first:]), color=BLUE, lw=2, label="공통 유행파 (실시간, 표준화)")
    ax.plot(x[first:], z(clin[first:]), color=ORANGE, lw=2, label="코로나19 표본감시 신고수 (log10, 표준화)")
    ticks = [i for i in range(first, T) if P.weeks[i].endswith(("W01", "W27"))]
    ax.set_xticks(ticks, [P.weeks[i] for i in ticks], fontsize=8.5)
    ax.set_ylabel("표준화 값")
    ax.set_ylim(-2.8, 3.6)
    ax.set_title("하수 공통 유행파와 임상 지표 (2024-W01 이후)", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    fig.savefig(FIG / "fig5_공통파_임상_시계열.png"); plt.close(fig)


def fig_ablation():
    a = load(ROOT / "runs" / "ablation.json")
    names = [k for k in a if k != "full"]
    names.sort(key=lambda k: a[k]["delta"])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = np.arange(len(names))
    ax.barh(y, [a[k]["delta"] for k in names], color=BLUE, height=0.55)
    for i, k in enumerate(names):
        ax.text(a[k]["delta"] - 0.0015, i, f"{a[k]['delta']:+.4f}  (나빠진 창 {a[k]['windows_worse']}/10)",
                va="center", ha="right", color=INK2, fontsize=9)
    ax.set_yticks(y, [a[k]["desc"].split(" (")[0] for k in names])
    ax.yaxis.tick_right()
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlim(min(a[k]["delta"] for k in names) * 1.9, 0.004)
    ax.set_xlabel(f"빼면 바뀌는 최종 정량점수 (최종 모델 {a['full']['final']:.4f})")
    ax.set_title("성능의 대부분은 공통 유행파에서 나온다", loc="left", color=INK, fontsize=12)
    fig.savefig(FIG / "fig6_성분별_제거실험.png"); plt.close(fig)


def fig_coef_consistency():
    p = load(ROOT / "runs" / "fold_params.json")
    folds = list(p["folds"])
    C = np.array([p["folds"][f]["coef"] for f in folds])
    show = [2, 3, 4, 1]                         # 자기 1주 변화, 공통파 추세, 공통파 대비 위치, 현재/기준
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for j, col in zip(show, (ORANGE, BLUE, AQUA, GRAY)):
        ax.plot(range(len(folds)), C[:, j], marker="o", ms=5, lw=2, color=col, label=p["features"][j])
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(range(len(folds)), folds, fontsize=8.5)
    ax.set_ylabel("경보 분류기 계수 (표준화)")
    ax.set_title("10개 창에서 따로 학습해도 부호가 한 번도 바뀌지 않는다", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=9)
    fig.savefig(FIG / "fig7_경보계수_일관성.png"); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    import warnings; warnings.filterwarnings("ignore")
    for f in (fig_signal_sources, fig_noise, fig_window_scores, fig_clinical_lag, fig_wave_vs_clinical,
              fig_ablation, fig_coef_consistency):
        f(); print("saved", f.__name__)
