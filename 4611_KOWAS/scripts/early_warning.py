"""조기경보 화면 프로토타입 생성 → figures/조기경보_프로토타입.html

두 가지 보기를 한 파일에 담는다. 실시간 자료 연결은 없다.
1) 최신 스냅샷: 배포 패널 전체로 Model을 학습한 뒤 마지막 주(원점)에서 계산한 시·도별 2주 뒤 경보 확률.
   목표 주의 실제 값은 아직 없다.
2) 과거 평가 주 보기: 공식 백테스트(평가 창 10개)에서 그 시점의 자료만으로 낸 저장된 예측
   (submit/predictions.csv 의 예측 농도·경보, runs/alert_probs.csv 의 경보 확률)을 목표 주별로 고른다.
화면 아래 한계 수치는 runs/report_checks.json·runs/review_checks.json 에서 읽는다(먼저 두 스크립트를 실행).
python scripts/early_warning.py [--origin 2026-W36]
"""
import argparse
import csv
import html
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "KOWAS-EPI"
sys.path.insert(0, str(ROOT / "submit"))
import model as M  # noqa: E402

PHASE_TH = 0.08   # 공통파 2주 변화(log10)가 이 값을 넘으면 상승·하강으로 본다
TOP = 5


def read(path):
    with open(path, encoding="utf-8-sig") as h:
        return list(csv.DictReader(h))


def shift(k, n):
    """evaluation.py 의 shift 와 같은 주차 이동."""
    y, w = map(int, k.split("-W"))
    i = (date.fromisocalendar(y, w, 1) + timedelta(weeks=n)).isocalendar()
    return f"{i[0]}-W{i[1]:02d}"


def history():
    """평가 창 목표 주별 저장 예측 (모델 출력만, 실제 결과는 담지 않음)."""
    conc = {(r["region"], r["date_week"]): float(r["pred_conc"]) for r in read(ROOT / "submit" / "predictions.csv")}
    out = {}
    for r in read(ROOT / "runs" / "alert_probs.csv"):
        out.setdefault(r["date_week"], {"fold": r["fold_id"], "cut": float(r["threshold"]), "regions": [],
                                         "origin": shift(r["date_week"], -2)})
        out[r["date_week"]]["regions"].append({"region": r["region"], "prob": float(r["alert_prob"]),
                                               "alert": int(r["pred_alert"]),
                                               "pred": conc[(r["region"], r["date_week"])]})
    return dict(sorted(out.items()))


def limits():
    rc = json.loads((ROOT / "runs" / "report_checks.json").read_text(encoding="utf-8"))
    rv = json.loads((ROOT / "runs" / "review_checks.json").read_text(encoding="utf-8"))
    a = rc["alert"]
    rec = sorted((v["recall"], k) for k, v in a.items() if k != "전체")
    rk = rv["ranking"]
    return {"auc_week": a["전체"]["auc_within_week"], "precision": a["전체"]["precision"], "recall": a["전체"]["recall"],
            "low_recall": [{"fold": k, "recall": r} for r, k in rec[:2]],
            "top5_precision": rk["모델 확률"]["top5_precision_mean_weeks"],
            "top5_capture": rk["모델 확률"]["top5_capture_mean_weeks_with_pos"],
            "base_rate": rk["base_rate"], "zero_alert_weeks": rk["모델 경보"]["weeks_zero_alert_with_pos"],
            "weeks": rk["weeks"], "auc_weeks": a["전체"]["n_weeks_auc"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", default=None)
    args = ap.parse_args()

    rows = read(DATA / "kowas_epi_panel.csv")
    origin = args.origin or max(r["date_week"] for r in rows)
    known = [r for r in rows if r["date_week"] <= origin]
    # 원점 이후가 목표주인 라벨은 가린다 (evaluation.py와 같은 규칙)
    known = [dict(r, target_conc_t2="", target_alert_t2="") if r["target_week_t2"] > origin else r for r in known]

    model = M.Model()
    model.fit(known, [])
    P = M.Panel(known)
    o = P.weeks.index(origin)
    F = M.decompose(P.L, o, P.N)
    lv = M.forecast_level(P, o, F, model.params)
    prob = M._logit_prob(model.logit, M.alert_features(P, o, F, lv)) if model.logit is not None else None
    preds = model.predict(known, origin, [])
    # 지난주(원점 − 1) 같은 모델의 경보 확률 — 확률 변화(%p)를 함께 보여 준다
    prev_prob = None
    if model.logit is not None and o >= 1:
        Fp = M.decompose(P.L, o - 1, P.N)
        lvp = M.forecast_level(P, o - 1, Fp, model.params)
        prev_prob = M._logit_prob(model.logit, M.alert_features(P, o - 1, Fp, lvp))

    j2, F2 = M._last(F, o - 2)
    d2 = F[o] - F2
    phase = "상승" if d2 > PHASE_TH else "하강" if d2 < -PHASE_TH else "정체"
    target = next(r["target_week_t2"] for r in known if r["date_week"] == origin)

    regions = []
    for i, g in enumerate(P.regions):
        b = P.base[i, o]
        pc, alert = preds[g]
        regions.append({
            "region": g, "pred": pc,
            "ratio": pc / b if b == b and b > 0 else None,
            "prob": float(prob[i]) if prob is not None and lv[i] == lv[i] else None,
            "prev": float(prev_prob[i]) if prev_prob is not None and not np.isnan(prev_prob[i]) else None,
            "alert": alert, "obs": ("관측 없음" if P.N[i, o] == 0 else "1~2곳" if P.N[i, o] <= 2 else "3~6곳" if P.N[i, o] <= 6 else "7곳 이상"),
        })
    for d in regions:
        d["delta"] = d["prob"] - d["prev"] if d["prob"] is not None and d["prev"] is not None else None

    wave = [{"week": P.weeks[j], "v": None if np.isnan(F[j]) else round(float(F[j]), 4)}
            for j in range(max(0, o - 51), o + 1)]
    payload = {"origin": origin, "target": target, "phase": phase, "d2": round(float(d2), 3),
               "cut": getattr(model, "cut", None), "params": model.params, "regions": regions, "wave": wave,
               "generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "history": history(), "limits": limits()}
    out = ROOT / "figures" / "조기경보_프로토타입.html"
    out.write_text(TEMPLATE.replace("__DATA__", html.escape(json.dumps(payload, ensure_ascii=False), quote=False)),
                   encoding="utf-8")
    n_alert = sum(r["alert"] for r in regions)
    print(f"원점 {origin} → 목표 {target} · 공통파 {phase}({d2:+.3f}) · 경보 {n_alert}/17 · 과거 평가 주 {len(payload['history'])}개 → {out}")


TEMPLATE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>하수 조기경보 2주 앞</title>
<style>
:root{--bg:#f6f7f7;--surface:#fff;--ink:#14212a;--ink2:#56636b;--line:#dde3e6;--wave:#2a78d6;
--alert:#c2410c;--alert-soft:#fde8dc;--calm:#15803d;--calm-soft:#e1f2e6;--mid:#a16207;--mid-soft:#fbf0d6;
--f-body:"Malgun Gothic","Apple SD Gothic Neo",system-ui,sans-serif;--f-mono:ui-monospace,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#11181c;--surface:#182227;--ink:#e4ecef;--ink2:#9aabb3;
--line:#2b3940;--wave:#3987e5;--alert:#f08a4b;--alert-soft:#3d2416;--calm:#4ade80;--calm-soft:#153222;--mid:#e5b84a;--mid-soft:#382c12;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#11181c;--surface:#182227;--ink:#e4ecef;--ink2:#9aabb3;--line:#2b3940;--wave:#3987e5;
--alert:#f08a4b;--alert-soft:#3d2416;--calm:#4ade80;--calm-soft:#153222;--mid:#e5b84a;--mid-soft:#382c12;color-scheme:dark}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--f-body);font-size:15px;line-height:1.55}
.wrap{max-width:980px;margin:0 auto;padding-block:28px 48px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));display:grid;gap:20px}
h1{font-size:1.5rem;margin:0}h2{font-size:1.05rem;margin:0 0 10px}
.muted{color:var(--ink2)}.mono{font-family:var(--f-mono);font-variant-numeric:tabular-nums}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:14px 16px;display:grid;gap:4px}
.tile .k{font-size:.78rem;color:var(--ink2);letter-spacing:.04em}.tile .v{font-size:1.6rem;font-weight:700}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 18px;min-width:0}
.tbl{overflow-x:auto}table{border-collapse:collapse;width:100%;min-width:620px;font-size:.92rem}
th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left}th{font-size:.75rem;color:var(--ink2);font-weight:600}
td.n,th.n{text-align:right;font-family:var(--f-mono);font-variant-numeric:tabular-nums}
.pill{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.8rem;font-weight:600;white-space:nowrap}
.pill.on{background:var(--alert-soft);color:var(--alert)}.pill.off{background:var(--line);color:var(--ink2)}
.bar{height:8px;background:var(--line);border-radius:4px;overflow:hidden;min-width:90px;position:relative}.bar i{display:block;height:100%;background:var(--alert);border-radius:4px}
.bar b{position:absolute;top:-2px;bottom:-2px;width:2px;background:var(--ink)}
svg text{fill:var(--ink2);font-size:11px;font-family:var(--f-body)}
.note{font-size:.82rem;color:var(--ink2)}
.banner{background:var(--alert-soft);color:var(--ink);border-radius:10px;padding:12px 16px;font-size:.92rem}
.meta{background:var(--mid-soft);border-radius:10px;padding:10px 14px;font-size:.88rem}
tr.pri td{background:var(--alert-soft)}.tag{display:inline-block;border:1px solid var(--alert);color:var(--alert);border-radius:6px;font-size:.7rem;padding:0 6px;white-space:nowrap}
.tabs{display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.tabs button{font:inherit;border:1px solid var(--line);background:var(--surface);color:var(--ink);border-radius:8px;padding:6px 12px;cursor:pointer}
.tabs button[aria-pressed="true"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
select{font:inherit;padding:5px 8px;border-radius:8px;border:1px solid var(--line);background:var(--surface);color:var(--ink)}
ul.lim{margin:6px 0 0;padding-left:20px}ul.lim li{margin:3px 0}
details summary{cursor:pointer;color:var(--ink2);font-size:.82rem}
</style></head><body><div class="wrap">
<header><h1>하수 감시 2주 앞 조기경보 (프로토타입)</h1></header>
<div class="tabs" role="group" aria-label="보기 선택">
 <button id="b-now" aria-pressed="true">최신 스냅샷</button>
 <button id="b-hist" aria-pressed="false">과거 평가 주 보기</button>
 <label id="pick" hidden>목표 주 <select id="sel"></select></label>
</div>
<div class="meta" id="meta"></div>
<section class="tiles" id="tiles"></section>
<div class="banner" id="banner" hidden></div>
<section class="card" id="wavecard"><h2>전국 공통 유행파 (최근 52주, 그 주까지 자료로 추정)</h2><div id="chart"></div>
<p class="note">공통 유행파는 시·도별 고유 수준을 뺀 뒤 남는 전국 공통 성분입니다(log10). 점선이 원점 주차입니다.</p></section>
<section class="card"><h2>시·도별 2주 뒤 경보 확률</h2><div class="tbl"><table id="tbl"></table></div>
<p class="note" id="tblnote"></p></section>
<section class="card"><h2>이 화면을 쓸 때 알아야 할 한계</h2><ul class="lim" id="lim"></ul></section>
<footer class="note"><details><summary>모델 내부 설정 보기</summary><p id="params" class="mono"></p></details></footer>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const L=D.limits;
const TOP=5;
(function(){const W=900,H=200,pl=44,pr=12,pt=10,pb=26;const pts=D.wave.filter(p=>p.v!=null);
 const vs=pts.map(p=>p.v),lo=Math.min(...vs),hi=Math.max(...vs),pad=(hi-lo)*.1||.1;
 const x=i=>pl+(W-pl-pr)*i/(D.wave.length-1),y=v=>pt+(H-pt-pb)*(1-(v-lo+pad)/(hi-lo+2*pad));
 let d='';D.wave.forEach((p,i)=>{if(p.v==null){d+=' ';return}d+=(d.trim()&&!d.endsWith(' ')?'L':'M')+x(i).toFixed(1)+','+y(p.v).toFixed(1)});
 const g=[lo,(lo+hi)/2,hi].map(t=>`<line x1="${pl}" x2="${W-pr}" y1="${y(t)}" y2="${y(t)}" stroke="var(--line)"/><text x="${pl-6}" y="${y(t)+4}" text-anchor="end">${t.toFixed(2)}</text>`).join('');
 const lab=D.wave.map((p,i)=>i%13===0||i===D.wave.length-1?`<text x="${x(i)}" y="${H-6}" text-anchor="middle">${p.week}</text>`:'').join('');
 const last=pts[pts.length-1],li=D.wave.indexOf(last);
 const miss=D.wave.filter(p=>p.v==null).length;
 document.getElementById('chart').innerHTML=`<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="전국 공통 유행파 추이">${g}<path d="${d}" fill="none" stroke="var(--wave)" stroke-width="2"/><line x1="${x(li)}" x2="${x(li)}" y1="${pt}" y2="${H-pb}" stroke="var(--ink2)" stroke-dasharray="3 3"/><circle cx="${x(li)}" cy="${y(last.v)}" r="5" fill="var(--wave)" stroke="var(--surface)" stroke-width="2"/>${lab}</svg>`+
  (miss?`<p class="note">관측 시·도가 적어 공통파를 계산하지 않은 주 ${miss}개는 선을 끊어 표시했습니다.</p>`:'');})();
const fmt=n=>n==null?'–':n>=1000?Math.round(n).toLocaleString('ko-KR'):n.toFixed(1);
const pct=p=>p==null?'–':(p*100).toFixed(0)+'%';
const pp=d=>d==null?'–':Math.abs(d)<0.005?'±0%p':(d>0?'▲ ':'▼ ')+Math.abs(d*100).toFixed(0)+'%p';
const weeks=Object.keys(D.history);
const sel=document.getElementById('sel');
sel.innerHTML=weeks.map(w=>`<option value="${w}">${w} (${D.history[w].fold})</option>`).join('');
sel.value=weeks[weeks.length-1];

function render(mode){
 const hist=mode==='hist';
 document.getElementById('b-now').setAttribute('aria-pressed',!hist);
 document.getElementById('b-hist').setAttribute('aria-pressed',hist);
 document.getElementById('pick').hidden=!hist;
 document.getElementById('wavecard').hidden=hist;
 let R,cut,origin,target;
 if(hist){const h=D.history[sel.value];target=sel.value;origin=h.origin;cut=h.cut;R=h.regions.map(r=>({...r}));}
 else{R=D.regions.map(r=>({...r}));cut=D.cut;origin=D.origin;target=D.target;}
 R.sort((a,b)=>(b.prob??-1)-(a.prob??-1)||a.region.localeCompare(b.region,'ko'));
 R.forEach((r,i)=>r.rank=i+1);
 const nA=R.filter(r=>r.alert).length;
 document.getElementById('meta').innerHTML=hist
  ?`<b>과거 평가 주 보기</b> · 원점 ${origin} → 목표 주 ${target} · 평가 창 ${D.history[target].fold}. 공식 백테스트에서 그 시점까지의 자료만으로 낸 <b>저장된 예측</b>입니다. 실제 결과는 원자료 반입 제한 때문에 화면에 넣지 않았습니다. 화면 생성 ${D.generated}.`
  :`<b>최신 스냅샷</b> · 원점 ${origin} → 목표 주 ${target}. 배포 자료(2026-09-13 수집본)의 마지막 주로 한 번 계산한 결과이며, 목표 주의 실제 값은 아직 없습니다. 실시간 자료 연결은 없습니다. 화면 생성 ${D.generated}.`;
 const up=hist?null:R.filter(r=>r.delta!=null).sort((a,b)=>b.delta-a.delta)[0];
 let tiles=`<div class="tile"><span class="k">2주 뒤 급증 경보</span><span class="v">${nA} / ${R.length}</span><span class="muted">시·도 · 경보 기준 확률 ${pct(cut)} 이상</span></div>`+
  `<div class="tile"><span class="k">경보 확률 1위</span><span class="v">${R[0].region}</span><span class="muted mono">${pct(R[0].prob)}</span></div>`;
 if(!hist){const ph={상승:'var(--alert)',하강:'var(--calm)',정체:'var(--mid)'}[D.phase];
  tiles=`<div class="tile"><span class="k">전국 공통 유행파</span><span class="v" style="color:${ph}">${D.phase}</span><span class="muted mono">최근 2주 변화 ${D.d2>0?'+':''}${D.d2.toFixed(3)} (log10)</span></div>`+tiles+
  (up?`<div class="tile"><span class="k">지난주보다 확률이 가장 많이 오름</span><span class="v">${up.region}</span><span class="muted mono">${pp(up.delta)}</span></div>`:'');}
 document.getElementById('tiles').innerHTML=tiles;
 const b=document.getElementById('banner');
 if(nA>=12){b.hidden=false;b.innerHTML=`<b>${nA}곳이 경보입니다.</b> 전국 유행이 함께 오르는 시기에는 경보만으로 지역을 가리기 어렵습니다. 확률 상위 ${TOP}곳은 <b>추가 확인 후보</b>일 뿐이며, 같은 주 안 시·도 순위의 변별력은 약합니다(아래 한계).`;}
 else b.hidden=true;
 const head='<thead><tr><th>순위</th><th>시·도</th><th>경보</th><th>경보 확률 (세로선 = 기준)</th>'+(hist?'':'<th class="n">지난주 대비</th>')+
  '<th class="n">2주 뒤 예측 농도 (copies/mL)</th>'+(hist?'':'<th class="n">최근 4주 평균 대비</th><th>이번 주 관측 규모 (처리장 수 구간)</th>')+'</tr></thead>';
 document.getElementById('tbl').innerHTML=head+'<tbody>'+R.map(r=>{const cand=nA>=12&&r.rank<=TOP;
  return `<tr class="${cand?'pri':''}"><td class="mono">${r.rank}</td><td>${r.region}${cand?' <span class="tag">추가 확인 후보</span>':''}</td>`+
  `<td><span class="pill ${r.alert?'on':'off'}">${r.alert?'경보':'경보 없음'}</span></td>`+
  `<td><div style="display:flex;gap:8px;align-items:center"><div class="bar"><i style="width:${(r.prob||0)*100}%"></i><b style="left:${cut*100}%"></b></div><span class="mono">${pct(r.prob)}</span></div></td>`+
  (hist?'':`<td class="n" style="color:${r.delta>0?'var(--alert)':r.delta<0?'var(--calm)':'var(--ink2)'}">${pp(r.delta)}</td>`)+
  `<td class="n">${fmt(r.pred)}</td>`+
  (hist?'':`<td class="n">${r.ratio==null?'–':'×'+r.ratio.toFixed(2)}</td><td>${r.obs}</td>`)+`</tr>`}).join('')+'</tbody>';
 document.getElementById('tblnote').innerHTML=`경보 = 경보 분류기 확률이 학습 구간에서 정한 기준(${pct(cut)}) 이상. 급증은 2주 뒤 농도가 직전 4개 주 관측 평균의 1.3배 이상인 경우입니다(대회용 정의이며 보건당국의 공식 경보 기준이 아님). `+
  (hist?'':`'관측 규모'는 그 주 측정 처리장 수를 구간으로 보여 주는 참고값이며 예측의 신뢰구간이 아닙니다. 처리장 수가 적은 주의 지역 편차는 모델에서 덜 반영될 수 있습니다. 확률 변화는 지난주 같은 모델 확률과의 차이(%p)입니다.`);
}
document.getElementById('lim').innerHTML=[
 `같은 주 안에서 시·도를 순위 매기는 변별력은 약합니다(급증·비급증 시·도가 함께 있는 평가 주 ${L.auc_weeks}주의 평균 AUC ${L.auc_week.toFixed(3)}, 0.5가 무작위). 확률 상위 5곳 중 실제 급증 비율은 평가 주 ${L.weeks}주 평균 ${pct(L.top5_precision)}로, 무작위 기대치(${pct(L.base_rate)})보다 조금 높은 수준입니다.`,
 `급증이 드문 시기에는 놓치는 경우가 많습니다(재현율 ${L.low_recall.map(x=>x.fold+' '+x.recall.toFixed(2)).join(', ')}). 실제 급증이 있었는데 경보가 하나도 없던 평가 주가 ${L.zero_alert_weeks}주 있었습니다.`,
 `<b>경보가 없다고 안전하다는 뜻은 아닙니다.</b>`,
 `자원 배치의 단독 판단 근거로 쓰지 말고, 임상·현장 정보와 함께 추가 확인할 지역을 고르는 데만 쓰세요.`,
 `모든 수치는 공식 백테스트(2024-Q2~2026-Q3) 안의 사후 평가이며, 앞으로의 성능을 보장하지 않습니다.`
].map(x=>`<li>${x}</li>`).join('');
document.getElementById('params').textContent=`최신 스냅샷 학습 설정 ${JSON.stringify(D.params)}, 경보 기준 확률 ${D.cut}. 모델: 공통 유행파 + 지역 편차 축소 + 로지스틱 경보.`;
document.getElementById('b-now').onclick=()=>render('now');
document.getElementById('b-hist').onclick=()=>render('hist');
sel.onchange=()=>render('hist');
render('now');
</script></body></html>"""

if __name__ == "__main__":
    main()
