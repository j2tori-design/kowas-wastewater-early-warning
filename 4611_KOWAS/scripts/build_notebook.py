"""notebooks/analysis.ipynb 생성 (실행은 jupyter nbconvert --execute)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

cells = [
    md("# 파도는 함께, 물결은 따로 — KOWAS-EPI 4611 분석·학습·평가\n\n"
       "하수 바이러스 농도를 **전국 공통 유행파(파도) + 지역 편차(물결) + 측정 잡음**으로 나눠 "
       "2주 뒤 농도와 급증 경보를 예측합니다. 이 노트북은 위에서 아래로 실행하면 데이터 확인 → 신호 출처 분석 → "
       "모델 구성·학습 → 공식 채점기 백테스트 → 임상 선행성 → 제거 실험·기상 결합 → 그림·조기경보 화면을 차례로 재현합니다.\n\n"
       "실행 위치: `4611_KOWAS/notebooks/`. 배포 데이터는 `4611_KOWAS/data/KOWAS-EPI/`에 압축을 풀어 둡니다."),
    code("import json, subprocess, sys, os\nfrom pathlib import Path\nimport pandas as pd\n"
         "from IPython.display import Image, display, Markdown\n"
         "ROOT = Path.cwd().parent\nDATA = ROOT / 'data' / 'KOWAS-EPI'\nSUBMIT = ROOT / 'submit'\n"
         "env = dict(os.environ, PYTHONIOENCODING='utf-8')\n"
         "def run(*args, cwd=ROOT):\n"
         "    r = subprocess.run([sys.executable, *args], cwd=cwd, env=env, capture_output=True, text=True, encoding='utf-8')\n"
         "    print((r.stdout + r.stderr).strip()[-3000:])\n"
         "    r.check_returncode()"),
    code("# 실행 환경 기록\nimport platform, numpy, matplotlib\n"
         "pd.Series({'OS': platform.platform(), 'Python': platform.python_version(), 'numpy': numpy.__version__,\n"
         "           'pandas': pd.__version__, 'matplotlib': matplotlib.__version__, 'GPU': '사용하지 않음'})"),
    md("## 1. 데이터 확인\n\n공식 데이터 설명의 수치(3,043행, 관측 2,847행, 결측 196행, 경보 양성 30.6%)와 맞는지 확인합니다."),
    code("panel = pd.read_csv(DATA / 'kowas_epi_panel.csv')\n"
         "summary = {'행': len(panel), '주차': panel.date_week.nunique(), '시·도': panel.region.nunique(),\n"
         "           'conc_mean 관측': int(panel.conc_mean.notna().sum()), 'conc_mean 결측': int(panel.conc_mean.isna().sum()),\n"
         "           '경보 라벨': int(panel.target_alert_t2.notna().sum()), '경보 양성 비율': round(panel.target_alert_t2.mean(), 3)}\n"
         "pd.Series(summary)"),
    md("## 2. 신호의 출처 — 자기 지역 vs 다른 지역 vs 측정 잡음\n\n"
       "- 자기 시·도의 최근 2주 상승은 다음 2주 변화와 **음의 상관**(되돌아감)\n"
       "- 다른 시·도들의 최근 상승은 **17개 시·도 모두 양의 상관**(공통 유행의 신호)\n"
       "- 공통파 대비 지역 편차는 그 주 측정 처리장 수가 적을수록 큼(측정 잡음과 부합하지만 실제 지역 유행과 따로 정량화하지는 못함)"),
    code("run('src/analysis_signal.py')\nsig = json.loads((ROOT / 'runs' / 'signal_analysis.json').read_text(encoding='utf-8'))['v2.5']\n"
         "pd.DataFrame({'자기 시·도': sig['own'], '다른 시·도': sig['others']}).sort_values('다른 시·도', ascending=False).round(3)"),
    code("pd.DataFrame(sig['noise_by_nsites']).T.round(3)"),
    md("## 3. 모델 구성 및 학습 — 한 평가 창 예시\n\n"
       "제출 모델(`submit/model.py`)을 노트북 안에서 직접 학습합니다. 평가 창 **2026-Q3**를 예로, "
       "`evaluation.py`와 같은 규칙(학습 자료 기준 주 이후 행 제외, 이후가 목표주인 라벨 가림)으로 학습 자료를 만들고 "
       "`fit` → `predict` 순서로 실행합니다.\n\n"
       "- `fit`: 학습 구간 마지막 52개 원점으로 롤링 검증해 `w, phi, rho, k`를 고르고, 경보 분류기를 학습합니다.\n"
       "- `predict`: 원점 주차까지의 자료로 공통 유행파를 다시 분해해 2주 뒤 농도와 경보를 냅니다."),
    code("import csv, importlib, warnings\nwarnings.filterwarnings('ignore')\n"
         "sys.path.insert(0, str(SUBMIT))\nimport model as M\nimportlib.reload(M)\n\n"
         "rows = list(csv.DictReader(open(DATA / 'kowas_epi_panel.csv', encoding='utf-8-sig')))\n"
         "fold = next(f for f in csv.DictReader(open(DATA / 'folds.csv', encoding='utf-8-sig')) if f['fold_id'] == '2026-Q3')\n"
         "tu = fold['train_until']\n"
         "train = [dict(r, target_conc_t2='', target_alert_t2='') if r['target_week_t2'] > tu else r\n"
         "         for r in rows if r['date_week'] <= tu]\n"
         "m = M.Model()\nm.fit(train, [])\n"
         "display(Markdown(f\"학습 자료 기준 주 **{tu}** · 학습 행 {len(train):,}개\"))\n"
         "pd.Series({**m.params, '경보 임계값': m.cut, '학습 구간 검증 점수': round(m.cv_score, 4)}, name='fit이 고른 설정')"),
    code("feat = ['예측의 경보선 대비 여유', '현재 농도의 기준 대비 위치', '자기 1주 변화', '공통파 추세', '공통파 대비 위치', '처리장 수(log)']\n"
         "pd.Series(m.logit[2][1:], index=feat, name='경보 분류기 계수 (표준화)').round(3)"),
    code("# 원점 = 학습 자료 기준 주, 목표 = 2주 뒤 (평가 창의 첫 목표주)\n"
         "# 문제 규정 §9-1에 따라 패널 원값(실제 농도·라벨)은 출력하지 않고, 예측과 이번 주 채점 요약만 보인다\n"
         "pred = m.predict(train, tu, [])\n"
         "truth = {r['region']: r for r in rows if r['date_week'] == tu}\n"
         "reg = [(pc, float(truth[g]['target_conc_t2'])) for g, (pc, a) in pred.items() if truth[g]['target_conc_t2']]\n"
         "cls = [(a, int(float(truth[g]['target_alert_t2']))) for g, (pc, a) in pred.items() if truth[g]['target_alert_t2']]\n"
         "smape = 100 / len(reg) * sum(2 * abs(p - y) / (abs(p) + abs(y)) for p, y in reg)\n"
         "display(Markdown(f\"목표 주 {truth['서울']['target_week_t2']} · 회귀 sMAPE **{smape:.1f}%** (n={len(reg)}) · \"\n"
         "                 f\"경보 정답 **{sum(p == y for p, y in cls)}/{len(cls)}**\"))\n"
         "pd.DataFrame([{'시·도': g, '예측 농도 (copies/mL)': round(pc, 1), '예측 경보': a} for g, (pc, a) in pred.items()]).set_index('시·도')"),
    md("## 4. 공식 채점기 백테스트\n\n배포 `evaluation.py`로 평가 창 10개를 창마다 새 프로세스에서 다시 학습·예측합니다. "
       "하이퍼파라미터와 경보 분류기는 각 창의 학습 구간 안에서만 고릅니다."),
    code("run('evaluation.py', 'backtest', '--model', '../../submit/model.py',\n"
         "    '--out', '../../runs/nb_pred.csv', '--json', '../../runs/nb_score.json', cwd=DATA)"),
    code("ours = json.loads((ROOT / 'runs' / 'nb_score.json').read_text(encoding='utf-8'))\n"
         "base = json.loads((ROOT / 'runs' / 'baseline_score.json').read_text(encoding='utf-8'))\n"
         "t = pd.DataFrame(ours['folds']).set_index('fold_id')[['smape', 'smape_baseline', 'macro_f1', 'score']]\n"
         "t['기준 모델 점수'] = pd.DataFrame(base['folds']).set_index('fold_id')['score']\n"
         "display(Markdown(f\"**최종 정량점수 {ours['final']:.4f}** (기준 모델 {base['final']:.4f})\"))\n"
         "t.round(4)"),
    md("## 5. 공통 유행파의 임상 선행성 (사후 검증)\n\n각 주에 그 주까지의 자료만으로 추정한 공통파를 전국 코로나19 표본감시 신고수와 시차별로 비교합니다."),
    code("run('scripts/clinical_lead.py')"),
    md("## 6. 성분별 제거 실험과 평가 창별 학습 계수\n\n"
       "최종 모델에서 성분을 하나씩 빼고 공식 채점기로 다시 백테스트합니다(약 2분). "
       "이어서 평가 창마다 고른 하이퍼파라미터와 경보 분류기 계수를 기록합니다."),
    code("run('scripts/ablation.py')\nrun('scripts/fold_params.py')\n"
         "abl = json.loads((ROOT / 'runs' / 'ablation.json').read_text(encoding='utf-8'))\n"
         "pd.DataFrame({k: {'설명': v['desc'], '점수': round(v['final'], 4), '변화': round(v['delta'], 4),\n"
         "                  '나빠진 창': v['windows_worse']} for k, v in abl.items()}).T"),
    md("### 경보 지표·확률 보정·임상 변화량 상관\n\n"
       "Macro-F1 밖의 경보 지표(정밀도·재현율·급증 기저율), 경보 확률의 Brier score와 보정 곡선, "
       "임상 신고수와의 1주 변화량 상관, 임상 유행 전환점(상승 시작·정점)의 선행 주수, 처리장 수별 되돌림을 계산합니다(약 1분)."),
    code("run('scripts/report_checks.py')\n"
         "rc = json.loads((ROOT / 'runs' / 'report_checks.json').read_text(encoding='utf-8'))\n"
         "cols = ['base_rate', 'precision', 'recall', 'baseline_precision', 'baseline_recall', 'brier', 'brier_ref', 'auc', 'auc_within_week']\n"
         "display(pd.DataFrame(rc['alert']).T[cols].astype(float).round(3))\n"
         "display(pd.DataFrame({k: v['1주 차분'] for k, v in rc['clinical_diff'].items()}).T)\n"
         "display(pd.DataFrame(rc['wave_timing']['events']).set_index('임상 주차'))\n"
         "pd.DataFrame(rc['own_reversion_by_sites']).T"),
    md("### 심사 피드백 추가 검증 — 지역 순위·당시 정보 경보 규칙·블록 재표집\n\n"
       "주별 확률 상위 5곳의 정밀도·포착률, 미래 정보 없이 고정한 경보 규칙의 임상 전환점 대비 선행·미탐·헛신호, "
       "평가 창 안 목표주 블록 재표집으로 본 점수 차이의 95% 구간을 계산합니다(약 2분). 모두 사후 분석이며 제출 예측에는 쓰지 않습니다."),
    code("run('scripts/review_checks.py')\n"
         "rv = json.loads((ROOT / 'runs' / 'review_checks.json').read_text(encoding='utf-8'))\n"
         "display(pd.DataFrame({k: v for k, v in rv['ranking'].items() if isinstance(v, dict)}).T)\n"
         "display(pd.DataFrame({(s, k): v for s, d in rv['clinical_rule']['summary'].items() for k, v in d.items()}).T)\n"
         "pd.DataFrame(rv['bootstrap']['diff']).T"),
    md("## 7. ASOS 기상 결합 실험\n\n"
       "처리장마다 가장 가까운 ASOS 관측소로 시·도별 주간 강수·기온을 만들어(기상은 `aux` 인자로만 받음) "
       "희석 보정과 경보 입력 두 가설을 공식 채점기로 시험합니다(약 2분). 이 시·도 집계 설정에서 효과가 없어 제출 모델에서는 꺼 두었습니다. 유역 단위 효과는 검증하지 않았습니다."),
    code("run('scripts/weather_experiment.py')\n"
         "wx = json.loads((ROOT / 'runs' / 'weather.json').read_text(encoding='utf-8'))\n"
         "pd.DataFrame({k: {'가설': v['desc'], '점수': round(v['final'], 4), '변화': round(v['delta'], 4),\n"
         "                  '좋아진 창': v['windows_better']} for k, v in wx.items()}).T"),
    md("## 8. 외부자료 실험 — 바다 건너 오는 파도\n\n"
       "미국 CDC 하수 감시(NWSS) 표본으로 만든 주간 지수(`external/us_wastewater_weekly.csv`, `scripts/fetch_us_wastewater.py`)의 "
       "최근 2주 변화가 한국 공통파의 다음 2주 변화를 앞서는지 봅니다. 원점보다 1주 앞선 값까지만 씁니다. "
       "공식 채점기 백테스트(`scripts/external_experiment.py`, 7개 설정, 약 30분)는 미리 실행한 결과 `runs/external.json`을 읽습니다. "
       "탐색적 연관성은 있지만 발표 지연 가정에 민감해(같은 설정 2주 지연 시 하락) 제출 모델에는 넣지 않았습니다."),
    code("run('scripts/external_lead.py')\n"
         "ext = json.loads((ROOT / 'runs' / 'external.json').read_text(encoding='utf-8'))\n"
         "pd.DataFrame({k: {'설정': v['desc'], '점수': round(v['final'], 4), '변화': round(v['delta'], 4),\n"
         "                  '최저 창': round(v['min_window'], 4)} for k, v in ext.items()}).T"),
    md("## 9. 그림과 조기경보 화면"),
    code("run('scripts/make_figures.py')\nrun('scripts/early_warning.py')\n"
         "for f in sorted((ROOT / 'figures').glob('fig*.png')):\n    display(Image(filename=str(f), width=720))"),
    md("조기경보 화면은 `figures/조기경보_프로토타입.html`을 브라우저로 엽니다."),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
out = ROOT / "notebooks" / "analysis.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("written", out)
