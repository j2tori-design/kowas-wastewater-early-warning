"""데모: 공식 채점기로 2개 평가 창을 빠르게 재학습·예측하고, 최신 주 조기경보 화면을 만든다.

python scripts/demo.py            # 약 10초
"""
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "KOWAS-EPI"
ENV = dict(os.environ, PYTHONIOENCODING="utf-8")
FOLDS = "2025-Q4,2026-Q3"


def step(title, args, cwd):
    print(f"\n=== {title} ===", flush=True)
    subprocess.run([sys.executable, *args], cwd=cwd, env=ENV, check=True)


if __name__ == "__main__":
    step(f"1. 공식 채점기 백테스트 ({FOLDS}) — 창마다 새로 학습", [
        "evaluation.py", "backtest", "--model", str(ROOT / "submit" / "model.py"),
        "--folds", FOLDS, "--out", str(ROOT / "runs" / "demo_pred.csv")], DATA)
    step("2. 최신 주 2주 앞 조기경보", [str(ROOT / "scripts" / "early_warning.py")], ROOT)
    page = ROOT / "figures" / "조기경보_프로토타입.html"
    print(f"\n조기경보 화면: {page}")
    if "--no-open" not in sys.argv:
        webbrowser.open(page.as_uri())
