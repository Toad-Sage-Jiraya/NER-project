import os
import sys
import subprocess
from pathlib import Path

PYTHON_EXE = r"C:\Users\manis\.gemini\antigravity\scratch\v\Scripts\python.exe"
SRC_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SRC_DIR.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

MODELS = ["bilstm", "bilstm_attn", "cnn_bilstm", "cnn_bilstm_attn", "cnn_bigru_attn"]
SEEDS = [1, 7, 42]
TYPES = ["disease"]

def run_cmd(cmd, log_file):
    print(f">> Executing: {' '.join(cmd)}")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n--- COMMAND: {' '.join(cmd)} ---\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(SRC_DIR))
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            f.write(line)
            f.flush()
        proc.wait()
        if proc.returncode != 0:
            raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")

def main():
    for t in TYPES:
        data_dir = PROJECT_ROOT / f"processed_{t}"
        runs_dir = PROJECT_ROOT / f"runs_{t}"
        runs_dir.mkdir(exist_ok=True)
        log_file = LOGS_DIR / f"experiment_{t}.log"

        for model in MODELS:
            for seed in SEEDS:
                run_out = runs_dir / f"{model}_seed{seed}"
                metrics_file = run_out / "metrics_test.json"
                if metrics_file.exists():
                    print(f"[*] Skipping {model} seed {seed} on {t} (already completed: {metrics_file})")
                    continue

                print(f"\n=======================================================")
                print(f"[*] Starting {model} (seed {seed}) on {t.upper()}")
                print(f"=======================================================")
                
                # Train
                train_cmd = [
                    PYTHON_EXE, "train.py",
                    "--data", str(data_dir),
                    "--model", model,
                    "--seed", str(seed),
                    "--epochs", "20",
                    "--out", str(runs_dir)
                ]
                run_cmd(train_cmd, log_file)

                # Evaluate
                eval_cmd = [
                    PYTHON_EXE, "evaluate.py",
                    "--data", str(data_dir),
                    "--run", str(run_out)
                ]
                run_cmd(eval_cmd, log_file)

        # Summarize
        print(f"\n=======================================================")
        print(f"[*] Summarizing results for {t.upper()}")
        print(f"=======================================================")
        summary_cmd = [PYTHON_EXE, "summarize.py", str(runs_dir)]
        run_cmd(summary_cmd, log_file)

    # Generate full markdown report
    print(f"\n=======================================================")
    print(f"[*] Generating full results summary report")
    print(f"=======================================================")
    report_cmd = [PYTHON_EXE, "make_report.py", str(PROJECT_ROOT)]
    run_cmd(report_cmd, LOGS_DIR / "report.log")
    print("\n[+] All experiments completed successfully!")

if __name__ == "__main__":
    main()
