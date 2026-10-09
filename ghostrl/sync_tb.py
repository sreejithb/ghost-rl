"""Pull a Modal run's TensorBoard logs into runs/<name>/ every few seconds.

    python -m ghostrl.sync_tb arcade1          # leave running
    tensorboard --logdir runs                  # in another terminal; open http://localhost:6006
"""
import subprocess
import sys
import time
from pathlib import Path


def main():
    name = sys.argv[1]
    every = float(sys.argv[2]) if len(sys.argv) > 2 else 30
    dest = Path("runs") / name
    dest.mkdir(parents=True, exist_ok=True)
    while True:
        for f in ("tb", "metrics.csv", "eval.json", "config.json"):
            r = subprocess.run(["modal", "volume", "get", "--force", "ghost-rl-runs", f"{name}/{f}", str(dest)],
                               capture_output=True, text=True)
            if r.returncode and f == "tb":
                print("waiting for the run to publish logs...", r.stderr.strip().splitlines()[-1:] or "", flush=True)
        print(time.strftime("%H:%M:%S"), "synced", flush=True)
        time.sleep(every)


if __name__ == "__main__":
    main()
