import csv
import time
from pathlib import Path


class ExperimentLogger:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.output_dir / "metrics.csv"
        self.start_time = time.perf_counter()

        with open(self.metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "wallclock_seconds", "split", "loss"])

    def log(self, step: int, split: str, loss: float):
        elapsed = time.perf_counter() - self.start_time

        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                elapsed,
                split,
                loss,
            ])