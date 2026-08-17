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

        self.activation_norms_path = self.output_dir / "activation_norms.csv"
        with open(self.activation_norms_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "wallclock_seconds", "split", "layer", "mean_l2_norm"])

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

    def log_layer_norms(self, step: int, split: str, layer_norms: list[float]):
        elapsed = time.perf_counter() - self.start_time

        with open(self.activation_norms_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(
                [
                    [step, elapsed, split, layer, norm]
                    for layer, norm in enumerate(layer_norms)
                ]
            )
