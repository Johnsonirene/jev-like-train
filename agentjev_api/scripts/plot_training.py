"""Plot AgentJev loss, development metrics, and elapsed time from saved logs."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    folder = args.output or args.run_dir / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    events_path = args.run_dir / "events.jsonl"
    rows = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("The run has no training events.")
    steps = [row["step"] for row in rows]
    losses = np.asarray([row["loss"] for row in rows], dtype=float)
    # Available trailing observations only; no future steps enter smoothing.
    ends = np.arange(1, len(losses) + 1)
    starts = np.maximum(0, ends - 50)
    cumulative = np.concatenate(([0.0], np.cumsum(losses)))
    smoothed = (cumulative[ends] - cumulative[starts]) / (ends - starts)
    dev = [row for row in rows if "dev" in row]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes[0, 0].plot(steps, losses, alpha=0.2, linewidth=0.7, label="Raw")
    axes[0, 0].plot(steps, smoothed, label="Trailing 50-step mean")
    axes[0, 0].set_title("Training loss")
    axes[0, 0].legend()
    axes[0, 1].plot([r["step"] for r in dev], [r["dev"]["accuracy"] * 100 for r in dev], "-o")
    axes[0, 1].set_title("Uncalibrated development accuracy")
    axes[0, 1].set_ylabel("Accuracy (%)")
    axes[1, 0].plot([r["step"] for r in dev], [r["dev"]["soft_cross_entropy"] for r in dev], "-o")
    axes[1, 0].set_title("Development soft cross entropy (selection metric)")
    axes[1, 1].plot(steps, [r["seconds"] for r in rows])
    axes[1, 1].set_title("Cumulative training wall time")
    axes[1, 1].set_ylabel("Seconds")
    for ax in axes.flat:
        ax.set_xlabel("Optimizer step")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    artifacts = []
    for extension in ("png", "svg"):
        path = folder / ("training_curves." + extension)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        artifacts.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    plt.close(fig)
    manifest = {"passed": True, "training_steps": len(rows), "dev_observations": len(dev), "source_file": "events.jsonl", "source_sha256": hashlib.sha256(events_path.read_bytes()).hexdigest(), "artifacts": artifacts, "notes": ["Only saved training and development observations are plotted; no test inference is run.", "Development observations exclude the separate initialization report."]}
    (folder / "plot-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved PNG/SVG plots and source hashes to {folder}")


if __name__ == "__main__":
    main()
