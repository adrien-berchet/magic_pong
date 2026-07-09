#!/usr/bin/env python3
"""
Architecture comparison dashboard for Magic Pong DQN training runs.

Loads *_metrics.json files produced by train_optimized.py and generates a
4×2 comparison figure plus a summary CSV/table for picking the best model.

Usage:
    # Compare specific files
    python dashboard.py --metrics models/run_A_metrics.json models/run_B_metrics.json

    # Load all metrics files from a directory
    python dashboard.py --dir models/

    # Save to custom path
    python dashboard.py --dir models/ --output results/comparison.png

    # Show interactively instead of saving
    python dashboard.py --dir models/ --show
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def load_metrics(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def find_metrics_files(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).glob("*_metrics.json"))


def find_convergence_episode(
    episodes: list[int],
    win_rates: list[float | None],
    threshold: float = 0.70,
    window: int = 5,
) -> int | None:
    rates = [r for r in win_rates if r is not None]
    if len(rates) < window:
        return None
    for i in range(window - 1, len(win_rates)):
        chunk = [win_rates[j] for j in range(i - window + 1, i + 1) if win_rates[j] is not None]
        if len(chunk) == window and np.mean(chunk) >= threshold:
            return episodes[i]
    return None


def _smooth(values: list, w: int = 5) -> list:
    result = []
    for i, v in enumerate(values):
        if v is None:
            result.append(None)
            continue
        window = [x for x in values[max(0, i - w + 1) : i + 1] if x is not None]
        result.append(float(np.mean(window)) if window else None)
    return result


def _valid_xy(episodes, series):
    pairs = [(e, v) for e, v in zip(episodes, series) if v is not None]
    if not pairs:
        return [], []
    return zip(*pairs)


def _plot_metric(ax, runs: list[dict], key: str, title: str, ylabel: str, ylim=None):
    has_data = False
    for run in runs:
        meta = run["metadata"]
        label = (
            f"{meta['model_prefix']} "
            f"(h={meta['architecture']['hidden_size']}, "
            f"l={meta['architecture']['layer_size']}, "
            f"{meta['architecture']['param_count']}p)"
        )
        episodes = run["per_interval"].get("episode", [])
        series = run["per_interval"].get(key, [None] * len(episodes))
        xs, ys = _valid_xy(episodes, series)
        if not xs:
            continue
        xs, ys = list(xs), list(ys)
        has_data = True
        (line,) = ax.plot(xs, ys, alpha=0.25, linewidth=0.8)
        ax.plot(xs, _smooth(ys), linewidth=1.5, label=label, color=line.get_color())

    ax.set_title(title)
    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if not has_data:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=14)
    else:
        ax.legend(fontsize=7)


def generate_comparison_figure(runs: list[dict], output_path: Path | None, show: bool) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(15, 18))
    fig.suptitle("DQN Architecture Comparison", fontsize=15, fontweight="bold")

    # [0,0] Point win rate
    ax = axes[0, 0]
    _plot_metric(ax, runs, "point_win_rate", "Point Win Rate", "Win rate", ylim=(0, 1))
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # [0,1] Match win rate
    ax = axes[0, 1]
    _plot_metric(ax, runs, "match_win_rate", "Match Win Rate", "Win rate", ylim=(0, 1))
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    # [1,0] TD loss
    _plot_metric(axes[1, 0], runs, "td_loss", "TD Loss (Bellman error)", "Loss")

    # [1,1] Mean Q-value
    _plot_metric(axes[1, 1], runs, "mean_q", "Mean Q-Value (probe batch)", "Q-value")

    # [2,0] Gradient norm
    _plot_metric(axes[2, 0], runs, "grad_norm", "Gradient Norm (pre-clip)", "L2 norm")

    # [2,1] Epsilon
    _plot_metric(axes[2, 1], runs, "epsilon", "Epsilon (exploration rate)", "Epsilon", ylim=(0, 1))

    # [3,0] Draw rate + episode length (dual y-axis)
    ax = axes[3, 0]
    ax2 = ax.twinx()
    for run in runs:
        episodes = run["per_interval"].get("episode", [])
        draw = run["per_interval"].get("draw_rate", [])
        ep_len = run["per_interval"].get("avg_episode_length", [])
        dx, dy = _valid_xy(episodes, draw)
        if dx:
            dx, dy = list(dx), list(dy)
            (line,) = ax.plot(dx, dy, alpha=0.25, linewidth=0.8)
            ax.plot(dx, _smooth(dy), linewidth=1.5, color=line.get_color())
        lx, ly = _valid_xy(episodes, ep_len)
        if lx:
            lx, ly = list(lx), list(ly)
            (line,) = ax2.plot(lx, ly, alpha=0.25, linewidth=0.8, linestyle="--")
            ax2.plot(lx, _smooth(ly), linewidth=1.5, linestyle="--", color=line.get_color())
    ax.set_title("Draw Rate (solid) & Episode Length (dashed)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Draw rate")
    ax2.set_ylabel("Avg episode length")

    # [3,1] Goals for vs against
    ax = axes[3, 1]
    has_goals = False
    for run in runs:
        episodes = run["per_interval"].get("episode", [])
        gf = run["per_interval"].get("avg_goals_for", [])
        ga = run["per_interval"].get("avg_goals_against", [])
        x1, y1 = _valid_xy(episodes, gf)
        if x1:
            x1, y1 = list(x1), list(y1)
            (line,) = ax.plot(x1, y1, alpha=0.25, linewidth=0.8)
            ax.plot(x1, _smooth(y1), linewidth=1.5, color=line.get_color(), label="For")
            has_goals = True
        x2, y2 = _valid_xy(episodes, ga)
        if x2:
            x2, y2 = list(x2), list(y2)
            ax.plot(x2, y2, alpha=0.25, linewidth=0.8, linestyle="--", color=line.get_color())
            ax.plot(
                x2,
                _smooth(y2),
                linewidth=1.5,
                linestyle="--",
                color=line.get_color(),
                label="Against",
            )
    ax.set_title("Avg Goals For (solid) vs Against (dashed)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Goals per episode")
    if has_goals:
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes, fontsize=14)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Dashboard saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def _fmt(v, fmt=".1f"):
    if v is None:
        return "—"
    return format(v, fmt)


def build_summary(runs: list[dict]) -> list[dict]:
    rows = []
    for run in runs:
        meta = run["metadata"]
        arch = meta["architecture"]
        pi = run["per_interval"]
        episodes = pi.get("episode", [])
        pwr = pi.get("point_win_rate", [])
        mwr = pi.get("match_win_rate", [])
        conv = find_convergence_episode(episodes, pwr)
        final_pwr = next((v for v in reversed(pwr) if v is not None), None)
        final_mwr = next((v for v in reversed(mwr) if v is not None), None)
        rows.append(
            {
                "model": meta["model_prefix"],
                "hidden_size": arch["hidden_size"],
                "layer_size": arch["layer_size"],
                "param_count": arch["param_count"],
                "inference_us": meta.get("inference_us"),
                "ep_to_70pct": conv,
                "final_point_wr": None if final_pwr is None else final_pwr * 100,
                "final_match_wr": None if final_mwr is None else final_mwr * 100,
                "training_time_min": meta.get("training_time_seconds", 0) / 60,
                "total_episodes": meta.get("total_episodes"),
            }
        )
    return rows


def print_summary(rows: list[dict]) -> None:
    headers = [
        "Model",
        "h_size",
        "layers",
        "Params",
        "Infer(µs)",
        "Ep→70%WR",
        "Final pWR%",
        "Final mWR%",
        "Time(min)",
    ]
    col_w = [max(len(h), 12) for h in headers]

    def fmt_row(r):
        return [
            r["model"][: col_w[0]],
            str(r["hidden_size"]),
            str(r["layer_size"]),
            str(r["param_count"]),
            _fmt(r.get("inference_us"), ".1f"),
            str(r["ep_to_70pct"]) if r["ep_to_70pct"] else "—",
            _fmt(r.get("final_point_wr"), ".1f"),
            _fmt(r.get("final_match_wr"), ".1f"),
            _fmt(r.get("training_time_min"), ".1f"),
        ]

    header_line = "  ".join(h.ljust(col_w[i]) for i, h in enumerate(headers))
    print("\n" + "=" * len(header_line))
    print("ARCHITECTURE COMPARISON SUMMARY")
    print("=" * len(header_line))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        cells = fmt_row(row)
        print("  ".join(c.ljust(col_w[i]) for i, c in enumerate(cells)))
    print("=" * len(header_line))


def save_summary_csv(rows: list[dict], output_path: Path) -> None:
    fields = [
        "model",
        "hidden_size",
        "layer_size",
        "param_count",
        "inference_us",
        "ep_to_70pct",
        "final_point_wr",
        "final_match_wr",
        "training_time_min",
        "total_episodes",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary CSV saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Magic Pong DQN architecture comparison dashboard")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--metrics",
        nargs="+",
        metavar="FILE",
        help="One or more *_metrics.json files to compare",
    )
    group.add_argument(
        "--dir",
        metavar="DIR",
        help="Directory to scan for *_metrics.json files",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Output path for the comparison PNG (default: <dir>/comparison.png or ./comparison.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the figure interactively (can combine with --output)",
    )
    parser.add_argument(
        "--convergence_threshold",
        type=float,
        default=0.70,
        help="Win rate threshold for 'episodes to convergence' metric (default: 0.70)",
    )
    parser.add_argument(
        "--no_save_csv",
        action="store_true",
        help="Skip saving the summary CSV",
    )
    args = parser.parse_args()

    if args.metrics:
        paths = [Path(p) for p in args.metrics]
        missing = [p for p in paths if not p.exists()]
        if missing:
            print(f"Error: files not found: {missing}", file=sys.stderr)
            sys.exit(1)
        base_dir = paths[0].parent
    else:
        base_dir = Path(args.dir)
        paths = find_metrics_files(base_dir)
        if not paths:
            print(f"No *_metrics.json files found in {base_dir}", file=sys.stderr)
            sys.exit(1)

    print(f"Loading {len(paths)} metrics file(s):")
    runs = []
    for p in paths:
        print(f"  {p}")
        runs.append(load_metrics(p))

    if not runs:
        print("No valid metrics loaded.", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    output_path = None
    if not args.show or args.output:
        output_path = Path(args.output) if args.output else base_dir / "comparison.png"

    generate_comparison_figure(runs, output_path, args.show)

    summary_rows = build_summary(runs)
    print_summary(summary_rows)

    if not args.no_save_csv:
        csv_path = (
            Path(args.output).with_suffix(".csv")
            if args.output
            else base_dir / "comparison_summary.csv"
        )
        save_summary_csv(summary_rows, csv_path)


if __name__ == "__main__":
    main()
