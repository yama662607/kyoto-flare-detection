#!/usr/bin/env python3
"""Lightweight profiling script for BaseFlareDetector.

Runs `BaseFlareDetector.process_data` on a specified TESS FITS file and
exports ``cProfile`` results to CSV and plots.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.io as pio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile BaseFlareDetector.process_data and generate plots."
    )
    parser.add_argument(
        "--fits",
        type=Path,
        required=True,
        help="TESS light curve FITS file to analyze",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "performance",
        help="Output directory for results (CSV / PNG / .prof)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=12,
        help="Number of functions to show (sorted by cumulative time)",
    )
    parser.add_argument(
        "--skip-remove",
        action="store_true",
        help="Set BaseFlareDetector.process_data skip_remove=True",
    )
    parser.add_argument(
        "--ene-thres-low",
        type=float,
        default=None,
        help="Minimum energy threshold for process_data (defaults if omitted)",
    )
    parser.add_argument(
        "--ene-thres-high",
        type=float,
        default=None,
        help="Maximum energy threshold for process_data (defaults if omitted)",
    )
    parser.add_argument(
        "--run-process-data-2",
        action="store_true",
        help="Set BaseFlareDetector constructor run_process_data_2=True",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Open the generated plot in a browser (Plotly default renderer)",
    )
    return parser.parse_args(argv)


def profile_detector(args: argparse.Namespace) -> tuple[pd.DataFrame, Path]:
    if not args.fits.exists():
        raise FileNotFoundError(f"FITS file not found: {args.fits}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from src.base_flare_detector import BaseFlareDetector

    detector = BaseFlareDetector(
        file=str(args.fits),
        process_data=False,
        run_process_data_2=args.run_process_data_2,
    )

    profiler = cProfile.Profile()
    profiler.enable()
    detector.process_data(
        ene_thres_low=args.ene_thres_low,
        ene_thres_high=args.ene_thres_high,
        skip_remove=args.skip_remove,
    )
    profiler.disable()

    prof_path = args.output_dir / f"{args.fits.stem}_base_flare_detector.prof"
    profiler.dump_stats(str(prof_path))

    stats = pstats.Stats(profiler).strip_dirs()
    rows = []
    for func_desc, stat in stats.stats.items():
        ccalls, ncalls, tottime, cumtime, callers = stat
        filename, line_no, func_name = func_desc
        rows.append(
            {
                "function": f"{Path(filename).name}:{line_no}:{func_name}",
                "ncalls": ncalls,
                "tottime": tottime,
                "cumtime": cumtime,
            }
        )

    df = pd.DataFrame(rows)
    return df, prof_path


def save_outputs(df: pd.DataFrame, args: argparse.Namespace) -> tuple[Path, Path]:
    csv_path = args.output_dir / f"{args.fits.stem}_profile_full.csv"
    df.to_csv(csv_path, index=False)

    filtered = df[df["function"].str.contains("base_flare_detector.py")]
    top_df = filtered.nlargest(args.top_n, "cumtime").copy()
    if top_df.empty:
        top_df = df.nlargest(args.top_n, "cumtime").copy()

    fig_path = args.output_dir / f"{args.fits.stem}_profile_top{args.top_n}.png"
    fig = px.bar(
        top_df,
        x="cumtime",
        y="function",
        orientation="h",
        title="BaseFlareDetector cumulative time (top contributors)",
        labels={
            "cumtime": "Cumulative time [s]",
            "function": "Function (file:line:name)",
        },
        text=top_df["cumtime"].map(lambda v: f"{v:.3f}s"),
    )
    fig.update_layout(
        height=max(400, 60 * len(top_df)),
        yaxis={"autorange": "reversed"},
        margin={"l": 120, "r": 40, "t": 60, "b": 40},
    )
    fig.write_image(fig_path, scale=2)
    if args.show_plot:
        pio.show(fig, renderer="browser")
    return csv_path, fig_path


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    df, prof_path = profile_detector(args)
    csv_path, fig_path = save_outputs(df, args)

    filtered = df[df["function"].str.contains("base_flare_detector.py")]
    top = filtered.nlargest(args.top_n, "cumtime")
    summary = top[["function", "ncalls", "tottime", "cumtime"]]
    print("=== Profile results (top functions in base_flare_detector.py) ===")
    print(summary.to_string(index=False))
    print()
    print(f"Profile data: {prof_path}")
    print(f"CSV output:  {csv_path}")
    print(f"Plot output: {fig_path}")


if __name__ == "__main__":
    main()
