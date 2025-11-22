#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.flarepy_DS_Tuc_A import FlareDetector_DS_Tuc_A  # noqa: E402
from src.flarepy_EK_Dra import FlareDetector_EK_Dra  # noqa: E402
from src.flarepy_V889_Her import FlareDetector_V889_Her  # noqa: E402

TARGET_CLASSES = {
    "DS_Tuc_A": FlareDetector_DS_Tuc_A,
    "EK_Dra": FlareDetector_EK_Dra,
    "V889_Her": FlareDetector_V889_Her,
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare LombScargle rotation periods for method='auto' vs 'fast'.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "TESS",
        help="Root directory containing TESS FITS files (per-target subdirectories).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "performance" / "rotation_lomb_methods",
        help="Directory to write CSV and plots.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.fits",
        help="Glob pattern for FITS files within each target directory.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Open generated Plotly figures in a browser.",
    )
    return parser.parse_args(argv)


def collect_results(args: argparse.Namespace) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    for target, cls in TARGET_CLASSES.items():
        target_root = args.data_root / target
        if not target_root.exists():
            continue

        fits_paths = sorted(target_root.rglob(args.pattern))
        if not fits_paths:
            continue

        for fits_path in fits_paths:
            row: dict[str, object] = {
                "target": target,
                "fits_path": str(fits_path),
            }

            try:
                detector = cls(file=str(fits_path), process_data=False)
            except Exception as exc:  # noqa: BLE001
                row["status"] = "error_init"
                row["error"] = repr(exc)
                records.append(row)
                continue

            data_name = getattr(detector, "data_name", fits_path.stem)
            row["data_name"] = data_name

            try:
                detector.remove()
            except Exception as exc:  # noqa: BLE001
                row["status"] = "error_remove"
                row["error"] = repr(exc)
                records.append(row)
                continue

            for method in ("auto", "fast"):
                try:
                    detector.rotation_ls_method = method
                    t0 = time.perf_counter()
                    detector.rotation_period()
                    t1 = time.perf_counter()
                    period = detector.per
                    per_err = detector.per_err
                    power = detector.power
                    max_power = float(np.nanmax(power)) if power is not None else np.nan
                except Exception as exc:  # noqa: BLE001
                    row.setdefault("status", "ok")
                    row["status"] = f"error_{method}"
                    row.setdefault("error", repr(exc))
                    row[f"{method}_period"] = np.nan
                    row[f"{method}_period_err"] = np.nan
                    row[f"{method}_peak_power"] = np.nan
                    row[f"{method}_runtime_s"] = np.nan
                    continue

                row[f"{method}_period"] = float(period) if period is not None else np.nan
                row[f"{method}_period_err"] = float(per_err)
                row[f"{method}_peak_power"] = max_power
                row[f"{method}_runtime_s"] = t1 - t0
                row.setdefault("status", "ok")

            auto_period = row.get("auto_period")
            fast_period = row.get("fast_period")
            if (
                isinstance(auto_period, (int, float))
                and isinstance(fast_period, (int, float))
                and not np.isnan(auto_period)
                and not np.isnan(fast_period)
            ):
                delta = fast_period - auto_period
                row["period_delta"] = delta
                row["period_abs_delta"] = abs(delta)
                row["period_abs_delta_hours"] = abs(delta) * 24.0
                row["period_rel_delta"] = delta / auto_period if auto_period != 0 else np.nan

            records.append(row)

    if not records:
        return pd.DataFrame()

    return pd.DataFrame.from_records(records)


def build_plots(df: pd.DataFrame, args: argparse.Namespace) -> None:
    if df.empty:
        return

    valid = df.dropna(subset=["auto_period", "fast_period"]).copy()
    if valid.empty:
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fig_scatter = px.scatter(
        valid,
        x="auto_period",
        y="fast_period",
        color="target",
        hover_data=["fits_path", "data_name"],
        title="Rotation period: LombScargle method='auto' vs 'fast'",
    )
    x_min = float(min(valid["auto_period"].min(), valid["fast_period"].min()))
    x_max = float(max(valid["auto_period"].max(), valid["fast_period"].max()))
    fig_scatter.add_trace(
        go.Scatter(
            x=[x_min, x_max],
            y=[x_min, x_max],
            mode="lines",
            name="y = x",
            line=dict(color="black", dash="dash"),
            showlegend=True,
        ),
    )
    scatter_path = args.output_dir / "rotation_period_auto_vs_fast_scatter.png"
    fig_scatter.write_image(scatter_path, scale=2)
    if args.show_plot:
        pio.show(fig_scatter, renderer="browser")

    non_nan = valid.dropna(subset=["period_abs_delta_hours"]).copy() if "period_abs_delta_hours" in valid.columns else valid
    if not non_nan.empty:
        fig_hist = px.histogram(
            non_nan,
            x="period_abs_delta_hours",
            color="target",
            nbins=30,
            barmode="overlay",
            title="|fast - auto| rotation period difference [hours]",
        )
        fig_hist.update_xaxes(title="|Δ period| [hours]")
        fig_hist.update_yaxes(title="Number of light curves")
        hist_path = args.output_dir / "rotation_period_diff_hist_hours.png"
        fig_hist.write_image(hist_path, scale=2)
        if args.show_plot:
            pio.show(fig_hist, renderer="browser")

    if "period_abs_delta" in valid.columns:
        summary = (
            valid.dropna(subset=["period_abs_delta"])
            .groupby("target")
            .agg(
                n=("target", "size"),
                mean_abs_delta_days=("period_abs_delta", "mean"),
                max_abs_delta_days=("period_abs_delta", "max"),
                mean_abs_delta_hours=("period_abs_delta_hours", "mean"),
                max_abs_delta_hours=("period_abs_delta_hours", "max"),
                mean_rel_delta=("period_rel_delta", "mean"),
                max_rel_delta=("period_rel_delta", "max"),
            )
        )
        summary_path = args.output_dir / "rotation_period_diff_summary.csv"
        summary.to_csv(summary_path)
        print("=== Rotation period diff summary per target ===")
        print(summary.to_string())
        print(f"Summary CSV: {summary_path}")

        summary_reset = summary.reset_index()
        header_values = list(summary_reset.columns)
        cell_values = [summary_reset[col].tolist() for col in header_values]
        fig_table = go.Figure(
            data=[
                go.Table(
                    header=dict(values=header_values),
                    cells=dict(values=cell_values),
                ),
            ],
        )
        table_path = args.output_dir / "rotation_period_diff_summary_table.png"
        fig_table.write_image(table_path, scale=2)
        if args.show_plot:
            pio.show(fig_table, renderer="browser")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = collect_results(args)
    csv_path = args.output_dir / "rotation_period_lomb_methods_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"Results CSV: {csv_path}")
    build_plots(df, args)


if __name__ == "__main__":
    main()
