#!/usr/bin/env python3
"""BaseFlareDetector プロファイル結果を比較し、差分グラフを生成するスクリプト。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


def canonical_function_label(func: str) -> str:
    parts = func.split(":")
    if len(parts) >= 3:
        file_part = parts[0]
        func_part = parts[-1]
        return f"{file_part}::{func_part}"
    return func


def load_profile(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"プロファイル CSV が見つかりません: {path}")
    df = pd.read_csv(path)
    mask = df["function"].str.contains("base_flare_detector.py")
    df = df[mask].copy()
    df["key"] = df["function"].apply(canonical_function_label)
    return df


def build_comparison(before: pd.DataFrame, after: pd.DataFrame, label_before: str, label_after: str) -> pd.DataFrame:
    merged = pd.merge(
        before[["key", "function", "cumtime"]],
        after[["key", "function", "cumtime"]],
        on="key",
        how="outer",
        suffixes=("_before", "_after"),
    )
    for col in ("cumtime_before", "cumtime_after"):
        merged[col] = merged[col].fillna(0.0)
    merged["function_before"] = merged["function_before"].fillna(merged["function_after"])
    merged["function_after"] = merged["function_after"].fillna(merged["function_before"])
    merged["delta"] = merged["cumtime_after"] - merged["cumtime_before"]
    merged["percent_change"] = np.where(
        merged["cumtime_before"] > 0,
        merged["delta"] / merged["cumtime_before"] * 100,
        np.nan,
    )
    merged.sort_values("cumtime_before", ascending=False, inplace=True)
    merged.rename(
        columns={
            "cumtime_before": f"{label_before}_cumtime",
            "cumtime_after": f"{label_after}_cumtime",
        },
        inplace=True,
    )
    merged.insert(0, "function_key", merged["key"])
    merged.drop(columns=["key"], inplace=True)
    return merged


def plot_comparison(
    df: pd.DataFrame, output_png: Path, label_before: str, label_after: str, top_n: int, show_plot: bool = False
) -> None:
    top = df.head(top_n).copy()
    before_col = f"{label_before}_cumtime"
    after_col = f"{label_after}_cumtime"
    labels = list(top["function_key"])
    percent_fmt = top["percent_change"].apply(lambda v: f"{v:+.1f}%" if not np.isnan(v) else "N/A").tolist()

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            f"Top {top_n} cumulative time comparison",
            "Share of after cumulative time (relative bottleneck)",
        ),
        specs=[[{"type": "xy"}], [{"type": "xy"}]],
        vertical_spacing=0.12,
    )

    fig.add_trace(
        go.Bar(
            y=labels,
            x=top[before_col],
            orientation="h",
            name=label_before,
            marker_color="#1f77b4",
            text=top[before_col].map(lambda v: f"{v:.3f}s"),
            textposition="auto",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            y=labels,
            x=top[after_col],
            orientation="h",
            name=label_after,
            marker_color="#ff7f0e",
            text=top[after_col].map(lambda v: f"{v:.3f}s"),
            textposition="auto",
        ),
        row=1,
        col=1,
    )

    after_total = top[after_col].sum()
    share_percent = np.where(after_total > 0, top[after_col] / after_total * 100, np.nan)
    share_colors = [
        "#d62728" if pct >= 40 else "#ff7f0e" if pct >= 20 else "#2ca02c" for pct in share_percent
    ]
    fig.add_trace(
        go.Bar(
            y=labels,
            x=share_percent,
            orientation="h",
            name="After share",
            marker_color=share_colors,
            text=[f"{pct:.1f}%" if not np.isnan(pct) else "N/A" for pct in share_percent],
            textposition="auto",
            customdata=top["percent_change"],
            hovertemplate="Function: %{y}<br>After share: %{x:.1f}%<br>Δ% vs before: %{customdata:+.1f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        barmode="group",
        height=max(900, 65 * len(labels)),
        yaxis=dict(autorange="reversed", title="Function"),
        xaxis=dict(title="Cumulative time [s]"),
        xaxis2=dict(title="Share of after cumulative time [%]", range=[0, 100], zeroline=True, zerolinecolor="#888"),
        margin=dict(l=220, r=60, t=80, b=40),
        title="BaseFlareDetector cumulative time comparison",
    )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(output_png, scale=2)
    if show_plot:
        pio.show(fig, renderer="browser")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="プロファイル CSV を比較し、比率付きグラフを生成します。")
    parser.add_argument("--before", type=Path, required=True, help="最適化前の profile_full.csv")
    parser.add_argument("--after", type=Path, required=True, help="最適化後の profile_full.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="比較結果の出力ディレクトリ")
    parser.add_argument("--label-before", type=str, default="before", help="前データの凡例ラベル")
    parser.add_argument("--label-after", type=str, default="after", help="後データの凡例ラベル")
    parser.add_argument("--top-n", type=int, default=12, help="グラフに表示する関数数")
    parser.add_argument("--show-plot", action="store_true", help="生成したグラフをブラウザで表示します")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    before_df = load_profile(args.before)
    after_df = load_profile(args.after)

    comparison = build_comparison(before_df, after_df, args.label_before, args.label_after)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "base_flare_detector_cumtime_comparison.csv"
    comparison.to_csv(csv_path, index=False)

    png_path = args.output_dir / "base_flare_detector_cumtime_comparison.png"
    plot_comparison(comparison, png_path, args.label_before, args.label_after, args.top_n, args.show_plot)

    print(f"比較 CSV: {csv_path}")
    print(f"比較グラフ: {png_path}")


if __name__ == "__main__":
    main()
