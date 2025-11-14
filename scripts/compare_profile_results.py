#!/usr/bin/env python3
"""BaseFlareDetector プロファイル結果を比較し、差分グラフを生成するスクリプト。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px


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


def plot_comparison(df: pd.DataFrame, output_png: Path, label_before: str, label_after: str, top_n: int) -> None:
    top = df.head(top_n).copy()
    before_col = f"{label_before}_cumtime"
    after_col = f"{label_after}_cumtime"

    melted = top.melt(
        id_vars=["function_key", "percent_change"],
        value_vars=[before_col, after_col],
        var_name="label",
        value_name="cumtime",
    )
    melted["label"] = melted["label"].replace(
        {
            before_col: label_before,
            after_col: label_after,
        }
    )

    fig = px.bar(
        melted,
        x="cumtime",
        y="function_key",
        color="label",
        orientation="h",
        labels={"cumtime": "Cumulative time [s]", "function_key": "Function"},
        barmode="group",
        text=melted["cumtime"].map(lambda v: f"{v:.3f}s"),
    )
    fig.update_layout(
        height=max(500, 80 * len(top)),
        yaxis=dict(autorange="reversed"),
        title=f"BaseFlareDetector cumulative time comparison (top {top_n})",
        margin=dict(l=200, r=120, t=60, b=40),
    )

    max_val = float(melted["cumtime"].max() or 1.0)
    for _, row in top.iterrows():
        pct = row["percent_change"]
        if np.isnan(pct):
            continue
        fig.add_annotation(
            x=max_val * 1.05,
            y=row["function_key"],
            text=f"{pct:+.1f}%",
            showarrow=False,
            font=dict(size=12),
            xanchor="left",
            yanchor="middle",
        )

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(output_png, scale=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="プロファイル CSV を比較し、比率付きグラフを生成します。")
    parser.add_argument("--before", type=Path, required=True, help="最適化前の profile_full.csv")
    parser.add_argument("--after", type=Path, required=True, help="最適化後の profile_full.csv")
    parser.add_argument("--output-dir", type=Path, required=True, help="比較結果の出力ディレクトリ")
    parser.add_argument("--label-before", type=str, default="before", help="前データの凡例ラベル")
    parser.add_argument("--label-after", type=str, default="after", help="後データの凡例ラベル")
    parser.add_argument("--top-n", type=int, default=12, help="グラフに表示する関数数")
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
    plot_comparison(comparison, png_path, args.label_before, args.label_after, args.top_n)

    print(f"比較 CSV: {csv_path}")
    print(f"比較グラフ: {png_path}")


if __name__ == "__main__":
    main()
