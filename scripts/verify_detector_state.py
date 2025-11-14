#!/usr/bin/env python3
"""BaseFlareDetector の状態が既知のベースラインと一致するか検証するスクリプト。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = PROJECT_ROOT / "reports" / "state_validation" / "base_flare_detector_state_baseline.json"


def _array_factory(dtype=float) -> Callable[[], np.ndarray]:
    return lambda: np.array([], dtype=dtype)


CLASS_DEFAULT_FACTORIES: Dict[str, Callable[[], Any]] = {
    "array_flare_ratio": _array_factory(float),
    "array_observation_time": _array_factory(float),
    "array_energy_ratio": _array_factory(float),
    "array_amplitude": _array_factory(float),
    "array_starspot": _array_factory(float),
    "array_starspot_ratio": _array_factory(float),
    "array_data_name": _array_factory(object),
    "array_per": _array_factory(float),
    "array_per_err": _array_factory(float),
    "average_flare_ratio": lambda: 0.0,
}


def reset_class_state(BaseFlareDetector: Any) -> None:
    for attr, factory in CLASS_DEFAULT_FACTORIES.items():
        value = factory()
        if isinstance(value, np.ndarray):
            value = value.copy()
        setattr(BaseFlareDetector, attr, value)


def hash_ndarray(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()


def serialize_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        shape = list(value.shape)
        dtype_str = str(value.dtype)
        if value.dtype == object:
            elements = [repr(v) for v in value.tolist()]
            joined = "\u241f".join(elements)  # unit separator-like char
            digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()
            return {
                "type": "ndarray_object",
                "dtype": dtype_str,
                "shape": shape,
                "length": len(elements),
                "sha256": digest,
            }
        return {
            "type": "ndarray",
            "dtype": dtype_str,
            "shape": shape,
            "sha256": hash_ndarray(value),
        }
    if isinstance(value, (np.floating, np.integer)):
        return {
            "type": "scalar",
            "value": value.item(),
        }
    if isinstance(value, (int, float, str, bool)) or value is None:
        return {"type": "scalar", "value": value}
    if isinstance(value, Path):
        return {"type": "path", "value": value.as_posix()}
    if isinstance(value, (list, tuple)):
        return {
            "type": "list",
            "items": [serialize_value(v) for v in value],
        }
    if isinstance(value, dict):
        return {
            "type": "dict",
            "items": {k: serialize_value(v) for k, v in sorted(value.items())},
        }
    return {"type": "repr", "value": repr(value)}


def value_repr(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def compare_sections(section_name: str, baseline_section: Dict[str, Any], current_section: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    keys = sorted(set(baseline_section.keys()) | set(current_section.keys()))
    for key in keys:
        base = baseline_section.get(key)
        curr = current_section.get(key)
        if base is None:
            status = "missing_in_baseline"
            row = {
                "section": section_name,
                "key": key,
                "status": status,
                "baseline_repr": None,
                "current_repr": value_repr(curr),
            }
        elif curr is None:
            status = "missing_in_current"
            row = {
                "section": section_name,
                "key": key,
                "status": status,
                "baseline_repr": value_repr(base),
                "current_repr": None,
            }
        else:
            base_repr = value_repr(base)
            curr_repr = value_repr(curr)
            status = "match" if base_repr == curr_repr else "diff"
            row = {
                "section": section_name,
                "key": key,
                "status": status,
                "baseline_repr": base_repr,
                "current_repr": curr_repr,
            }
        rows.append(row)
    return rows


def summarize_differences(baseline: Dict[str, Any], snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    rows.extend(compare_sections("instance", baseline.get("instance", {}), snapshot.get("instance", {})))
    rows.extend(compare_sections("class", baseline.get("class", {}), snapshot.get("class", {})))
    return rows


def plot_summary(rows: List[Dict[str, Any]], output_path: Path, show_plot: bool = False) -> None:
    sections = sorted({row["section"] for row in rows})
    status_order = ["match", "diff", "missing_in_baseline", "missing_in_current"]
    counts = {section: {status: 0 for status in status_order} for section in sections}
    for row in rows:
        counts[row["section"]][row["status"]] += 1

    summary_rows = []
    for section in sections:
        total = sum(counts[section].values())
        match_rate = (counts[section]["match"] / total * 100) if total else 0.0
        summary_rows.append(
            {
                "Section": section,
                "Total vars": total,
                "Matches": counts[section]["match"],
                "Diffs": counts[section]["diff"],
                "Missing in baseline": counts[section]["missing_in_baseline"],
                "Missing in current": counts[section]["missing_in_current"],
                "Match rate (%)": round(match_rate, 1),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=list(summary_df.columns),
                    fill_color="#1f2a44",
                    font=dict(color="white"),
                    align="left",
                ),
                cells=dict(
                    values=[summary_df[col] for col in summary_df.columns],
                    fill_color=[["#f9f9f9", "#e8eff7"] * (len(summary_df) // 2 + 1)],
                    align="left",
                ),
            )
        ]
    )
    fig.update_layout(
        title="BaseFlareDetector state summary",
        height=300 + 30 * len(summary_df),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(output_path, scale=2)
    if show_plot:
        pio.show(fig, renderer="browser")


def plot_detailed(rows: List[Dict[str, Any]], output_path: Path, show_plot: bool = False) -> None:
    rows_sorted = sorted(rows, key=lambda r: (r["section"], r["key"]))
    if not rows_sorted:
        return
    detail_df = pd.DataFrame(rows_sorted)
    detail_df["Variable"] = detail_df["section"] + "." + detail_df["key"]
    detail_df["Baseline"] = detail_df["baseline_repr"].apply(lambda v: (v or "")[:120])
    detail_df["Current"] = detail_df["current_repr"].apply(lambda v: (v or "")[:120])
    detail_df["Status"] = detail_df["status"]

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Variable", "Baseline", "Current", "Status"],
                    fill_color="#1f2a44",
                    font=dict(color="white"),
                    align="left",
                ),
                cells=dict(
                    values=[detail_df[col] for col in ["Variable", "Baseline", "Current", "Status"]],
                    fill_color=[["#f9f9f9", "#eef3fb"] * (len(detail_df) // 2 + 1)],
                    align="left",
                ),
            )
        ]
    )
    fig.update_layout(
        title="BaseFlareDetector variable comparison",
        height=max(600, 25 * len(detail_df)),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(output_path, scale=2)
    if show_plot:
        pio.show(fig, renderer="browser")


def capture_state(args: argparse.Namespace) -> Dict[str, Any]:
    if not args.fits.exists():
        raise FileNotFoundError(f"FITS ファイルが存在しません: {args.fits}")

    if str(PROJECT_ROOT) not in __import__("sys").path:
        __import__("sys").path.insert(0, str(PROJECT_ROOT))

    from src.base_flare_detector import BaseFlareDetector

    reset_class_state(BaseFlareDetector)

    detector = BaseFlareDetector(
        file=str(args.fits),
        process_data=False,
        run_process_data_2=args.run_process_data_2,
        sector_threshold=args.sector_threshold,
    )
    detector.process_data(
        skip_remove=args.skip_remove,
        ene_thres_low=args.ene_thres_low,
        ene_thres_high=args.ene_thres_high,
    )

    instance_state = {k: serialize_value(v) for k, v in sorted(detector.__dict__.items())}
    class_state = {
        k: serialize_value(getattr(BaseFlareDetector, k)) for k in sorted(CLASS_DEFAULT_FACTORIES.keys())
    }
    return {"instance": instance_state, "class": class_state}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BaseFlareDetector のクラス変数／インスタンス変数が baseline と一致するか検証します。"
    )
    parser.add_argument("--fits", type=Path, required=True, help="検証対象の FITS ファイル")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help=f"baseline JSON の保存場所 (既定: {DEFAULT_BASELINE})",
    )
    parser.add_argument("--skip-remove", action="store_true", help="process_data(skip_remove=True) で検証")
    parser.add_argument("--run-process-data-2", action="store_true", help="コンストラクタ引数 run_process_data_2 を有効化")
    parser.add_argument("--sector-threshold", type=int, default=None, help="コンストラクタ引数 sector_threshold の指定値")
    parser.add_argument("--ene-thres-low", type=float, default=None, help="process_data の ene_thres_low 上書き値")
    parser.add_argument("--ene-thres-high", type=float, default=None, help="process_data の ene_thres_high 上書き値")
    parser.add_argument("--update-baseline", action="store_true", help="baseline を現在の結果で更新します")
    parser.add_argument("--plot", type=Path, help="集計グラフ (セクション別) の保存先 PNG")
    parser.add_argument("--detail-plot", type=Path, help="変数ごとのステータス可視化グラフの保存先 PNG")
    parser.add_argument("--table-csv", type=Path, help="各変数の比較結果を CSV で保存するパス")
    parser.add_argument("--show-plot", action="store_true", help="生成した Plotly グラフをブラウザで表示します")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = capture_state(args)

    if args.update_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        print(f"Baseline を更新しました: {args.baseline}")
        summary_rows = summarize_differences(snapshot, snapshot)
    else:
        if not args.baseline.exists():
            raise FileNotFoundError(
                f"baseline ファイルが見つかりません。まず --update-baseline で作成してください: {args.baseline}"
            )
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        summary_rows = summarize_differences(baseline, snapshot)

    if args.table_csv:
        args.table_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.table_csv.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["section", "key", "status", "baseline_repr", "current_repr"])
            writer.writeheader()
            for row in summary_rows:
                writer.writerow(row)
        print(f"比較結果テーブルを出力しました: {args.table_csv}")

    if args.plot:
        plot_summary(summary_rows, args.plot, show_plot=False)
        print(f"比較結果（集計）を図示しました: {args.plot}")
    if args.detail_plot:
        plot_detailed(summary_rows, args.detail_plot, show_plot=args.show_plot)
        print(f"比較結果（一覧）を図示しました: {args.detail_plot}")

    if args.update_baseline:
        if args.table_csv or args.detail_plot:
            print("基準状態の可視化／テーブルを生成しました。")
        return

    diff_rows = [row for row in summary_rows if row["status"] != "match"]
    if diff_rows:
        print("状態差分を検出しました:")
        for row in diff_rows:
            path = f"{row['section']}.{row['key']}"
            if row["status"] == "diff":
                print(f"- {path}: baseline={row['baseline_repr']} vs current={row['current_repr']}")
            elif row["status"] == "missing_in_baseline":
                print(f"- {path}: baseline に存在せず、現行のみ {row['current_repr']}")
            elif row["status"] == "missing_in_current":
                print(f"- {path}: baseline には存在しますが、現行には存在しません")
            else:
                print(f"- {path}: status={row['status']}")
        raise SystemExit(1)

    print("BaseFlareDetector のクラス変数／インスタンス変数は baseline と一致しています。")


if __name__ == "__main__":
    main()
