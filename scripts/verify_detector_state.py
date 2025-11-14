#!/usr/bin/env python3
"""BaseFlareDetector の状態が既知のベースラインと一致するか検証するスクリプト。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = PROJECT_ROOT / "docs" / "regression" / "base_flare_detector_state.json"


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


def diff_states(baseline: Any, current: Any, path: str = "") -> Iterable[str]:
    if baseline == current:
        return []

    if isinstance(baseline, dict) and isinstance(current, dict):
        diffs = []
        keys = sorted(set(baseline.keys()) | set(current.keys()))
        for key in keys:
            new_path = f"{path}.{key}" if path else key
            if key not in baseline:
                diffs.append(f"{new_path}: baseline に存在しません (現行のみ)")
                continue
            if key not in current:
                diffs.append(f"{new_path}: 現行結果に存在しません (baseline のみ)")
                continue
            diffs.extend(diff_states(baseline[key], current[key], new_path))
        return diffs

    return [f"{path}: baseline={baseline} != current={current}"]


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = capture_state(args)

    if args.update_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        print(f"Baseline を更新しました: {args.baseline}")
        return

    if not args.baseline.exists():
        raise FileNotFoundError(f"baseline ファイルが見つかりません。まず --update-baseline で作成してください: {args.baseline}")

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    diffs = list(diff_states(baseline, snapshot))
    if diffs:
        print("状態差分を検出しました:")
        for diff in diffs:
            print(f"- {diff}")
        raise SystemExit(1)

    print("BaseFlareDetector のクラス変数／インスタンス変数は baseline と一致しています。")


if __name__ == "__main__":
    main()
