#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import math


def read_csv(path):
    with path.open() as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = []
        for row in reader:
            if not row:
                continue
            rows.append([float(x) for x in row])
    return header, rows


def compare_files(a, b, rtol=1e-10, atol=0.0):
    h1, r1 = read_csv(a)
    h2, r2 = read_csv(b)
    if h1 != h2:
        return False, f"header mismatch: {h1} vs {h2}"
    if len(r1) != len(r2):
        return False, f"row count mismatch: {len(r1)} vs {len(r2)}"
    max_diff = 0.0
    for i, (row1, row2) in enumerate(zip(r1, r2)):
        if len(row1) != len(row2):
            return False, f"row {i} length mismatch: {len(row1)} vs {len(row2)}"
        for j, (x, y) in enumerate(zip(row1, row2)):
            if math.isnan(x) and math.isnan(y):
                continue
            if math.isinf(x) or math.isinf(y):
                if x != y:
                    return False, f"row {i} col {j} inf mismatch: {x} vs {y}"
                continue
            diff = abs(x - y)
            tol = atol + rtol * abs(y)
            if diff > tol:
                return False, f"row {i} col {j} diff {diff} > tol {tol} (x={x}, y={y})"
            if diff > max_diff:
                max_diff = diff
    return True, f"ok (max_diff={max_diff})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir_a', help='debug dir A (e.g., outputs/debug/20260206_120000)')
    ap.add_argument('dir_b', help='debug dir B')
    ap.add_argument('--rtol', type=float, default=1e-10)
    ap.add_argument('--atol', type=float, default=0.0)
    args = ap.parse_args()

    dir_a = Path(args.dir_a)
    dir_b = Path(args.dir_b)

    files_a = {p.name: p for p in dir_a.glob('*.csv')}
    files_b = {p.name: p for p in dir_b.glob('*.csv')}

    missing_a = sorted(set(files_b) - set(files_a))
    missing_b = sorted(set(files_a) - set(files_b))

    if missing_a:
        print('Missing in A:')
        for name in missing_a:
            print('  ', name)
    if missing_b:
        print('Missing in B:')
        for name in missing_b:
            print('  ', name)

    common = sorted(set(files_a) & set(files_b))
    if not common:
        print('No common CSV files to compare.')
        return 1

    ok_all = True
    for name in common:
        ok, msg = compare_files(files_a[name], files_b[name], rtol=args.rtol, atol=args.atol)
        status = 'OK' if ok else 'DIFF'
        print(f"{status} {name}: {msg}")
        if not ok:
            ok_all = False

    return 0 if ok_all else 2


if __name__ == '__main__':
    raise SystemExit(main())
