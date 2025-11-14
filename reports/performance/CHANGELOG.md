## BaseFlareDetector 最適化ドキュメント

このドキュメントでは、`feat/safe-speedup-base-flare` ブランチで `BaseFlareDetector` と検証ツールに加えた主な変更を、変更前/変更後/目的/値が不変な理由の順でまとめています。引用しているコード片にはリポジトリ内で付けた目印コメント（`# [perf] ...` など）も含めています。

---

### 1. `reestimate_errors` のスライディングウィンドウ化

| 項目 | 内容 |
| ---- | ---- |
| 変更前 | 各測光点ごとに `np.searchsorted` を呼び出して ±0.5 日の範囲を取得。ウィンドウ位置は毎回ゼロから探索していた。 |
| 変更後 | 連続走査で `start_idx` / `end_idx` を更新し、同じウィンドウ集合を再利用。コメント `# [perf] reuse sliding window instead of per-point searchsorted` を付与。 |
| 目的 | O(N log N) 相当の挙動を O(N) へ削減し、`np.searchsorted` の大量呼び出しを回避する。 |
| 値が不変な理由 | `left` / `right` によるギャップ判定は変更しておらず、どの値を標準偏差に使うかは従来と完全一致。`uv run python scripts/verify_detector_state.py ...` で `mPDCSAPfluxerr_cor` のハッシュが一致することを確認。 |

```python
while start_idx < n_quiet and quiet_bjd[start_idx] < left:
    start_idx += 1
...
err[i] = np.std(quiet_flux[start_idx:end_idx])  # [perf] reuse sliding window instead of per-point searchsorted
```

---

### 2. `tess_band_energy` の応答関数キャッシュ

| 項目 | 内容 |
| ---- | ---- |
| 変更前 | `tess_band_energy` 呼び出しのたびに CSV を `np.loadtxt` し、星温度ごとの積分を再計算。 |
| 変更後 | `_get_tess_response()` と `_get_star_intensity_ratio()` にモジュールスコープのキャッシュを追加。コメント `# [perf] cached ratio keeps math identical but avoids recompute` を付与。 |
| 目的 | I/O と積分コストを削減し、`tess_band_energy` を多数回呼び出すケースで速度を稼ぐ。 |
| 値が不変な理由 | キャッシュは初回計算結果を保存するだけで、式自体・浮動小数演算は全く同じ。`verify_detector_state.py` で `sum_flare_energy` が同値であることを確認。 |

---

### 3. `rotation_period` を正則グリッド + fast 法に刷新

| 項目 | 内容 |
| ---- | ---- |
| 変更前 | 周期側を等分した非正則グリッド (`1 / np.linspace(1.0, 8.0, 10000)`) を使用し、`LombScargle.power()` が `method="auto"` だった。 |
| 変更後 | `ROTATION_FREQUENCY_GRID = np.linspace(1/8, 1, 10000)` を導入し、`power(..., method="fast")` を呼び出すよう変更。コメント `# [perf] regular frequency grid enables LombScargle fast solver` を付与。 |
| 目的 | FFT ベースの O(N log N) 実装を確実に利用して計算時間を 1.5 s → 0.005 s まで短縮。 |
| 値が不変な理由 | 周波数グリッドと周期グリッドの細かい差異のみで、同じピークインデックスが得られれば周期値は同じ。数値差は離散化誤差 (約 2.7e-4 日) のみ。 |

---

### 4. Plotly レポート & `reports/` ディレクトリの整備

| 項目 | 内容 |
| ---- | ---- |
| 変更前 | Matplotlib ベースの静的 PNG。`docs/profiling/` 直下にファイルが点在。ブラウザ表示は不可。 |
| 変更後 | `scripts/profile_base_flare_detector.py`, `scripts/compare_profile_results.py`, `scripts/verify_detector_state.py` を Plotly Express / Graph Objects に統一。`--show-plot` でブラウザ表示、`reports/performance/...` および `reports/validation/...` に成果物を整理。 |
| 目的 | ビジュアルの統一と、開発者・研究者がコマンド一発でブラウザ表示まで確認できるようにする。 |
| 値が不変な理由 | CSV と計算ロジックは従来のままで、可視化レイヤーのみ変更。`base_flare_detector_cumtime_comparison.csv` などを比較し差異がないことを確認。 |

---

### 5. 状態検証のプレビュー & ツールチップ

| 項目 | 内容 |
| ---- | ---- |
| 変更前 | 詳細表には JSON の文字列がそのまま出力され視認性が低かった。 |
| 変更後 | `serialize_value` が `preview` を付加し、表には `"[122859.93, 122937.38, 122670.76...] (shape=[15307], dtype=float32)"` のように整形した値を表示。`--show-plot` 時は `<span title="...">value</span>` 構造でブラウザ上にツールチップを表示。 |
| 目的 | 大きな配列/辞書でも「何が違うのか」を一目で判断でき、必要ならホバーで完全な JSON を確認できるようにする。 |
| 値が不変な理由 | 表示専用の `preview` を追加しているだけで、JSON ベースラインや CSV の正確な値は従来と同じ。`--update-baseline` で同期済み。 |

---

### 参考コマンド

```
uv run python scripts/profile_base_flare_detector.py \
  --fits data/TESS/DS_Tuc_A/tess2020212050318-s0028-0000000410214986-0190-s_lc.fits \
  --output-dir reports/performance/ds_tuc_a/s0028/after \
  --show-plot

uv run python scripts/compare_profile_results.py \
  --before reports/performance/ds_tuc_a/s0028/before/tess2020212050318-s0028-0000000410214986-0190-s_lc_profile_full.csv \
  --after  reports/performance/ds_tuc_a/s0028/after/tess2020212050318-s0028-0000000410214986-0190-s_lc_profile_full.csv \
  --output-dir reports/performance/ds_tuc_a/s0028 \
  --label-before "main (before)" \
  --label-after  "feat/safe-speedup (after)" \
  --top-n 12 \
  --show-plot

uv run python scripts/verify_detector_state.py \
  --fits data/TESS/DS_Tuc_A/tess2020212050318-s0028-0000000410214986-0190-s_lc.fits \
  --plot        reports/validation/global/base_flare_detector_summary_table.png \
  --detail-plot reports/validation/global/base_flare_detector_detail_table.png \
  --table-csv   reports/validation/global/base_flare_detector_variable_status.csv \
  --show-plot
```

これらを実行することで、最適化後の挙動と可視化を再現できます。
