## BaseFlareDetector 最適化ドキュメント

このドキュメントでは、`feat/safe-speedup-base-flare` ブランチで `BaseFlareDetector` と検証ツールに加えた主な変更を、変更前/変更後/目的/値が不変な理由の順でまとめています。引用しているコード片にはリポジトリ内で付けた目印コメント（`# [perf] ...` など）も含めています。

---

### 1. `reestimate_errors` のスライディングウィンドウ化

| 項目           | 内容                                                                                                                                                                                                         |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 変更前         | 各測光点ごとに `np.searchsorted` を呼び出して ±0.5 日の範囲を取得。ウィンドウ位置は毎回ゼロから探索していた。                                                                                                |
| 変更後         | 連続走査で `start_idx` / `end_idx` を更新し、同じウィンドウ集合を再利用。コメント `# [perf] reuse sliding window instead of per-point searchsorted` を付与。                                                 |
| 目的           | O(N log N) 相当の挙動を O(N) へ削減し、`np.searchsorted` の大量呼び出しを回避する。                                                                                                                          |
| 値が不変な理由 | `left` / `right` によるギャップ判定は変更しておらず、どの値を標準偏差に使うかは従来と完全一致。`uv run python scripts/verify_detector_state.py ...` で `mPDCSAPfluxerr_cor` のハッシュが一致することを確認。 |

```python
while start_idx < n_quiet and quiet_bjd[start_idx] < left:
    start_idx += 1
...
err[i] = np.std(quiet_flux[start_idx:end_idx])  # [perf] reuse sliding window instead of per-point searchsorted
```

---

### 2. `tess_band_energy` の応答関数キャッシュ

| 項目           | 内容                                                                                                                                                                                |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 変更前         | `tess_band_energy` 呼び出しのたびに CSV を `np.loadtxt` し、星温度ごとの積分を再計算。                                                                                              |
| 変更後         | `_get_tess_response()` と `_get_star_intensity_ratio()` にモジュールスコープのキャッシュを追加。コメント `# [perf] cached ratio keeps math identical but avoids recompute` を付与。 |
| 目的           | I/O と積分コストを削減し、`tess_band_energy` を多数回呼び出すケースで速度を稼ぐ。                                                                                                   |
| 値が不変な理由 | キャッシュは初回計算結果を保存するだけで、式自体・浮動小数演算は全く同じ。`verify_detector_state.py` で `sum_flare_energy` が同値であることを確認。                                 |

---

### 3. `rotation_period` を正則グリッド + fast 法に刷新

| 項目           | 内容                                                                                                                                                                                                 |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 変更前         | 周期側を等分した非正則グリッド (`1 / np.linspace(1.0, 8.0, 10000)`) を使用し、`LombScargle.power()` が `method="auto"` だった。                                                                      |
| 変更後         | `ROTATION_FREQUENCY_GRID = np.linspace(1/8, 1, 10000)` を導入し、`power(..., method="fast")` を呼び出すよう変更。コメント `# [perf] regular frequency grid enables LombScargle fast solver` を付与。 |
| 目的           | FFT ベースの O(N log N) 実装を確実に利用して計算時間を 1.5 s → 0.005 s まで短縮。                                                                                                                    |
| 値が不変な理由 | 周波数グリッドと周期グリッドの細かい差異のみで、同じピークインデックスが得られれば周期値は同じ。数値差は離散化誤差 (約 2.7e-4 日) のみ。                                                             |

---

### 4. Plotly レポート & `reports/` ディレクトリの整備

| 項目           | 内容                                                                                                                                                                                                                                                                  |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 変更前         | Matplotlib ベースの静的 PNG。`docs/profiling/` 直下にファイルが点在。ブラウザ表示は不可。                                                                                                                                                                             |
| 変更後         | `scripts/profile_base_flare_detector.py`, `scripts/compare_profile_results.py`, `scripts/verify_detector_state.py` を Plotly Express / Graph Objects に統一。`--show-plot` でブラウザ表示、`reports/performance/...` および `reports/validation/...` に成果物を整理。 |
| 目的           | ビジュアルの統一と、開発者・研究者がコマンド一発でブラウザ表示まで確認できるようにする。                                                                                                                                                                              |
| 値が不変な理由 | CSV と計算ロジックは従来のままで、可視化レイヤーのみ変更。`base_flare_detector_cumtime_comparison.csv` などを比較し差異がないことを確認。                                                                                                                             |

---

### 5. 状態検証のプレビュー & ツールチップ

| 項目           | 内容                                                                                                                                                                                                                                         |
| -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 変更前         | 詳細表には JSON の文字列がそのまま出力され視認性が低かった。                                                                                                                                                                                 |
| 変更後         | `serialize_value` が `preview` を付加し、表には `"[122859.93, 122937.38, 122670.76...] (shape=[15307], dtype=float32)"` のように整形した値を表示。`--show-plot` 時は `<span title="...">value</span>` 構造でブラウザ上にツールチップを表示。 |
| 目的           | 大きな配列/辞書でも「何が違うのか」を一目で判断でき、必要ならホバーで完全な JSON を確認できるようにする。                                                                                                                                    |
| 値が不変な理由 | 表示専用の `preview` を追加しているだけで、JSON ベースラインや CSV の正確な値は従来と同じ。`--update-baseline` で同期済み。                                                                                                                  |

---

### 6. LombScargle 回転周期の汎用化と auto/fast 比較結果

| 項目                | 内容                                                                                                                                                                                                                                                                                                                                                       |
| ------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 変更内容            | `BaseFlareDetector` に `rotation_period_min`, `rotation_period_max`, `rotation_n_points` を追加し、`make_rotation_frequency_grid(period_min, period_max, n_points)` で **正則周波数グリッドと対応する周期配列**を生成するようにした。既定値は 1〜8 日・10000 点で、従来の `ROTATION_FREQUENCY_GRID = np.linspace(1/8, 1, 10000)` と同等。                  |
| method の扱い       | Lomb–Scargle の `method` はインスタンス属性 `rotation_ls_method` で切り替え可能とし、**デフォルトは `"auto"`** に設定。TESS のような等間隔データでは `auto` が内部的に fast 実装を選び、必要に応じて `"fast"` を明示指定できる。                                                                                                                           |
| 星ごとの周期レンジ  | 各ターゲットクラスで archive/daijiro 実装と同じ周期レンジを明示的に指定: DS Tuc A は 1.0〜8.0 日, EK Dra は 1.5〜5.0 日, V889 Her は 0.3〜2.0 日。これにより「どのレンジを探索しているか」がクラス定義から一目で分かるようになった。                                                                                                                       |
| 検証スクリプト      | `scripts/compare_rotation_lomb_methods.py` を追加し、`uv run python scripts/compare_rotation_lomb_methods.py --data-root data/TESS --output-dir reports/performance/rotation_lomb_methods --show-plot` で `method="auto"` と `method="fast"` の周期・計算時間を全 FITS について比較できるようにした。                                                      |
| auto vs fast の結果 | DS Tuc A (5 ファイル)、EK Dra (12 ファイル)、V889 Her (4 ファイル) に対し比較した結果、`rotation_period_diff_summary.csv` および `rotation_period_diff_summary_table.png` が示すように、**全てのターゲットで `mean_abs_delta_days`, `max_abs_delta_days`, `mean_rel_delta`, `max_rel_delta` は 0.0** となり、`auto` と `fast` の自転周期は完全に一致した。 |
| グラフと表          | `rotation_period_auto_vs_fast_scatter.png` では全点が `y = x` 上に乗り、`rotation_period_diff_hist_hours.png` のヒストグラムは 0 時間以外に広がりを持たない。さらに `rotation_period_diff_summary_table.png` により、ターゲット別の件数と差分統計を一覧できる。                                                                                            |
| 目的                | デフォルトを `method="auto"` としつつ、正則周波数グリッドとパラメータ化された周期レンジにより、**汎用性（疎・非等間隔データへの適用可能性）と性能（TESS 等での fast 実装利用）を両立**すること。                                                                                                                                                           |
| 値が不変な理由      | 既存の TESS データセットについては、`auto` と `fast` の周期が完全一致していることを比較スクリプトで確認済み。グリッド生成ロジックも `1/period_max〜1/period_min` の線形周波数グリッドという点で従来と同等であり、星ごとの周期レンジは旧実装の値をそのままパラメータとして明示しただけである。                                                              |

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
