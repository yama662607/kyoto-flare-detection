# Method（詳細版 / Kyoto Flare Detection）

## 0. 本節の位置づけ

本節は、本プロジェクトの現行実装（`src/base_flare_detector.py` および `src/flarepy_*.py`）に基づき、TESS の光度曲線からフレアを検出し、フレアエネルギー・発生頻度・回転周期・スポット指標を推定するための解析手順を、**再現可能な粒度**で記述する。

- **共通処理**: `src/base_flare_detector.py`（`BaseFlareDetector`）
- **星固有の上書き**:
  - DS Tuc A: `src/flarepy_DS_Tuc_A.py`（`FlareDetector_DS_Tuc_A`）
  - EK Dra: `src/flarepy_EK_Dra.py`（`FlareDetector_EK_Dra`）
  - V889 Her: `src/flarepy_V889_Her.py`（`FlareDetector_V889_Her`）

以降、時刻は TESS FITS の `time` 列（BJD 系）を `t`、正規化フラックスを `f(t)` と記す。配列上のインデックスを `i` とし、`t_i`・`f_i` はそれぞれ時刻配列・フラックス配列の `i` 番目を表す。

## 1. 解析パイプライン（関数と処理順）

`BaseFlareDetector.process_data(ene_thres_low=None, ene_thres_high=None, skip_remove=False)` は以下を順に実行する。

1. `remove()`（派生クラスで上書きされる場合がある。`skip_remove=True` の場合は省略）
2. `apply_gap_correction()`（ギャップ補正 + バッファ拡張）
3. `detrend_flux()`（回転変動等の低周波成分を除去し、基線を推定）
4. `reestimate_errors()`（検出に用いる誤差系列を再推定）
5. `flaredetect()`（一次検出：候補抽出 + 粗いイベント区間 + 粗い積分）
6. `flaredetect_check()`（再検証：局所基線でイベントを再定義 + 減衰時定数 + 精密積分）
7. `calculate_precise_obs_time()`（有効観測時間の推定）
8. `flare_energy(energy_threshold_low, energy_threshold_high)`（エネルギー範囲でイベント数と総エネルギーを集計）
9. `flux_diff()`（回転変動振幅とスポット指標の推定）
10. `rotation_period()`（Lomb–Scargle による回転周期推定）

## 2. 入力データ・前処理

### 2.1 入力（TESS FITS light curve）

`BaseFlareDetector.load_TESS_data()` は入力 FITS（`*_lc.fits`）を `astropy.io.fits.open(..., memmap=True)` で読み込み、拡張 1（`hdulist[1].data`）から以下の列を用いる。

- **時刻**: `time`
- **フラックス**: `PDCSAP_FLUX`（既定）
- **フラックス誤差**: `PDCSAP_FLUX_ERR`（既定）

フラックス列の欠損（NaN）は除外する。

### 2.2 セクタ閾値によるフラックス列の切替（PDCSAP → SAP）

ファイル名 `*_lc.fits` からセクタ番号を抽出し、派生クラスで `sector_threshold` が指定されている場合に限り、

- セクタ番号 > `sector_threshold`

なら `SAP_FLUX` / `SAP_FLUX_ERR` を使用する。それ以外は `PDCSAP_FLUX` / `PDCSAP_FLUX_ERR` を用いる。

### 2.3 フラックス正規化

星ごとに与える `flux_mean` を用い、

- `f_i = flux_i / flux_mean`
- `σ_i = flux_err_i / flux_mean`

として正規化する。以降の解析はこの正規化系列で行う。

## 3. ギャップ補正とバッファ拡張

### 3.1 ギャップ補正（オフセット補正による連続化）

時刻差分 `Δt_i = t_{i+1} - t_i` を計算し、

- `Δt_i >= gap_threshold`

を満たす `i` をギャップとして検出する。各ギャップ位置 `i` について、ギャップ直後のフラックス列全体を

- `f_{i+1:} ← f_{i+1:} - (f_{i+1} - f_i)`

でシフトし、ギャップ前後での不連続（段差）を補正する。

`gap_threshold` は `BaseFlareDetector` 既定では `0.1` だが、派生クラスが星ごとに上書きする（表 1）。

### 3.2 バッファ拡張（端点影響の低減）

デトレンドの端点影響を低減するため、長さ `buffer_size` のバッファを前後に付加する。

- フラックス拡張: `f_ext = [f_0 を buffer_size 点] + f + [f_{N-1} を buffer_size 点]`
- 誤差拡張: `σ_ext = [0.0001 を buffer_size 点] + σ + [0.0001 を buffer_size 点]`

時刻も `dt_min = 2/(24*60)` 日（=2 分）を刻みとして線形外挿し、拡張時刻列 `t_ext` を作成する。これらを

- `gtessBJD ← t_ext`
- `gmPDCSAPflux ← f_ext`
- `gmPDCSAPfluxerr ← σ_ext`

として保持する。

## 4. デトレンド（低周波成分除去と基線推定）

### 4.1 FFT によるローパスフィルタ

`lowpass(x, y, fc)` は FFT を用いて `y` の周波数成分のうち、カットオフ `fc` より高い周波数の成分を 0 にするローパスフィルタである。サンプリング間隔は実装上固定で

- `dt = 2/(24*60)` 日

を用いる。周波数軸 `freq` を `linspace(0, 1/dt, n)` として構成し、`freq > fc` の成分をゼロ化して逆変換する。

### 4.2 スプライン基線によるデトレンド（共通実装）

`BaseFlareDetector.detrend_flux()` は以下で基線を推定し、デトレンド系列 `f_det`（実装名 `s2mPDCSAPflux`）を生成する。

1. ローパス成分（低周波）: `LP(t) = lowpass(t_ext, f_ext, fc=f_cut_lowpass)`
2. 残差: `r(t) = f_ext(t) - LP(t)`
3. 基線推定に用いる候補点集合 `S` を
   - `fac = 3`
   - `S = { i | r_i <= σ_ext,i * fac } ∪ { i | t_ext,i < t_ext,10 } ∪ { i | t_ext,i > t_ext,N-11 }`
     として定義（フレア候補点を基線推定から除外する意図）
4. `S` の点で `lowpass(t_ext[S], f_ext[S], fc=f_cut_spline)` を計算し、その系列に対して cubic の `interp1d(..., kind="cubic")` によるスプライン `B(t)` を構成
5. 元の区間（バッファを除いた区間）を `valid_slice = slice(buffer_size, buffer_size + len(tessBJD))` とし
   - `flux_spline = B(t_ext[valid_slice])`
   - `f_det = f_ext[valid_slice] - flux_spline`

ここで `f_det`（`s2mPDCSAPflux`）が以降のフレア検出に用いられる。

### 4.3 V889 Her に対する特別なデトレンド（フレア区間マスク補間）

`FlareDetector_V889_Her.detrend_flux()` は、上記 4.2 の基線推定の前に、急峻変動を示す区間を「フレアらしい」区間としてマスクし、マスク区間をスプライン補間してから基線推定を行う。

- 差分: `Δf_i = f_{i+1} - f_i`, `Δt_i = t_{i+1} - t_i`
- ラグ差分（n=2..5）も計算し、しきい値 `0.01` を超える点をフレア開始候補とする
- さらに `Δt_i < 0.005` を条件として、時間差が大きい点を候補から除外する
- 各開始候補 `j` に対し、`|f_j - f_i| < 0.008` を満たす最初の `i (>= j+5)` を終端候補として採用し、区間 `[j, i]` をマスク
- マスク外の点で cubic スプラインを構成し、マスク区間を補間

その後、補間系列を用いて 4.2 と同様の lowpass + spline 基線推定を行い `f_det` を作る。

## 5. 誤差系列の再推定

`BaseFlareDetector.reestimate_errors()` は、デトレンド系列 `f_det` のうち

- `f_det <= 0.005`

を「静穏（quiet）」とみなし、各時刻 `t_i` の周辺 ±0.5 日（`window=0.5`）の静穏点から局所的な散布（標準偏差）を推定して誤差系列 `σ_i^{(re)}`（実装名 `mPDCSAPfluxerr_cor`）を構成する。

- 窓内の静穏点が 0 の場合: `σ_i^{(re)} = NaN`
- 窓内が 1 点のみの場合: `σ_i^{(re)} = 0.0`
- それ以外は prefix-sum を用いた分散推定から `σ_i^{(re)} = sqrt(var)`

最後に、元の誤差配列 `mPDCSAPfluxerr` の平均と、星ごとに与える `err_constant_mean` の比でスケールする。

- `σ_i^{(re)} ← σ_i^{(re)} * mean(mPDCSAPfluxerr) / err_constant_mean`

## 6. フレア候補の一次検出と粗いイベント定義

以降、デトレンド系列を `f_det,i`、再推定誤差を `σ_i^{(re)}` と書く。

### 6.1 一次検出（5σ 相当）

`flaredetect()` は

- `f_det,i >= 5 * σ_i^{(re)}`

を満たすインデックス集合 `I_5σ` を抽出し、隣接点（インデックス差が 1）を同一群とみなす。各群の先頭インデックスを「検出種点」として `ss_detect` に登録する。

### 6.2 イベント区間（1σ 以上の連続領域）

次に

- `f_det,i >= 1 * σ_i^{(re)}`

を満たすインデックス集合 `I_1σ` を定義し、各 `ss_detect` を含む `I_1σ` 上の連続区間を前後に拡張してイベント区間候補を得る。

#### 6.2.1 端点除外

以下に該当するイベントは除外する。

- 区間が時系列端に近い: `(n_start <= 30)` または `(n_end >= N-30)`

（実装は `n + j <= 30` または `n + k >= len(bjd)-30`）

#### 6.2.2 大ギャップ近傍の除外

イベント近傍の `diff_bjd` の局所スライス `a = diff_bjd[(n+j-10):(n+k+10)]` を取り、

- `max(a) >= (2/(24*60))*20`

の場合は除外する。これは **約 40 分以上の大きなギャップ**がイベント近傍に存在するケースを排除する。

### 6.3 ピーク時刻と粗い積分

採用されたイベント区間について

- `t_start`: 区間開始時刻
- `t_end`: 区間終了時刻
- `t_peak`: 区間内で `f_det` が最大となる時刻

を定義し、粗い積分量 `count` を

- `count = Σ_{i=n_start..n_end} f_det,i`

として計算する。

## 7. フレア再検証（イベント再定義と品質フィルタ）

`flaredetect_check()` は、一次検出イベントごとに局所基線を推定し、イベント区間と積分量を再定義する。

### 7.1 局所線形基線の推定

元の正規化フラックス系列（デトレンド前）を `f_raw,i`（実装名 `mPDCSAPflux`）、元の誤差 `σ_raw,i`（`mPDCSAPfluxerr`）とする。

イベント `k` の一次検出区間の開始・終了を `t_start(k)`, `t_end(k)` としたとき、以下の前後窓でフラックス中央値を計算する。

- 前窓: `t_start(k) - 0.025` の近傍（条件 `|t - (t_start-0.025)| <= 0.0125`）
- 後窓: `t_end(k) + 0.05` の近傍（条件 `|t - (t_end+0.05)| <= 0.025`）

それぞれの中央値を `val_pre`, `val_post`、その時刻中央値を `t_pre`, `t_post` とし、局所線形基線

- `b(t) = a t + c`

を

- `a = (val_post - val_pre)/(t_post - t_pre)`
- `c = val_pre - a t_pre`

で求める。

### 7.2 基線差分系列とイベント区間の再定義

基線差分

- `f_diff,i = f_raw,i - (a t_i + c)`

を定義し、ピーク時刻 `t_peak(k)` に一致するインデックス `n_peak` から前後に

- `f_diff,i >= σ_raw,i`

を満たす範囲へ拡張して、新しい `n_start`, `n_end` を決定する。

### 7.3 採用条件（実装上の品質フィルタ）

新しいイベント区間に対し、以下を満たす場合のみイベントを採用する。

1. 区間長が 2 点以上（`len(ss_flare) > 1`）
2. 区間内で
   - `f_diff,i - 3*σ_raw,i >= 0`
     を満たす点が 2 点以上

すなわち、**3σ を 2 点以上**で満たすイベントのみ最終採用とする。

### 7.4 減衰時定数（e-folding time）

区間内のピーク `f_peak = max(f_diff)` を求め、ピーク位置 `t_peak_loc` から

- `f_diff(t) >= f_peak * e^{-1}`

を満たす最後の時刻までの差を

- `t_decay = t_{e^{-1}} - t_peak_loc`

として `edecay` に保存する。

### 7.5 精密積分量

最終採用イベントについて、精密積分量を

- `count_new = Σ_{i=n_start..n_end} f_diff,i`

として計算し、エネルギー推定に用いる。

## 8. フレアエネルギー推定（TESS 帯）

### 8.1 TESS 応答関数

TESS 応答関数は CSV `data/tess-response-function-v1.0.csv`（`TESS_RESPONSE_PATH`）から読み込み、

- `wave`（波長）
- `resp`（応答）
- `dw = [diff(wave), 0]`

を構成する。

### 8.2 黒体強度（Planck）

`planck(λ, T)` を用いて黒体強度を計算する（実装は SI 定数での Planck 関数）。波長は `wave*1e-9`（nm→m）として与える。

### 8.3 共通実装（BaseFlareDetector）

`BaseFlareDetector.tess_band_energy(count)` は次でフレアエネルギーを推定する。

- 時間刻み: `Δt = 120.0` 秒
- 恒星半径: `R = 695510e5 * R_sunstar_ratio`（cm）
- ステファン・ボルツマン定数（実装値）: `σ_SB = 5.67e-5`

応答で重み付けした強度

- `I_star = Σ (dw * planck(wave, T_star) * resp)`
- `I_ref = Σ (dw * planck(wave, 10000) * resp)`

から

- `η = I_star / I_ref`
- `A = (π R^2) * η`

を定義し、

- `E = σ_SB * (10000^4) * A * Δt * count`

を返す。

`count` は `flaredetect`（粗い積分）または `flaredetect_check`（精密積分）により得られる。最終的には `flaredetect_check` の `count_new` に基づく値が採用される。

### 8.4 DS Tuc A（伴星込みの上書き）

`FlareDetector_DS_Tuc_A.tess_band_energy(count)` は、伴星の寄与を含めるため、主星と伴星の強度を面積加重で合成する。

- 主星半径: `R_primary = R_sun * R_sunstar_ratio`
- 伴星半径: `R_companion = R_sun * 0.864`
- 伴星温度: `T_companion = 4700`

強度

- `I_main = Σ (dw * planck(wave, T_star) * resp)`
- `I_comp = Σ (dw * planck(wave, 4700) * resp)`
- `I_ref = Σ (dw * planck(wave, 10000) * resp)`

を用い

- `I_area = I_main * R_primary^2 + I_comp * R_companion^2`
- `A = π * (I_area / I_ref)`

として

- `E = σ_SB * (10000^4) * A * Δt * count`

を返す。

## 9. 有効観測時間

`calculate_precise_obs_time()` は

- `Δt_i >= 0.2` 日

を観測ギャップとして扱い、

- `T_obs = (t_{N-1} - t_0) - Σ(gap durations)`

で有効観測時間 `precise_obs_time` を計算する。

## 10. エネルギー範囲に基づくフレア数と総エネルギー

`flare_energy(energy_threshold_low, energy_threshold_high)` はエネルギー配列 `E_k` を昇順にソートし、

- `E_low <= E_k <= E_high`

を満たすイベント数 `N_flare`（実装名 `flare_number`）と、その総和 `ΣE`（`sum_flare_energy`）を計算する。

しきい値は `process_data()` の引数 `ene_thres_low` / `ene_thres_high` が与えられればそれを採用し、与えられなければインスタンスの既定値 `self.ene_thres_low` / `self.ene_thres_high`（既定 5e33〜2e40 erg）を用いる。

## 11. 回転変動振幅とスポット指標

### 11.1 回転変動振幅

`flux_diff(min_percent=0.02, max_percent=0.98)` は正規化フラックス `f_raw` をソートし、

- `A_rot = P98(f_raw) - P2(f_raw)`

を回転変動振幅（`brightness_variation_amplitude`）として定義する。

### 11.2 温度低下量（スポット温度差）

実装は温度低下量を

- `ΔT = 3.58e-5*T_star^2 + 0.249*T_star - 808`

で与える（`d_T_star`）。

### 11.3 スポット指標

実装は以下をスポット指標として計算する。

- `starspot_ratio = (T^4/(T^4 - (T-ΔT)^4)) * A_rot`
- `starspot = 2π (R_sunstar_ratio * 695510e3)^2 * starspot_ratio`

（単位系・係数は実装に従う。）

### 11.4 DS Tuc A の上書き

`FlareDetector_DS_Tuc_A.flux_diff()` は `super().flux_diff()` 実行後、主星+伴星の面積を用いる形で `starspot` を再計算する。

## 12. 回転周期推定（Lomb–Scargle）

`rotation_period()` は `LombScargle(tessBJD - tessBJD[0], mPDCSAPflux)` に対し、周波数グリッドを

- `f_min = 1/period_max`
- `f_max = 1/period_min`
- `N_f = rotation_n_points`（既定 10000）

で一様に構成し、パワースペクトル `P(f)` を計算する（`assume_regular_frequency=True`）。最大パワーの周波数 `f*` に対応する周期

- `P_rot = 1/f*`

を回転周期 `per` とする。

誤差 `per_err` は、半値幅（`P(f) > max(P)/2` を満たす区間の端点周期差の半分）として定義する。

## 13. 星固有パラメータ（現行 `src/flarepy_*.py`）

表 1 に、現行実装でハードコードされている星固有パラメータを示す。

| Star / Detector                     | `R_sunstar_ratio` | `T_star` [K] | `flux_mean` | `err_constant_mean` | `rot_period` [day] | Rotation search [`min`, `max`] [day] | `f_cut_lowpass` | `f_cut_spline` | `sector_threshold` | `gap_threshold` | 特記事項                                                                                        |
| ----------------------------------- | ----------------: | -----------: | ----------: | ------------------: | -----------------: | ------------------------------------ | --------------: | -------------: | -----------------: | --------------: | ----------------------------------------------------------------------------------------------- |
| DS Tuc A (`FlareDetector_DS_Tuc_A`) |              0.87 |         5428 | 119633.9953 |        0.0005505874 |          0.3672258 | [1.0, 8.0]                           |               3 |              6 |                 74 |            0.05 | `remove()`（トランジット区間除去）、`tess_band_energy()`（伴星込み）、`flux_diff()`（面積補正） |
| EK Dra (`FlareDetector_EK_Dra`)     |              0.94 |         5700 | 249320.3537 |        0.0004111605 |          0.2094793 | [1.5, 5.0]                           |               3 |              6 |                 74 |             0.2 | ギャップ閾値のみ上書き                                                                          |
| V889 Her (`FlareDetector_V889_Her`) |              1.00 |         6550 | 300710.6233 |        0.0003969586 |          0.4398277 | [0.3, 2.0]                           |              30 |             40 |                 90 |           0.004 | `detrend_flux()` を独自実装（フレア区間マスク補間 + lowpass/spline）                            |

## 14. 実装・再現性情報

### 14.1 主要依存

`pyproject.toml` に基づく主要依存は以下である。

- Python: `>=3.13`
- `astropy`（FITS 読み込み、Lomb–Scargle）
- `numpy`（FFT、統計、配列）
- `scipy`（`interp1d`）
- `matplotlib` / `plotly`（可視化）

### 14.2 実行方法（概略）

通常は `BaseFlareDetector` または派生クラスを `file=...` で生成し、`process_data()` を呼ぶことで一連の解析が実行される。

## 付録 A. 本詳細 Method とコードの対応

- **Data / Normalization**: `load_TESS_data()`
- **Gap correction**: `apply_gap_correction()`
- **Detrending**: `detrend_flux()`（V889 Her は `FlareDetector_V889_Her.detrend_flux()`）
- **Error model**: `reestimate_errors()`
- **Flare detection**: `flaredetect()`, `flaredetect_check()`
- **Energy estimation**: `tess_band_energy()`（DS Tuc A は上書き）
- **Observation time**: `calculate_precise_obs_time()`
- **Rotation period**: `rotation_period()`
- **Spot proxy**: `flux_diff()`（DS Tuc A は上書き）

## 付録 B. 既存簡易版との関係

- 本ファイルは `docs/method_section_current_project.md`（簡易版）を基に、**閾値・窓幅・除外条件・イベント定義**を明文化し、論文の Method 節としてそのまま使用できるように詳細化したものである。
