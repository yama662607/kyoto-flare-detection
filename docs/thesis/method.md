# Method（日本語版 / Kyoto Flare Detection）

## 1. 概要（解析パイプライン）

本研究では、TESS の光度曲線（FITS light curve; `*_lc.fits`）からフレアを自動検出し、フレアエネルギー、発生頻度、回転周期、スポット指標などを推定する。解析は `src/base_flare_detector.py` の `BaseFlareDetector.process_data()` に定義されたパイプラインに従う。

- **実装の一次ソース**
  - **共通処理**: `src/base_flare_detector.py`（`BaseFlareDetector`）
  - **星固有の上書き**:
    - `src/flarepy_DS_Tuc_A.py`（`FlareDetector_DS_Tuc_A`）
    - `src/flarepy_EK_Dra.py`（`FlareDetector_EK_Dra`）
    - `src/flarepy_V889_Her.py`（`FlareDetector_V889_Her`）

### 1.1 `process_data()` の処理順

`BaseFlareDetector.process_data(ene_thres_low=None, ene_thres_high=None, skip_remove=False)` は以下を順に実行する。

1. `remove()`（派生クラスで上書きされることがある。`skip_remove=True` の場合は省略）
2. `apply_gap_correction()`（ギャップ補正とバッファ拡張）
3. `detrend_flux()`（低周波変動除去と基線推定）
4. `reestimate_errors()`（誤差推定の再計算）
5. `flaredetect()`（一次検出・イベント区間推定・粗いエネルギー積算）
6. `flaredetect_check()`（再検証・イベント区間再推定・減衰時定数推定・精密積算）
7. `calculate_precise_obs_time()`（有効観測時間の推定）
8. `flare_energy(energy_threshold_low, energy_threshold_high)`（エネルギー範囲でイベント数と総エネルギーを集計）
9. `flux_diff()`（回転変動振幅・スポット指標推定）
10. `rotation_period()`（Lomb–Scargle による回転周期推定）

## 2. データと正規化

### 2.1 入力データ（TESS FITS Light Curve）

`BaseFlareDetector.load_TESS_data()` では、入力 `file`（FITS）を `astropy.io.fits.open(..., memmap=True)` で読み込み、拡張 1（`hdulist[1].data`）から以下の列を利用する。

- **時刻**: `time`
- **フラックス**: 既定は `PDCSAP_FLUX`（後述の条件により `SAP_FLUX` に切替）
- **フラックス誤差**: `PDCSAP_FLUX_ERR` または `SAP_FLUX_ERR`

欠損値は `~np.isnan(flux)` で除外する。

### 2.2 セクタ閾値によるフラックス列の切替

ファイル名 `*_lc.fits` からセクタ番号を抽出し、派生クラスで指定される `sector_threshold` より大きい場合のみ、以下へ切り替える。

- `flux_field = "SAP_FLUX"`
- `flux_err_field = "SAP_FLUX_ERR"`

それ以外は `PDCSAP_FLUX` / `PDCSAP_FLUX_ERR` を使用する。

### 2.3 観測フラックスの正規化

読み込んだフラックス `pdcsap_flux` と誤差 `pdcsap_flux_err` を、星ごとに与える `flux_mean` で正規化する。

- `norm_flux = pdcsap_flux / flux_mean`
- `norm_flux_err = pdcsap_flux_err / flux_mean`

これを以降の解析（検出・デトレンド・回転周期推定など）に用いる。

## 3. ギャップ補正とバッファ拡張

### 3.1 ギャップ補正（フラックスの連続化）

`BaseFlareDetector.apply_gap_correction()` では、時刻列 `bjd = self.tessBJD` の差分 `diff_bjd = np.diff(bjd)` を計算し、

- `diff_bjd >= gap_threshold`

となるインデックスをギャップとみなし、ギャップ直後のフラックス列をギャップ直前に合わせるよう **オフセット補正**する。

- ギャップ位置 `idx` ごとに
  - `flux[idx+1:] -= flux[idx+1] - flux[idx]`

`gap_threshold` は `BaseFlareDetector` の既定値 `0.1` だが、派生クラスで星ごとに上書きされる（後述）。

### 3.2 バッファ拡張（端点影響の低減）

デトレンドのため、フラックス配列の前後に `buffer_size` 点のバッファを付加する。

- `flux_ext = [flux[0] を buffer_size 点] + flux + [flux[-1] を buffer_size 点]`
- `flux_err_ext = [0.0001 を buffer_size 点] + flux_err + [0.0001 を buffer_size 点]`

時刻も `dt_min = 2 / (24*60)` 日（= 2 分）を基準に線形に外挿して `bjd_ext` を構築し、

- `self.gtessBJD = bjd_ext`
- `self.gmPDCSAPflux = flux_ext`
- `self.gmPDCSAPfluxerr = flux_err_ext`

として保持する。

## 4. デトレンド（基線推定と低周波除去）

### 4.1 低域通過フィルタ（FFT low-pass）

`BaseFlareDetector.lowpass(x, y, fc)` は FFT により周波数領域でカットオフ `fc`（単位は実装上 `day^{-1}` 相当）を超える成分を 0 にし、逆変換した実数部を返す。

- サンプリング間隔: `dt = 2/(24*60)` 日（=2 分）
- 周波数軸: `freq = linspace(0, 1/dt, n)`

### 4.2 基線推定（スプライン）

`BaseFlareDetector.detrend_flux()` は以下で基線を推定し、デトレンドフラックスを生成する。

1. 拡張列に low-pass を適用: `filtered_flux = lowpass(time_ext, flux_ext, fc=f_cut_lowpass)`
2. 残差: `s1_flux = flux_ext - filtered_flux`
3. 基線推定に用いる点集合 `ss_flarecan` を定義（フレア候補を除外する意図）
   - `fac = 3`
   - `ss_flarecan = where((s1_flux <= flux_err_ext*fac) | (time_ext < time_ext[10]) | (time_ext > time_ext[-11]))`
4. 基線スプライン
   - `baseline_spline = interp1d(time_ext[ss_flarecan], lowpass(time_ext[ss_flarecan], flux_ext[ss_flarecan], fc=f_cut_spline), kind="cubic")`
5. 元の長さの区間 `valid_slice = slice(buffer_size, buffer_size + len(self.tessBJD))` に対し
   - `flux_spline = baseline_spline(time_ext[valid_slice])`
   - `s2mPDCSAPflux = flux_ext[valid_slice] - flux_spline`

ここで `s2mPDCSAPflux` が、フレア検出に用いるデトレンド系列である。

### 4.3 V889 Her の特別なデトレンド

`FlareDetector_V889_Her.detrend_flux()` は、上記の基線推定の前に **急峻変動（フレアらしさ）をもつ区間をマスクして補間**し、回転変動の推定にフレアが混入するのを抑える。

- `diff_time = diff(time_ext)`
- `diff_flux = diff(flux_ext)`
- さらにラグ差分（n=2..5）を計算し、閾値 `0.01` を超える点を候補とする
- 候補 `j` から先で `abs(flux_ext[j] - flux_ext[i]) < 0.008` となる `i` を探索し、`j..i` をマスク
- マスク外の点で cubic spline を作り、マスク区間のフラックスを補間

その後、補間後系列に対して `f_cut_lowpass` / `f_cut_spline` を用いた lowpass + spline 推定を行い、`s2mPDCSAPflux` を構成する。

## 5. 誤差推定の再計算

`BaseFlareDetector.reestimate_errors()` は、デトレンド後系列 `s2mPDCSAPflux` のうち

- `quiet_mask = (flux <= 0.005)`

を「静穏」とみなし、時刻 `bjd[i]` ごとに ±0.5 日 (`window=0.5`) の静穏点を集めて分散を推定し、誤差配列 `mPDCSAPfluxerr_cor` を得る。

- 近傍サンプルが 0 の場合: `err[i]=NaN`
- サンプルが 1 の場合: `err[i]=0.0`
- それ以外: prefix-sum により分散を計算し `sqrt(var)`

最後に

- `err *= mean(self.mPDCSAPfluxerr) / self.err_constant_mean`

でスケーリングする。

## 6. フレア検出（一次検出とイベント定義）

### 6.1 一次検出（5σ 相当）

`BaseFlareDetector.flaredetect()` において、デトレンド系列 `flux_detrend = s2mPDCSAPflux` と再推定誤差 `err = mPDCSAPfluxerr_cor` を用い、

- `oversigma_idx = where(flux_detrend >= err*5)`

を検出候補点とする。

候補は「隣接点が連続する（インデックス差が 1）」ことを条件にイベント種点 `ss_detect` として採用する。

### 6.2 イベント区間（1σ 以上の連続領域）

`overonesigma_idx = where(flux_detrend >= err)` を定義し、各 `ss_detect` について `overonesigma_idx` 上の連続領域を前後に拡張する。

- 連続領域の始端・終端がデータ端に近すぎるイベントは除外
  - `(n + j) <= 30` または `(n + k) >= len(bjd) - 30` の場合は除外
- さらに、イベント周辺の時刻差分 `a = diff_bjd[(n+j-10):(n+k+10)]` に対し
  - `max(a) >= (2/(24*60))*20` の場合は除外
  - これは **約 40 分以上の大きなギャップ**がフレア近傍にあるケースを排除する実装である。

イベント区間が採用された場合

- `starttime = bjd[n+j]`
- `endtime = bjd[n+k]`
- `peaktime` は区間内の `flux_detrend` 最大点

とする。

### 6.3 粗いフラックス積算（count）

イベント区間内で

- `count = sum(flux_detrend[n+j : n+k+1])`

を計算し、次節のエネルギー推定に入力する。

## 7. フレア再検証（`flaredetect_check`）

一次検出後、`BaseFlareDetector.flaredetect_check()` は各イベントについて局所的な線形基線を推定し、フレア区間と物理量を再定義する。

### 7.1 局所線形基線（`a_i`, `b_i`）

- `t_pre`: `stime - 0.025` 日付近（±0.0125 日）にある点の中央値
- `t_post`: `etime + 0.05` 日付近（±0.025 日）にある点の中央値

対応するフラックス中央値 `val_pre`, `val_post` から

- `a = (val_post - val_pre)/(t_post - t_pre)`
- `b = val_pre - a*t_pre`

を計算し

- `flux_diff = flux - (a*bjd + b)`

をフレア抽出用の差分系列とする。

### 7.2 新しい開始・終了時刻

ピーク時刻 `ptime` に一致するインデックス `n_peak` から、

- 前後に `flux_diff >= err` を満たす範囲へ拡張して `n_start`, `n_end` を決定

する。

### 7.3 採用条件（実装上の品質フィルタ）

以下を満たすイベントのみ採用する。

- `ss_flare`（新しいフレア区間）の点数が 2 点以上
- `flux_diff[ss_flare] - 3*err[ss_flare] >= 0` を満たす点が 2 点以上

（すなわち 3σ を 2 点以上で満たすことが実装上の採用基準）

### 7.4 減衰時定数（e-folding time）

区間内のピーク値 `peak_flux` から

- `flux_diff >= peak_flux*exp(-1)`

を満たす最後の点までの時間差を `edecay` として記録する。

### 7.5 精密フラックス積算（count_new）

採用されたイベントについて

- `count_new = sum(flux_diff[n_start : n_end+1])`

を計算し、エネルギー推定に用いる。

## 8. フレアエネルギー推定（TESS 帯）

### 8.1 TESS 応答関数

TESS の応答関数は `data/tess-response-function-v1.0.csv`（`TESS_RESPONSE_PATH`）から読み込む。

- `wave`（波長）
- `resp`（応答）
- `dw = [diff(wave), 0]`

### 8.2 プランク関数

`BaseFlareDetector.planck(wav, T)` を用いて黒体放射の強度を計算する。

### 8.3 基本式（BaseFlareDetector）

`BaseFlareDetector.tess_band_energy(count)` は以下でフレアエネルギーを推定する。

- 時間刻み: `dt = 120.0` 秒
- 恒星半径: `Rstar = 695510e5 * R_sunstar_ratio`（cm）
- ステファン・ボルツマン定数: `sigma = 5.67e-5`

TESS 応答で重み付けした黒体強度

- `I_star = sum(dw * planck(wave*1e-9, T_star) * resp)`
- `I_ref = sum(dw * planck(wave*1e-9, 10000) * resp)`

を用い

- `star_intensity_ratio = I_star / I_ref`
- `area_factor = (pi * Rstar^2) * star_intensity_ratio`

として

- `E = sigma * (10000^4) * area_factor * dt * count`

を返す（`count` は `flaredetect` または `flaredetect_check` での積算値）。

### 8.4 DS Tuc A（伴星込みの上書き）

`FlareDetector_DS_Tuc_A.tess_band_energy(count)` は伴星の寄与を含めるため、面積加重した強度を使用する。

- 主星半径: `R_primary = Rsun_cm * R_sunstar_ratio`
- 伴星半径: `R_companion = Rsun_cm * 0.864`
- 伴星温度: `T_companion = 4700`

各強度

- `I_main = sum(dw * planck(wave*1e-9, T_star) * resp)`
- `I_comp = sum(dw * planck(wave*1e-9, 4700) * resp)`
- `I_ref = sum(dw * planck(wave*1e-9, 10000) * resp)`

面積加重

- `star_intensity = I_main*R_primary^2 + I_comp*R_companion^2`
- `area_factor = pi * (star_intensity/I_ref)`

として

- `E = sigma * (10000^4) * area_factor * dt * count`

を返す。

## 9. 有効観測時間の推定

`BaseFlareDetector.calculate_precise_obs_time()` は

- `diff_bjd >= 0.2` 日

を観測ギャップとして扱い、

- `precise_obs_time = (bjd[-1] - bjd[0]) - sum(gap_durations)`

で有効観測時間を求める。

## 10. エネルギー範囲でのフレア数と総エネルギー

`BaseFlareDetector.flare_energy(energy_threshold_low, energy_threshold_high)` は、イベントエネルギー配列 `energy` を昇順にソートし

- `energy_threshold_low <= E <= energy_threshold_high`

を満たすイベント数 `flare_number` と、その総和 `sum_flare_energy` を計算する。

パイプライン上は `process_data()` の引数 `ene_thres_low` / `ene_thres_high` が未指定なら、インスタンスの `self.ene_thres_low` / `self.ene_thres_high`（既定 `5e33` 〜 `2e40` erg）を用いる。

## 11. 回転変動振幅とスポット指標

### 11.1 変動振幅

`BaseFlareDetector.flux_diff(min_percent=0.02, max_percent=0.98)` は `mPDCSAPflux` をソートし

- `amplitude = P98 - P2`

（2–98 パーセンタイル幅）で回転変動振幅 `brightness_variation_amplitude` を定義する。

### 11.2 温度低下量（スポット温度差）

`d_T_star` は実装上

- `d_T_star = 3.58e-5*T_star^2 + 0.249*T_star - 808`

で計算される。

### 11.3 スポット指標

実装は以下の式で `starspot` と `starspot_ratio` を定義する。

- `starspot = 2*pi*(R_sunstar_ratio*695510e3)^2 * (T^4/(T^4 - (T-dT)^4)) * amplitude`
- `starspot_ratio = (T^4/(T^4 - (T-dT)^4)) * amplitude`

（単位系は実装に従う。`R` の係数はエネルギー推定部とは異なる表記になっている。）

### 11.4 DS Tuc A（主星+伴星の面積で上書き）

`FlareDetector_DS_Tuc_A.flux_diff()` は `super().flux_diff()` を実行後、主星+伴星の面積を加えたスケールで `starspot` を再計算する。

## 12. 回転周期推定（Lomb–Scargle）

`BaseFlareDetector.rotation_period()` は `astropy.timeseries.LombScargle` を用い、

- 入力時刻: `t = tessBJD - tessBJD[0]`
- 入力系列: `y = mPDCSAPflux`

に対して周波数グリッドを生成してパワースペクトルを計算する。

- 周波数範囲: `1/period_max` 〜 `1/period_min`
- 分割数: `rotation_n_points`（既定 `10000`）
- LS の method: `rotation_ls_method`（既定 `"auto"`）
- `assume_regular_frequency=True`

最大パワーの周波数を回転周期 `per` とし、半値幅（`power > max(power)/2` の端点差の半分）を `per_err` とする。

## 13. 星固有パラメータ（派生クラス）

以下は現行 `src/flarepy_*.py` にハードコードされている設定値である。

| Star / Detector                     | `R_sunstar_ratio` | `T_star` [K] | `flux_mean` | `err_constant_mean` | `rot_period` [day] | Rotation search [`min`,`max`] [day] | `f_cut_lowpass` | `f_cut_spline` | `sector_threshold` | `gap_threshold` | 特記事項                                                                                        |
| ----------------------------------- | ----------------: | -----------: | ----------: | ------------------: | -----------------: | ----------------------------------- | --------------: | -------------: | -----------------: | --------------: | ----------------------------------------------------------------------------------------------- |
| DS Tuc A (`FlareDetector_DS_Tuc_A`) |              0.87 |         5428 | 119633.9953 |        0.0005505874 |          0.3672258 | [1.0, 8.0]                          |               3 |              6 |                 74 |            0.05 | `remove()`（トランジット区間除去）、`tess_band_energy()`（伴星込み）、`flux_diff()`（面積補正） |
| EK Dra (`FlareDetector_EK_Dra`)     |              0.94 |         5700 | 249320.3537 |        0.0004111605 |          0.2094793 | [1.5, 5.0]                          |               3 |              6 |                 74 |             0.2 | ギャップ閾値のみ上書き                                                                          |
| V889 Her (`FlareDetector_V889_Her`) |              1.00 |         6550 | 300710.6233 |        0.0003969586 |          0.4398277 | [0.3, 2.0]                          |              30 |             40 |                 90 |           0.004 | `detrend_flux()` を独自実装（フレア区間マスク補間 + lowpass/spline）                            |

## 14. 実装・環境（再現性情報）

### 14.1 言語と主要依存

`pyproject.toml` の依存関係（抜粋）:

- Python: `>=3.13`
- `astropy`（FITS 読み込み、Lomb–Scargle）
- `numpy`（配列処理、FFT、統計）
- `scipy`（`interp1d`）
- `matplotlib` / `plotly`（可視化）

### 14.2 実行エントリ

利用例（README の意図に沿う）:

- `BaseFlareDetector(file=..., process_data=True)` または
- `FlareDetector_* (file=..., process_data=True)`

により、`process_data()` が呼ばれ解析が完結する。

---

## 付録 A. 本 Method 記述の対応関係（関数 ↔ 節）

- **Data / Normalization**: `load_TESS_data()`
- **Gap correction**: `apply_gap_correction()`
- **Detrending**: `detrend_flux()`（V889 Her は `FlareDetector_V889_Her.detrend_flux()`）
- **Error model**: `reestimate_errors()`
- **Flare detection**: `flaredetect()`, `flaredetect_check()`
- **Energy estimation**: `tess_band_energy()`（DS Tuc A は上書き）
- **Observation time**: `calculate_precise_obs_time()`
- **Rotation period**: `rotation_period()`
- **Spot proxy**: `flux_diff()`（DS Tuc A は上書き）
