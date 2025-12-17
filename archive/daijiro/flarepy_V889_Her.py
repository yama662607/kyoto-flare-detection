#!/usr/bin/env python3
"""
flare_improved.py

FlareDetector クラス
=================================

本クラスは、TESS の光度データをロードし、ギャップ補正・デトレンドを行い、
フレア検出とエネルギー算出をまとめて行うためのクラスです。以前のノートブックで
行っていた処理を、メソッドとして一貫した手続きを実装しています。
"""

import os
import re

import astropy.io.fits as fits
import numpy as np
import plotly.graph_objects as go
from astropy.timeseries import LombScargle
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d


class FlareDetector:
    """
    TESS光度データからのフレア検出とエネルギー推定を行うクラス。

    Attributes (クラス変数):
    -------------------------
        array_flare_ratio (np.ndarray):
            全インスタンスのフレア検出割合を格納するNumPy配列。
            (フレア検出割合 = フレアの検出数 / 観測時間)
        average_flare_ratio (float):
            全インスタンスのフレア検出割合の合計値を保持するクラス変数。
        array_observation_time (np.ndarray):
            全インスタンスの観測時間中央値を格納するNumPy配列。
        array_energy_ratio (np.ndarray):
            全インスタンスの時間あたりのフレアエネルギー合計値を格納するNumPy配列。

    Attributes (インスタンス変数):
    ------------------------------
        file (str):
            TESSのFITSファイルのパス。
        data_name (str):
            グラフに表示するデータ名。
        R_sunstar_ratio (float):
            恒星の半径を太陽半径で割った比（0.94 なら恒星半径は太陽の 0.94 倍）。
        T_star (float):
            恒星の有効温度 (K)。
        tessheader1 (astropy.io.fits.header.Header):
            FITSファイルのヘッダ情報（拡張0）。
        tessBJD (np.ndarray):
            観測時刻 (BJD) の配列。
        mPDCSAPflux (np.ndarray):
            PDCSAP フラックスを正規化した配列。
        mPDCSAPfluxerr (np.ndarray):
            PDCSAP フラックスの誤差を正規化した配列。
        gmPDCSAPflux (np.ndarray):
            ギャップ補正＋バッファを追加したフラックス配列。
        gmPDCSAPfluxerr (np.ndarray):
            ギャップ補正＋バッファを追加したフラックス誤差配列。
        gtessBJD (np.ndarray):
            バッファを追加した拡張 BJD。
        buffer_size (int):
            データ前後に追加するバッファ領域の大きさ。
        f_cut_lowpass (float):
            ローパスフィルターのカットオフ周波数。
        f_cut_spline (float):
            スプラインフィルターのカットオフ周波数。
        s2mPDCSAPflux (np.ndarray):
            デトレンド後のフラックス配列。
        mPDCSAPfluxerr_cor (np.ndarray):
            ローカルスキャッターによる誤差推定後のフラックス誤差配列。
        detecttime (np.ndarray):
            フレアを検出した際の時刻配列。
        starttime (np.ndarray):
            フレア開始時刻配列。
        endtime (np.ndarray):
            フレア終了時刻配列。
        peaktime (np.ndarray):
            フレアピーク時刻配列。
        energy (np.ndarray):
            フレアのエネルギー推定値配列。
        a_i (np.ndarray):
            フレア時のベースライン直線近似パラメータ（傾き）。
        b_i (np.ndarray):
            フレア時のベースライン直線近似パラメータ（切片）。
        duration (np.ndarray):
            フレア継続時間配列。
        edecay (np.ndarray):
            フレアの指数崩壊時間スケール配列。
        flare_ratio (float):
            1インスタンスあたりのフレア検出割合。 (フレア検出数 / 観測時間)
        precise_obs_time (float):
            正確な観測時間 (ギャップ時間を除外)。
        flare_number (int):
            2*10^33 erg 以上のエネルギーを持つフレアの数。
        sum_flare_energy (float):
             2*10^33 erg 以上のエネルギーを持つフレアの合計エネルギー。
    """

    ### クラス変数
    #
    array_flare_ratio = np.array([])
    array_observation_time = np.array([])
    array_energy_ratio = np.array([])
    array_energy_ave = np.array([])
    # ライトカーブの振幅を記録
    array_amplitude = np.array([])
    array_starspot = np.array([])
    array_starspot_ratio = np.array([])
    # データの名前を記録（グラフ描画用）
    array_data_name = np.array([])
    array_per = np.array([])
    array_per_err = np.array([])

    def __init__(
        self,
        process_data=False,
        R_sunstar_ratio=1.0,
        T_star=6550,
        file=None,
        buffer_size=10000,
        f_cut_lowpass=30,
        f_cut_spline=40,
        ene_thres_low=5e33,
        ene_thres_high=2e40,
    ):
        """
        FlareDetector クラスのコンストラクタ。

        Parameters
        ----------
        process_data : bool, optional
            True のとき、インスタンス生成直後に自動でデータを処理する。デフォルトはFalse。
        R_sunstar_ratio : float, optional
            恒星の半径を太陽半径で割った値。デフォルトは0.94。
        T_star : float, optional
            恒星の有効温度 (K)。デフォルトは5700。
        file : str, optional
            処理対象の FITS ファイルパス。デフォルトはNone。
        buffer_size : int, optional
            光度時系列の前後に追加するバッファ領域の大きさ。デフォルトは10000。
        f_cut_lowpass : float, optional
            ローパスフィルター用のカットオフ周波数。デフォルトは3。
        f_cut_spline : float, optional
            スプラインフィルター用のカットオフ周波数。デフォルトは6。
        """
        ### インスタンス変数
        # デフォルト値として初期値を設定
        self.file = file
        self.R_sunstar_ratio = R_sunstar_ratio
        self.T_star = T_star
        self.d_T_star = 3.58e-5 * self.T_star**2 + 0.249 * self.T_star - 808
        self.buffer_size = buffer_size
        self.f_cut_lowpass = f_cut_lowpass
        self.f_cut_spline = f_cut_spline

        # データ配列をNoneで初期化
        self.data_name = None
        self.tessheader1 = None
        self.tessBJD = None
        self.mPDCSAPflux = None
        self.mPDCSAPfluxerr = None
        self.gmPDCSAPflux = None
        self.gmPDCSAPfluxerr = None
        self.gtessBJD = None
        self.s2mPDCSAPflux = None
        self.mPDCSAPfluxerr_cor = None
        self.filtered_flux = None
        self.s1_flux = None
        self.flux_spline = None
        self.before_low_spline = None
        self.flux_splined = None

        # フレア検出率をゼロで初期化
        self.flare_ratio = 0.0

        # フレア検出結果をNoneで初期化
        self.detecttime = None
        self.starttime = None
        self.endtime = None
        self.peaktime = None
        self.energy = None
        self.a_i = None
        self.b_i = None
        self.duration = None
        self.edecay = None
        self.precise_obs_time = 0.0
        self.flare_number = 0
        self.sum_flare_energy = 0.0
        self.per = None
        self.per_err = 0.0
        self.brightness_variation_amplitude = 0.0
        self.starspot = 0.0
        self.starspot_ratio = 0.0
        self.rot_period = 0.4398277479138892

        # load_TESS_data() はインタンス生成時に実行
        self.load_TESS_data()

        # process_data=True のときのみ、コンストラクタ直後に全ての処理を実行
        if process_data and (self.file is not None):
            self.process_data(
                ene_thres_low=ene_thres_low, ene_thres_high=ene_thres_high
            )

    def load_TESS_data(self):
        """
        TESS の FITS ファイルを読み込み、フラックスを正規化するメソッド。
        Lazy Loading：まだデータが存在しない場合にのみ実行する。
        """
        if self.tessBJD is not None:
            # すでに読み込み済み
            return

        if self.file is None:
            print("Error: ファイルパスが指定されていません。")
            return

        fname = self.file
        fname_base = os.path.basename(fname)  # フルパスからファイル名を取得

        # 正規表現で必要な部分を抽出
        match = re.match(r"(.+)-\d+-\d+-s_lc\.fits$", fname_base)
        if match:
            self.data_name = match.group(1)  # グラフに表示するデータ名

        match = re.match(r"[a-z]+\d+-s00(.+)-\d+-\d+-s_lc\.fits$", fname_base)
        if match:
            data_number = int(match.group(1))  # グラフに表示するデータ名

        # FITS データを読み込み
        hdulist = fits.open(
            fname, memmap=True
        )  # memmap=True でメモリマップを使用し、メモリ節約
        hdr1 = hdulist[0].header
        data = hdulist[1].data

        # NaN を含む行を除外
        if data_number > 90:
            mask = ~np.isnan(data.field("SAP_FLUX"))
            bjd = data.field("time")[mask]
            # PDCを除いたものと比較
            pdcsap_flux = data.field("SAP_FLUX")[mask]
            pdcsap_flux_err = data.field("SAP_FLUX_ERR")[mask]
        else:
            mask = ~np.isnan(data.field("PDCSAP_FLUX"))
            bjd = data.field("time")[mask]
            # PDCを除いたものと比較
            pdcsap_flux = data.field("PDCSAP_FLUX")[mask]
            pdcsap_flux_err = data.field("PDCSAP_FLUX_ERR")[mask]

        # 光度データの正規化
        # self.flux_mean = np.mean(pdcsap_flux[pdcsap_flux > 0.0])
        self.flux_mean = 300710.62334465684
        norm_flux = pdcsap_flux / self.flux_mean
        norm_flux_err = pdcsap_flux_err / self.flux_mean

        # インスタンス変数へ格納
        self.tessheader1 = hdr1
        self.tessBJD = bjd
        self.mPDCSAPflux = norm_flux
        self.mPDCSAPfluxerr = norm_flux_err

    def apply_gap_correction(self):
        """
        時系列データ内のギャップを補正し、データの前後にバッファ領域を追加するメソッド。
        """
        # 短い名前のローカル変数を使う
        bjd = self.tessBJD.copy()
        flux = self.mPDCSAPflux.copy()
        flux_err = self.mPDCSAPfluxerr.copy()
        buf_size = self.buffer_size

        # ギャップ検出 (差分が0.2日以上の箇所をギャップとみなす)
        diff_bjd = np.diff(bjd)
        gap_indices = np.where(diff_bjd >= 0.004)[0]

        # ====== ギャップ補正 ======
        # flux をインプレースで更新するため、一時コピーを作らない
        for idx in gap_indices:
            # idx+1 以降を「差分だけ減らす」
            flux[idx + 1 :] -= flux[idx + 1] - flux[idx]

        # ====== バッファ追加 ======
        # np.full で先頭・末尾にバッファを追加
        flux_ext = np.hstack(
            [
                np.full(buf_size, flux[0]),
                flux,
                np.full(buf_size, flux[-1]),
            ]
        )
        flux_err_ext = np.hstack(
            [
                np.full(buf_size, 0.0001),
                flux_err,
                np.full(buf_size, 0.0001),
            ]
        )

        # 時刻配列にもバッファを追加
        # 2分間隔を日数に換算 → 2/(24*60)
        dt_min = 2 / (24 * 60)
        a = np.arange(buf_size) * dt_min
        bjd_ext = np.hstack(
            [
                (a - a[-1] - dt_min + bjd[0]),
                bjd,
                (a + dt_min + bjd[-1]),
            ]
        )

        self.gmPDCSAPflux = flux_ext
        self.gmPDCSAPfluxerr = flux_err_ext
        self.gtessBJD = bjd_ext

    def lowpass(self, x, y, fc=5.428):  # EK Draとの比率を計算するとfc=5.428
        """
        ローパスフィルタを用いて低周波成分を抽出するメソッド。

        Parameters
        ----------
        x : np.ndarray
            時刻配列
        y : np.ndarray
            フラックス配列
        fc : float
            カットオフ周波数

        Returns
        -------
        np.ndarray
            ローパスフィルタ後のフラックス配列
        """
        n = len(x)
        dt = 2 / (24 * 60)  # サンプリング間隔 (2 分)
        freq = np.linspace(0, 1.0 / dt, n)
        F = np.fft.fft(y) / (n / 2)
        F[0] /= 2
        F2 = F.copy()
        F2[freq > fc] = 0
        filtered = np.fft.ifft(F2)
        return np.real(filtered * n)

    def difference_at_lag(self, arr, n=1):
        """
        配列 arr の要素とその n 個先の要素との差を計算します。
        つまり、arr[i+n] - arr[i] を計算します。

        Parameters:
        arr (np.ndarray): 入力配列。
        n (int):         差を取る要素間の距離（ラグ）。デフォルトは1（np.diffと同じ）。

        Returns:
        np.ndarray:      差分を格納した配列。長さは len(arr) - n になります。
        """
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)  # NumPy配列でなければ変換

        if n <= 0:
            raise ValueError("n (lag) must be a positive integer.")
        if n >= len(arr):
            # return np.array([]) # 空の配列を返すかエラーにするか
            raise ValueError(
                f"n (lag) {n} is too large for array of length {len(arr)}."
            )

        # arr[n:] は arr の n番目の要素から最後まで
        # arr[:-n] は arr の最初から (最後からn番目の要素の直前)まで
        # これにより、arr[i+n] と arr[i] に対応する要素が同じインデックスで比較される
        return arr[n:] - arr[:-n]

    def detrend_flux(self):
        """
        フラックスをデトレンドするメソッド。
        1) ローパスフィルタによって長期変動を除去
        2) フレア候補点を除外したスプライン補完でベースラインを再評価
        3) 最終的なデトレンドフラックスを self.s2mPDCSAPflux に格納
        """
        # ローカル変数を短く
        time_ext = self.gtessBJD
        flux_ext = self.gmPDCSAPflux
        flux_err_ext = self.gmPDCSAPfluxerr
        buf_size = self.buffer_size

        # ローパスをする前に、大きいフレアが影響しないように取り除いておく、とりあえず急に1%以上上がるところを除く
        diff_flux = np.array([])
        flare_can_start = 0.0
        before_low_flare_list = []
        flare_can_end = np.array([])

        # flare_canにstart～endにあるindex全てを入れる
        diff_time = np.diff(time_ext)
        diff_flux = np.diff(flux_ext)
        flux_diff_lag2 = self.difference_at_lag(flux_ext, n=2)
        flux_diff_lag3 = self.difference_at_lag(flux_ext, n=3)
        flux_diff_lag4 = self.difference_at_lag(flux_ext, n=4)
        flux_diff_lag5 = self.difference_at_lag(flux_ext, n=5)

        flux_diff_lag2_appended = np.append(self.difference_at_lag(flux_ext, n=2), 0)
        flux_diff_lag3_appended = np.append(
            self.difference_at_lag(flux_ext, n=3), [0, 0]
        )
        flux_diff_lag4_appended = np.append(
            self.difference_at_lag(flux_ext, n=4), [0, 0, 0]
        )
        flux_diff_lag5_appended = np.append(
            self.difference_at_lag(flux_ext, n=5), [0, 0, 0, 0]
        )

        time_ext_for_diff = time_ext[:-1]  # diff_flux[k] に対応する時間 time_ext[k]

        flare_can_start_candidates = np.where(
            (
                (diff_flux > 0.01)  # 元の条件
                | (flux_diff_lag2_appended > 0.01)
                | (flux_diff_lag3_appended > 0.01)
                | (flux_diff_lag4_appended > 0.01)
                | (flux_diff_lag5_appended > 0.01)
            )
            & (diff_time < 0.005)
            # | (time_ext_for_diff < time_ext[10])  # 修正後の time_ext 条件 (例)
            # | (time_ext_for_diff > time_ext[-11]) # 修正後の time_ext 条件 (例)
        )[0]

        before_low_flare_list = []
        valid_flare_starts = []  # 実際に終了点が見つかった開始点のみを格納

        for j_start_candidate in flare_can_start_candidates:
            found_end = False
            for i_end_candidate in range(j_start_candidate + 5, len(flux_ext)):
                if abs(flux_ext[j_start_candidate] - flux_ext[i_end_candidate]) < 0.008:
                    before_low_flare_list.append(i_end_candidate)
                    valid_flare_starts.append(j_start_candidate)  # 対応する開始点を保存
                    break

        flare_starts = np.array(valid_flare_starts, dtype=int)
        flare_ends = np.array(before_low_flare_list, dtype=int)

        # print 文の前に flare_starts や flare_ends が空でないかチェックするのも良い
        # if flare_starts.size > 0: # もし空でなければ表示
        #     print(f"Identified flare start indices: {flare_starts, time_ext[flare_starts]}")
        # else:
        #     print("No flare starts identified.")

        # if flare_ends.size > 0: # もし空でなければ表示
        #     print(f"Identified flare end indices: {flare_ends, time_ext[flare_ends]}")
        # else:
        #     print("No flare ends identified.")

        # print(f"Identified flare start indices: {flare_starts,time_ext[flare_starts]}")
        # print(f"Identified flare end indices: {flare_ends,time_ext[flare_ends]}")

        # --- ここからフレア区間を除いてスプライン補間 ---

        # 1. 補間対象のfluxデータコピーを作成
        flux_to_interpolate = np.copy(flux_ext)

        # 2. マスクを作成 (True の部分がフレア区間 = 補間する部分)
        mask = np.zeros(len(flux_ext), dtype=bool)
        for start_idx, end_idx in zip(flare_starts, flare_ends):
            if start_idx < end_idx:  # 開始が終了より前であることを確認
                mask[start_idx : end_idx + 1] = True  # startからendまでをマスク
                # 注意: 上昇開始直前(start_idx)から下降完了(end_idx)までを補間対象とする
                # もし上昇が完了した点(start_idx + 1)から補間したい場合は調整

        # 3. マスクされていないデータポイント（フレア区間外）を取得
        time_valid = time_ext[~mask]
        flux_valid = flux_ext[~mask]

        # 4. スプライン補間

        # interp1d を使用 (線形、3次スプラインなど種類を選べる)
        # kind='linear', 'quadratic', 'cubic' など
        spline_func = interp1d(
            time_valid,
            flux_valid,
            kind="cubic",
        )

        # マスクされた部分（フレア区間）の時間を取得
        time_flare_intervals = time_ext[mask]

        # フレア区間の値を補間された値で置き換える
        if len(time_flare_intervals) > 0:
            flux_to_interpolate[mask] = spline_func(time_flare_intervals)

        # 補間後のflux (これがローパスされた結果に近いものになる)
        self.flux_splined = flux_to_interpolate
        # print("Spline interpolation applied.")

        # self.before_low_spline = interp1d(
        #     time_ext[flare_can],
        #     self.lowpass(
        #         time_ext[flare_can], flux_ext[flare_can], fc=self.f_cut_before_spline
        #     ),
        #     kind="cubic",
        # )

        # 1) ローパスフィルタ適用
        self.filtered_flux = self.lowpass(
            time_ext, self.flux_splined, fc=self.f_cut_lowpass
        )
        self.s1_flux = flux_ext - self.filtered_flux

        # 2) フレア候補点 (フラックスが誤差 * fac 未満) を除外してスプライン補完
        fac = 3
        ss_flarecan = np.where(
            (self.s1_flux <= flux_err_ext * fac)
            | (time_ext < time_ext[10])
            | (time_ext > time_ext[-11])
        )[0]

        # instance variable for test
        self.ss_flarecan = ss_flarecan
        self.time_ext = time_ext

        # スプライン用のベースライン作成
        baseline_spline = interp1d(
            time_ext[ss_flarecan],
            self.lowpass(
                time_ext[ss_flarecan], flux_ext[ss_flarecan], fc=self.f_cut_spline
            ),
            kind="cubic",
        )

        # 3) バッファを除いた範囲だけに適用して最終的なフラックスを取得
        valid_slice = slice(buf_size, buf_size + len(self.tessBJD))
        self.flux_spline = baseline_spline(time_ext[valid_slice])
        self.s2mPDCSAPflux = flux_ext[valid_slice] - self.flux_spline

    def reestimate_errors(self):
        """
        フラックスの誤差をローカルスキャッターから再推定するメソッド。
        """
        bjd = self.tessBJD
        flux = self.s2mPDCSAPflux
        err = np.ones(len(self.mPDCSAPfluxerr))

        # 各点において、周囲0.5日のフラックスが 0.005 以下の点を使ってスキャッターを評価
        for i in range(len(err)):
            nearby = (np.abs(bjd - bjd[i]) <= 0.5) & (flux <= 0.005)
            err[i] = np.std(flux[nearby])

        # 全体の平均スケールを元のエラーに合わせる
        print(np.mean(err))
        err_constant = np.mean(
            [
                0.000327511843591592,
                0.0002949516133656024,
                0.0005806540022277836,
                0.0003847150867966205,
            ]
        )
        err *= np.mean(self.mPDCSAPfluxerr) / err_constant
        self.mPDCSAPfluxerr_cor = err

    def flaredetect(self):
        """
        フレアの初期検出を行うメソッド。
        """
        flux_detrend = self.s2mPDCSAPflux
        err = self.mPDCSAPfluxerr_cor
        bjd = self.tessBJD

        # 5シグマを超える点を探す
        oversigma_idx = np.where(flux_detrend >= err * 5)[0]
        ss_detect = []
        detecttime = []

        # 連続した点をひとかたまりのフレア候補とする
        i = 0
        while i <= (len(oversigma_idx) - 3):
            marker = 0
            if i >= (len(oversigma_idx) - 1):
                break
            if (oversigma_idx[i + 1] - oversigma_idx[i]) != 1:
                marker += 1
            if marker >= 1:
                i += 1
                continue
            ndetect = oversigma_idx[i]
            ss_detect.append(ndetect)
            detecttime.append(bjd[ndetect])
            k = 0
            while ((i + k + 1) < len(oversigma_idx)) and (
                (oversigma_idx[i + k + 1] - oversigma_idx[i + k]) == 1
            ):
                k += 1
            i += k

        # 1シグマ以上の点を含めてフレア範囲を拡大して取得
        starttime, endtime, peaktime, count = [], [], [], []
        diff_bjd = np.diff(bjd)
        overonesigma_idx = np.where(flux_detrend >= err)[0]

        for idx in range(len(ss_detect)):
            n = ss_detect[idx]
            ss_ind = np.where(overonesigma_idx == n)[0]
            if len(endtime) > 0:
                if np.max(endtime) >= bjd[n]:
                    continue
            if len(ss_ind) == 0:
                continue
            ss_val = ss_ind[0]
            k = 0
            j = 0
            # 右方向
            while (ss_val + k + 1 < len(overonesigma_idx)) and (
                (overonesigma_idx[ss_val + k + 1] - overonesigma_idx[ss_val + k]) == 1
            ):
                k += 1
            # 左方向
            while (ss_val + j - 1 >= 0) and (
                (overonesigma_idx[ss_val + j] - overonesigma_idx[ss_val + j - 1]) == 1
            ):
                j -= 1

            # 十分なデータ数を確保
            if (n + j) <= 30 or (n + k) >= (len(bjd) - 30):
                continue

            # ギャップが大きすぎないかチェック
            a = diff_bjd[(n + j - 10) : (n + k + 10)]
            if len(a) > 0 and np.max(a) >= (2 / (24 * 60)) * 20:
                continue

            # フレアの始まりと終わりを確定
            starttime.append(bjd[n + j])
            endtime.append(bjd[n + k])
            subbjd = bjd[(n + j) : (n + k + 1)]
            peak_idx = np.where(
                flux_detrend[(n + j) : (n + k + 1)]
                == max(flux_detrend[(n + j) : (n + k + 1)])
            )[0]
            if len(peak_idx) == 0:
                continue
            peaktime.append(subbjd[peak_idx[0]])
            count.append(np.sum(flux_detrend[(n + j) : (n + k + 1)]))

        self.detecttime = np.array(detecttime)
        self.starttime = np.array(starttime)
        self.endtime = np.array(endtime)
        self.peaktime = np.array(peaktime)
        count = np.array(count)

        # フレアエネルギーを計算
        self.energy = self.tess_band_energy(count)

    def flaredetect_check(self):
        """
        フレア検出後の追加チェックを行ってフレア情報を洗練するメソッド。
        """
        n_detect = [
            len(self.detecttime),
            len(self.starttime),
            len(self.endtime),
            len(self.peaktime),
            len(self.energy) if self.energy is not None else 0,
        ]
        N = min(n_detect)

        # 結果を格納するための配列
        detecttime_new = []
        starttime_new = []
        endtime_new = []
        peaktime_new = []
        count_new = []
        edecay_new = []
        a_array = []
        b_array = []

        # 主要変数をローカル名に
        flux = self.mPDCSAPflux
        err = self.mPDCSAPfluxerr
        bjd = self.tessBJD
        dtime = self.detecttime
        stime = self.starttime
        etime = self.endtime
        ptime = self.peaktime

        flag = 0
        for i in range(N):
            if flag == 1:
                flag = 0
                continue
            dt_pre = 0.025
            dt_post = 0.05
            w_pre = 0.0125
            w_post = 0.025

            ss_pre = np.where(np.abs(bjd - (stime[i] - dt_pre)) <= w_pre)[0]
            val_pre = np.median(flux[ss_pre]) if len(ss_pre) > 0 else np.nan

            ss_post = np.where(np.abs(bjd - (etime[i] + dt_post)) <= w_post)[0]
            val_post = np.median(flux[ss_post]) if len(ss_post) > 0 else np.nan

            if np.isnan(val_pre) or np.isnan(val_post):
                continue

            t_pre = np.median(bjd[ss_pre]) if len(ss_pre) > 0 else np.nan
            t_post = np.median(bjd[ss_post]) if len(ss_post) > 0 else np.nan
            if (t_post - t_pre) == 0:
                a_val = 0
            else:
                a_val = (val_post - val_pre) / (t_post - t_pre)
            b_val = val_pre - a_val * t_pre

            flux_diff = flux - (a_val * bjd + b_val)

            peak_idx = np.where(bjd == ptime[i])[0]
            if len(peak_idx) == 0:
                continue
            n_peak = peak_idx[0]

            # 上方向
            k = 0
            while (n_peak + k < len(flux_diff)) and (
                flux_diff[n_peak + k] >= err[n_peak + k]
            ):
                k += 1
            # 下方向
            j = 0
            while (n_peak + j >= 0) and (flux_diff[n_peak + j] >= err[n_peak + j]):
                j -= 1

            n_end = n_peak + k - 1
            n_start = n_peak + j + 1
            if n_start < 0 or n_end >= len(flux_diff):
                continue

            ss_flare = np.where((bjd >= bjd[n_start]) & (bjd <= bjd[n_end]))[0]
            if len(ss_flare) <= 1:
                continue

            # 有効なフレアかどうかを確認 (3σ以上が2点以上あるか)
            if len(np.where((flux_diff[ss_flare] - 3 * err[ss_flare]) >= 0)[0]) <= 1:
                continue

            # ピークフラックスの指数崩壊スケールを計算
            peak_flux = flux_diff[ss_flare].max()
            peak_flux_indices = np.where(flux_diff[ss_flare] == peak_flux)[0]
            if len(peak_flux_indices) == 0:
                continue
            peak_loc = ss_flare[peak_flux_indices[0]]

            ll = 0
            while ((peak_loc + ll) < len(flux_diff)) and (
                flux_diff[peak_loc + ll] >= peak_flux * np.exp(-1)
            ):
                ll += 1
            if ll == 0:
                continue

            decay_time = bjd[peak_loc + ll] - bjd[peak_loc]

            # 結果を格納
            edecay_new.append(decay_time)
            a_array.append(a_val)
            b_array.append(b_val)
            starttime_new.append(bjd[n_start])
            endtime_new.append(bjd[n_end])
            peaktime_new.append(ptime[i])
            detecttime_new.append(dtime[i])
            count_new.append(np.sum(flux_diff[n_start : n_end + 1]))

            # 次のフレアと重ならないように
            if i < (N - 1):
                if bjd[n_end] >= stime[i + 1]:
                    flag = 1

        count_new = np.array(count_new)
        if len(count_new) > 0:
            energy_new = self.tess_band_energy(count_new)
            duration_new = (
                np.array(endtime_new) - np.array(starttime_new) + (2 / (24 * 60))
            )
            edecay_new = np.array(edecay_new)
            a_array = np.array(a_array)
            b_array = np.array(b_array)
        else:
            energy_new = np.array([])
            duration_new = np.array([])
            edecay_new = np.array([])
            a_array = np.array([])
            b_array = np.array([])

        # 最終的な結果をインスタンス変数へ更新
        self.detecttime = np.array(detecttime_new)
        self.starttime = np.array(starttime_new)
        self.endtime = np.array(endtime_new)
        self.peaktime = np.array(peaktime_new)
        self.energy = energy_new
        self.a_i = a_array
        self.b_i = b_array
        self.duration = duration_new
        self.edecay = edecay_new

    def planck(self, wav, T):
        """
        プランクの放射法則に基づく放射強度を計算する関数。

        Parameters
        ----------
        wav : float or np.ndarray
            波長 (メートル単位)
        T : float
            温度 (K)

        Returns
        -------
        float or np.ndarray
            プランク放射強度
        """
        h = 6.626e-34  # プランク定数 (J s)
        c = 3.0e8  # 光速 (m/s)
        k = 1.38e-23  # ボルツマン定数 (J/K)
        a = 2.0 * h * c**2
        b = h * c / (wav * k * T)
        intensity = a / ((wav**5) * (np.exp(b) - 1.0))
        return intensity

    def tess_band_energy(self, count):
        """
        TESS の応答関数を用いてフレアのエネルギーを推定するメソッド。

        Parameters
        ----------
        count : np.ndarray
            フレア領域のフラックス積分値

        Returns
        -------
        np.ndarray
            フレアエネルギー推定値の配列
        """
        try:
            # TESSの透過率 (応答関数) を読み込み
            wave, resp = np.loadtxt(
                "../data/tess-response-function-v1.0.csv", delimiter=","
            ).T
        except FileNotFoundError:
            print("Error: TESS応答関数のCSVファイルが見つかりません。")
            return np.array([])

        dt = 2 * 60.0  # 2 分を秒に変換
        dw = np.hstack([np.diff(wave), 0])
        Rsun_cm = 695510e5  # 太陽半径 (cm)
        sigma = 5.67e-5  # シュテファン・ボルツマン定数
        Rstar = Rsun_cm * self.R_sunstar_ratio

        # 恒星の放射強度 (T_star) と基準温度 (10000K) の比から、フレアエネルギーを推定
        star_intensity = np.sum(dw * self.planck(wave * 1e-9, self.T_star) * resp)
        ref_intensity = np.sum(dw * self.planck(wave * 1e-9, 10000) * resp)

        if ref_intensity == 0:
            print("Error: 参照強度がゼロです。")
            return np.array([])

        # count はフラックスの積分値。これに恒星の表面積と相対的な透過率を掛ける
        area_factor = (np.pi * Rstar**2) * (star_intensity / ref_intensity)
        energy = sigma * (10000**4) * area_factor * dt * count
        return energy

    def calculate_precise_obs_time(self):
        """
        観測データ中のギャップを考慮した正確な観測時間を計算するメソッド。
        """
        bjd = self.tessBJD

        # ギャップ検出 (差分が0.2日以上の箇所をギャップとみなす)
        diff_bjd = np.diff(bjd)
        gap_indices = np.where(diff_bjd >= 0.2)[0]

        gap_time = 0
        for i in range(len(gap_indices)):
            start_time = bjd[gap_indices[i]]
            end_time = bjd[gap_indices[i] + 1]
            gap_time += end_time - start_time
        self.precise_obs_time = bjd[-1] - bjd[0] - gap_time

    # 2*10^33以上のエネルギーを持つフレアの数
    def flare_energy(self, energy_threshold_low, energy_threshold_high):
        """
        検出されたフレアのエネルギーに関する情報を計算するメソッド。
        2*10^33 erg 以上のエネルギーを持つフレアの数と合計エネルギーを計算し、
        インスタンス変数に格納します。
        """
        if self.energy is None or len(self.energy) == 0:
            self.flare_number = 0
            self.sum_flare_energy = 0.0
            return

        energy_cor = np.sort(self.energy)  # フレアエネルギーを昇順にソート
        cumenergy = np.array([len(energy_cor) - i for i in range(len(energy_cor))])
        energy2e33_index = np.where(
            (energy_cor >= energy_threshold_low) & (energy_cor <= energy_threshold_high)
        )[0]  # 2*10^33以上のエネルギーを持つフレアのインデックス

        # print(f"ene_min={energy_cor[energy2e33_index[0]]} , ene_max={energy_cor[energy2e33_index[-1]]}")

        if len(energy2e33_index) > 0:
            self.flare_number = (
                cumenergy[energy2e33_index[0]] - cumenergy[energy2e33_index[-1]] + 1
            )
            # print(f"energy={energy_cor[energy2e33_index[0]:energy2e33_index[-1]+1]}{self.flare_number}、{self.precise_obs_time}")
            self.sum_flare_energy = np.sum(
                energy_cor[energy2e33_index[0] : energy2e33_index[-1] + 1]
            )
        else:
            self.flare_number = 0
            self.sum_flare_energy = 0.0

    def flux_diff(
        self, min_flux=0.02, max_flux=0.98
    ):  ##! このメゾットのみを呼び出しても、現状ではエネルギー情報は更新されない
        """
        フラックスの差分を計算して返すメソッド。

        Parameters
        ----------
        min_flux : float, optional
            最小フラックス値。デフォルトは0.02。
        max_flux : float, optional
            最大フラックス値。デフォルトは0.98。
        """
        # フラックスの差分を計算
        sorted_flux = sorted(self.mPDCSAPflux)  # mPDCSAPfluxの値をソート

        # brightness_variation_amplitudeを求めるための上下2%を抜く
        lower_bound = int(len(sorted_flux) * 0.02)
        upper_bound = int(len(sorted_flux) * 0.98)
        self.brightness_variation_amplitude = (
            sorted_flux[upper_bound] - sorted_flux[lower_bound]
        )
        self.starspot = (
            2
            * np.pi
            * (self.R_sunstar_ratio * 695510e3) ** 2
            * (self.T_star**4 / (self.T_star**4 - (self.T_star - self.d_T_star) ** 4))
            * self.brightness_variation_amplitude
        )
        self.starspot_ratio = (
            self.T_star**4 / (self.T_star**4 - (self.T_star - self.d_T_star) ** 4)
        ) * self.brightness_variation_amplitude

    def plot_flare(self):
        """
        データ全体の光度曲線と検出フレアを可視化するメソッド。

        Returns
        -------
        None
        """
        if self.tessBJD is None or self.mPDCSAPflux is None:
            print("Error: データが読み込まれていません。")
            return

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

        # 生の光度曲線
        fig.add_trace(
            go.Scatter(
                x=self.tessBJD,
                y=self.mPDCSAPflux,
                mode="lines",
                line=dict(color="black", width=1),
                name="Normalized Flux",
            ),
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="Normalized Flux", row=1, col=1)

        # デトレンド後
        if self.s2mPDCSAPflux is not None:
            fig.add_trace(
                go.Scatter(
                    x=self.tessBJD,
                    y=self.s2mPDCSAPflux,
                    mode="lines",
                    line=dict(color="black", width=1),
                    name="Detrended Flux",
                ),
                row=2,
                col=1,
            )

            # フレアのピーク位置を線で示す
            if self.peaktime is not None:
                for peak in self.peaktime:
                    fig.add_trace(
                        go.Scatter(
                            x=[peak, peak],
                            y=[0.023, 0.0243],
                            mode="lines",
                            line=dict(color="red", width=1, dash="dash"),
                            showlegend=False,
                        ),
                        row=2,
                        col=1,
                    )
        fig.update_xaxes(title_text="Time (BJD - 2457000)", row=2, col=1)
        fig.update_yaxes(title_text="Detrended Flux", row=2, col=1)
        # 生の光度曲線
        # fig.add_trace(
        #     go.Scatter(
        #         x=self.gtessBJD,
        #         y=self.filtered_flux,
        #         mode="lines",
        #         line=dict(color="black", width=1),
        #         name="filtered",
        #     ),
        #     row=3,
        #     col=1,
        # )
        # fig.update_yaxes(title_text="filtered Flux", row=3, col=1)

        #  # 生の光度曲線
        # fig.add_trace(
        #     go.Scatter(
        #         x=self.gtessBJD,
        #         y=self.s1_flux,
        #         mode="lines",
        #         line=dict(color="black", width=1),
        #         name="after lowpass Flux",
        #     ),
        #     row=4,
        #     col=1,
        # )
        # # if self.ss_flarecan is not None:
        # #     for i in self.ss_flarecan:
        # #         fig.add_trace(
        # #             go.Scatter(
        # #                 x=[self.gtessBJD[i], self.gtessBJD[i]],
        # #                 y=[0.023, 0.0243],
        # #                 mode="lines",
        # #                 line=dict(color="red", width=1, dash="dash"),
        # #                 showlegend=False,
        # #             ),
        # #             row=4,
        # #             col=1,
        # #         )
        # fig.update_yaxes(title_text="after lowpass Flux", row=4, col=1)

        # fig.add_trace(
        #     go.Scatter(
        #         x=self.tessBJD,
        #         y=self.flux_spline,
        #         mode="lines",
        #         line=dict(color="black", width=1),
        #         name="spline Flux",
        #     ),
        #     row=5,
        #     col=1,
        # )
        # fig.update_yaxes(title_text="spline Flux", row=5, col=1)

        # fig.add_trace(
        #     go.Scatter(
        #         x=self.gtessBJD,
        #         y=self.flux_splined,
        #         mode="lines",
        #         line=dict(color="black", width=1),
        #         name="before low spline Flux",
        #     ),
        #     row=6,
        #     col=1,
        # )

        # fig.add_trace(
        #     go.Scatter(
        #         x=self.gtessBJD,
        #         y=self.gmPDCSAPflux,
        #         mode="lines",
        #         line=dict(color="black", width=1),
        #         name="before low spline Flux",
        #     ),
        #     row=7,
        #     col=1,
        #  )

        # fig.update_yaxes(title_text="spline Flux", row=6, col=1)

        # Adding graph title
        fig.update_layout(
            title_text=f"Flare Detection Graph ({self.data_name})",
            title_font=dict(size=16),
            # width=900, # グラフの幅
            height=900,  # グラフの高さ
        )

        fig.show()

    def plot_energy(self):
        """
        エネルギー推定を含めたプロットを作成するメソッド。
        """
        if self.energy is None or len(self.energy) == 0:
            print("No flare energy data available")
            return

        # データの準備
        energy_cor = np.sort(self.energy)  # フレアエネルギーを昇順にソート
        cumenergy = np.array([len(energy_cor) - i for i in range(len(energy_cor))])

        # 累積分布のプロット
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=energy_cor,
                y=cumenergy / self.precise_obs_time,
                mode="lines",
                line=dict(color="gray", width=2),
                name="All Sector",
                line_shape="hv",  # steps-midに相当
            )
        )

        # Adding graph title
        fig.update_layout(
            title_text=f"Flare Energy Distribution ({self.data_name})",
            title_font=dict(size=16),
            # width=800, # グラフの幅
            # height=600, # グラフの高さ
        )

        # 軸の設定
        fig.update_xaxes(
            title_text="Flare Energy [erg]", type="log", title_font=dict(size=15)
        )
        fig.update_yaxes(
            title_text=r"Cumulative Number [day$^{-1}$]",
            type="log",
            title_font=dict(size=15),
        )

        # グラフ全体の設定
        fig.update_layout(
            legend=dict(x=0.05, y=0.95, font=dict(size=11)),
            # width=800,
            # height=600,
            showlegend=True,
        )

        # グリッド表示
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        # エネルギーが3e33以上のデータポイントだけを抽出
        mask = energy_cor >= 3e33
        energy_filtered = energy_cor[mask]
        cumenergy_filtered = cumenergy[mask]

        # エネルギーの対数を計算
        log_energy_filtered = np.log10(energy_filtered)
        log_cumenergy_filtered = np.log10(cumenergy_filtered / self.precise_obs_time)

        # 傾きをプロットに追加
        # if len(log_energy_filtered) > 1:
        #     for i in range(len(log_energy_filtered) - 1):
        #         slope = (log_cumenergy_filtered[i+1] - log_cumenergy_filtered[i]) / (log_energy_filtered[i+1] - log_energy_filtered[i])
        #         fig.add_trace(
        #             go.Scatter(
        #                 x=[energy_filtered[i], energy_filtered[i+1]],
        #                 y=[cumenergy_filtered[i] / self.precise_obs_time, cumenergy_filtered[i+1] / self.precise_obs_time],
        #                 mode="lines",
        #                 line=dict(color="red", width=2, dash="dot"),
        #                 name=f"Slope above 3e33: {slope:.2f}",
        #             )
        #         )

        fig.show()

    def rotation_period(self):
        # 2.4～3.0を1000等分して、その中のどれが自転周期なのかを探す
        frequency = 1 / np.linspace(0.3, 2.0, 10000)
        # 周波数毎にどれくらい強いか(どれくらい周期的であるか)を出す
        self.power = LombScargle(
            self.tessBJD - self.tessBJD[0], self.mPDCSAPflux
        ).power(frequency)
        # 一番強かった周波数について日数に換算する
        self.per = 1 / frequency[np.argmax(self.power)]

        # 誤差を求める
        # ノイズなどの影響できれいな周期性を持っていない。周波数のピークの幅は誤差と見れる
        half_max_power = np.max(self.power) / 2
        aa = np.where(self.power > half_max_power)[0]
        # maxの1/2以上の周波数だけを見ていて、その最初と最後の値(グラフでいうとピークの幅)を2で割ったものを誤差とみる
        self.per_err = (1 / frequency[aa[-1]] - 1 / frequency[aa[0]]) / 2

    def process_data(self, ene_thres_low, ene_thres_high):
        """
        TESS データの読み込みからフレア検出までの一連のプロセスを実行するメソッド。
        """
        # BJD が存在しない or 要素数が少ない場合は処理を中断
        if self.tessBJD is None or len(self.tessBJD) < 2:
            print("Error: BJD が正しく読み込まれていないか、要素数が不足しています。")
            return

        # 1) ギャップ補正 & バッファ追加
        self.apply_gap_correction()

        # 2) デトレンド
        self.detrend_flux()

        # 3) 誤差再推定
        self.reestimate_errors()

        # 4) フレア検出
        self.flaredetect()

        # 5) 検出結果を洗練
        self.flaredetect_check()

        # 6) 正確な観測時間の計算
        self.calculate_precise_obs_time()

        # 7) フレアエネルギーに関する計算
        self.flare_energy(
            energy_threshold_low=ene_thres_low, energy_threshold_high=ene_thres_high
        )

        # 8) ライトカーブの変動振幅を計算
        self.flux_diff()

        # 9)自転周期
        self.rotation_period()

        # 観測時間(最初から最後まで) がゼロでない場合に計算
        # フレア検出の割合を計算し、リストに追加
        if self.tessBJD is not None and len(self.tessBJD) > 1:
            obs_time = self.precise_obs_time
            if obs_time > 0:
                # n_flare = len(self.peaktime) if self.peaktime is not None else 0
                flare_ratio = self.flare_number / obs_time
                # リストに追加
                FlareDetector.array_flare_ratio = np.append(
                    FlareDetector.array_flare_ratio, flare_ratio
                )

                # 観測日数あたりのフレアエネルギーを計算
                sum_flare_energy_ratio = self.sum_flare_energy / obs_time
                FlareDetector.array_energy_ratio = np.append(
                    FlareDetector.array_energy_ratio, sum_flare_energy_ratio
                )

                # 観測時間の中央値をリストに追加 tessBJDの中央値を使用
                FlareDetector.array_observation_time = np.append(
                    FlareDetector.array_observation_time, np.median(self.tessBJD)
                )

                # ライトカーブの変動振幅をリストに追加
                FlareDetector.array_amplitude = np.append(
                    FlareDetector.array_amplitude, self.brightness_variation_amplitude
                )

                FlareDetector.array_starspot = np.append(
                    FlareDetector.array_starspot, self.starspot
                )

                FlareDetector.array_starspot_ratio = np.append(
                    FlareDetector.array_starspot_ratio, self.starspot_ratio
                )

                # データの名前をリストに追加
                FlareDetector.array_data_name = np.append(
                    FlareDetector.array_data_name, self.data_name
                )

                # 自転周期をリストに追加
                FlareDetector.array_per = np.append(FlareDetector.array_per, self.per)

                # 自転周期の誤差をリストに追加
                FlareDetector.array_per_err = np.append(
                    FlareDetector.array_per_err, self.per_err
                )

    def show_variables(self):
        """
        インスタンス変数とクラス変数の名前、役割、および要素数（配列の場合）を表示するメソッド。
        """
        print("-------- Instance Variables --------")

        # 各インスタンス変数の説明を辞書にまとめる (必要に応じて追加/修正)
        instance_var_info = {
            "file": "FITSファイルのパス",
            "data_name": "グラフに表示するデータ名",
            "R_sunstar_ratio": "恒星の半径を太陽半径で割った比",
            "T_star": "恒星の有効温度 [K]",
            "tessheader1": "FITSファイルのヘッダ情報",
            "tessBJD": "観測時刻 (BJD) の配列",
            "mPDCSAPflux": "正規化された PDCSAP フラックス配列",
            "mPDCSAPfluxerr": "正規化された PDCSAP フラックス誤差配列",
            "gmPDCSAPflux": "ギャップ補正＋バッファ付き フラックス配列",
            "gmPDCSAPfluxerr": "ギャップ補正＋バッファ付き フラックス誤差配列",
            "gtessBJD": "ギャップ補正＋バッファ付き BJD配列",
            "buffer_size": "データ前後のバッファ領域サイズ",
            "f_cut_lowpass": "ローパスフィルターのカットオフ周波数",
            "f_cut_spline": "スプラインフィルターのカットオフ周波数",
            "s2mPDCSAPflux": "最終的にデトレンドされたフラックス配列",
            "mPDCSAPfluxerr_cor": "ローカルスキャッターから再推定したフラックス誤差配列",
            "detecttime": "初期フレア検出時刻配列",
            "starttime": "フレア開始時刻配列",
            "endtime": "フレア終了時刻配列",
            "peaktime": "フレアピーク時刻配列",
            "energy": "フレアエネルギー推定値配列",
            "a_i": "フレア時のベースライン傾き",
            "b_i": "フレア時のベースライン切片",
            "duration": "フレア継続時間配列",
            "edecay": "フレアの指数崩壊時間配列",
            "flare_ratio": "フレア検出割合 (検出数 ÷ 観測時間)",
            "precise_obs_time": "正確な観測時間 (ギャップ時間を除く)",
            "flare_number": "2*10^33以上のフレア数",
            "sum_flare_energy": "2*10^33以上のフレアの合計エネルギー",
        }
        class_var_info = {
            "array_flare_ratio": "フレア検出割合のリスト",
            "array_observation_time": "観測時間のリスト",
            "array_energy_ratio": "時間あたりのフレアエネルギーのリスト",
        }

        for var_name, description in instance_var_info.items():
            value = getattr(self, var_name)
            if isinstance(value, np.ndarray):
                # 要素数と簡易情報を表示
                print(
                    f"{var_name}: {description} | type: np.ndarray | length: {value.size}"
                )
            else:
                # ndarray 以外はそのまま出力
                print(f"{var_name}: {description} | value: {value}")

        print("\n-------- Class Variables --------")
        for var_name, description in class_var_info.items():
            value = getattr(FlareDetector, var_name)
            if isinstance(value, np.ndarray):
                # 要素数と簡易情報を表示
                print(
                    f"{var_name}: {description} | type: np.ndarray | length: {value.size}"
                )
            else:
                # ndarray 以外はそのまま出力
                print(f"{var_name}: {description} | value: {value}")
        print("------------------------------------")
