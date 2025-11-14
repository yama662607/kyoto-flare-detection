#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d


class FlareDetector:
    """
    フレア検出のためのクラス

    Attributes (クラス変数):
    -------------------------
    array_flare_ratio : list
        全インスタンスのフレア検出割合を保持するクラス変数。
    average_flare_ratio : float
        全インスタンスのフレア検出割合の合計値を保持するクラス変数。
        (フレア検出割合 = フレアの検出数 / 観測時間)

    Attributes (インスタンス変数):
    ------------------------------
    file : str
        TESSのFITSファイルのパス。
    data_name : str
        グラフに表示するデータ名。
    R_sunstar_ratio : float
        恒星の半径を太陽半径で割った比（0.94 なら恒星半径は太陽の 0.94 倍）。
    T_star : float
        恒星の有効温度 (K)。
    tessheader1 : astropy.io.fits.header.Header
        FITSファイルのヘッダ情報（拡張0）。
    tessBJD : np.ndarray
        観測時刻 (BJD) の配列。
    mPDCSAPflux : np.ndarray
        PDCSAP フラックスを正規化した配列。
    mPDCSAPfluxerr : np.ndarray
        PDCSAP フラックスの誤差を正規化した配列。
    gmPDCSAPflux : np.ndarray
        ギャップ補正＋バッファを追加したフラックス配列。
    gmPDCSAPfluxerr : np.ndarray
        ギャップ補正＋バッファを追加したフラックス誤差配列。
    gtessBJD : np.ndarray
        バッファを追加した拡張 BJD。
    buffer_size : int
        データ前後に追加するバッファ領域の大きさ。
    f_cut_lowpass : float
        ローパスフィルターのカットオフ周波数。
    f_cut_spline : float
        スプラインフィルターのカットオフ周波数。
    s2mPDCSAPflux : np.ndarray
        デトレンド後のフラックス配列。
    mPDCSAPfluxerr_cor : np.ndarray
        ローカルスキャッターによる誤差推定後のフラックス誤差配列。
    detecttime : np.ndarray
        フレアを検出した際の時刻配列。
    starttime : np.ndarray
        フレア開始時刻配列。
    endtime : np.ndarray
        フレア終了時刻配列。
    peaktime : np.ndarray
        フレアピーク時刻配列。
    energy : np.ndarray
        フレアのエネルギー推定値配列。
    a_i, b_i : np.ndarray
        フレア時のベースライン直線近似パラメータ。
    duration : np.ndarray
        フレア継続時間配列。
    edecay : np.ndarray
        フレアの指数崩壊時間スケール配列。
    flare_ratio : float
        1インスタンスあたりのフレア検出割合。 (フレア検出数 / 観測時間)
    """

    ### クラス変数
    # 要件: 全てのインスタンスのフレア検出割合の合計を集計
    array_flare_ratio = np.array([])
    array_energy_ratio = np.array([])
    array_amplitude = np.array([])
    average_flare_ratio = 0.0

    # 観測した期間の中央値を保持するクラス変数のリスト
    array_observation_time = np.array([])

    def __init__(
        self,
        process_data=False,
        run_process_data_2=False,  # 👈 【追加1】新しい処理用のフラグ
        R_sunstar_ratio=0.987,
        T_star=5597,
        # V889_Her:6550,DS_Tuc_A:5597,AB_Dor:5081,EK_Dra:5700
        file=None,
        buffer_size=10000,
        f_cut_lowpass=3,
        # V889:
        f_cut_spline=6,
    ):
        """
        コンストラクタ

        Parameters
        ----------
        process_data : bool
            True のとき、インスタンス生成直後に自動でデータを処理する。
        R_sunstar_ratio : float
            恒星の半径を太陽半径で割った値。
        T_star : float
            恒星の有効温度 (K)。
        file : str
            処理対象の FITS ファイルパス。
        buffer_size : int
            光度時系列の前後に追加するバッファ領域の大きさ。
        f_cut_lowpass : float
            ローパスフィルター用のカットオフ周波数。
        f_cut_spline : float
            スプラインフィルター用のカットオフ周波数。
        """
        ### インスタンス変数
        # デフォルト値として初期値を設定
        self.file = file
        self.R_sunstar_ratio = R_sunstar_ratio
        self.T_star = T_star
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

        self.atessBJD = None
        self.amPDCSAPflux = None
        self.amPDCSAPfluxerr = None

        self.brightness_variation_amplitude = None
        self.precise_obs_time = None

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
        self.filtered_flux = None
        self.flux_spline = None
        self.s1_flux = None

        # load_TESS_data() はインタンス生成時に実行
        self.load_TESS_data()

        # process_data=True のときのみ、コンストラクタ直後に全ての処理を実行
        if process_data and (self.file is not None):
            self.process_data()

        # 👈 【追加3】 process_data_2=True のときのみ、新しい処理を実行
        if run_process_data_2 and (self.file is not None):
            self.process_data_2()  # ⚠️ 注意: このメソッドをFlareDetectorクラス内に定義する必要があります

    def load_TESS_data(self):
        """
        TESS の FITS ファイルを読み込み、フラックスを正規化するメソッド。
        Lazy Loading：まだデータが存在しない場合にのみ実行する。
        """
        if self.atessBJD is not None:
            # すでに読み込み済み
            return

        if self.file is None:
            print("Error: ファイルパスが指定されていません。")
            return

        fname = self.file
        fname_base = os.path.basename(fname)  # フルパスからファイル名を取得

        # 正規表現で必要な部分を抽出
        match = re.match(r"(.+)-s_lc\.fits$", fname_base)
        if match:
            self.data_name = match.group(1)  # グラフに表示するデータ名

        # FITS データを読み込み
        hdulist = fits.open(fname, memmap=True)  # memmap=True でメモリマップを使用し、メモリ節約
        hdr1 = hdulist[0].header
        data = hdulist[1].data

        # NaN を含む行を除外
        mask = ~np.isnan(data.field("PDCSAP_FLUX"))
        bjd = data.field("time")[mask]
        pdcsap_flux = data.field("PDCSAP_FLUX")[mask]
        pdcsap_flux_err = data.field("PDCSAP_FLUX_ERR")[mask]

        # 光度データの正規化
        flux_mean = np.mean(pdcsap_flux[pdcsap_flux > 0.0])
        norm_flux = pdcsap_flux / flux_mean
        norm_flux_err = pdcsap_flux_err / flux_mean

        # インスタンス変数へ格納
        self.tessheader1 = hdr1
        self.atessBJD = bjd
        self.amPDCSAPflux = norm_flux
        self.amPDCSAPfluxerr = norm_flux_err

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
        gap_indices = np.where(diff_bjd >= 0.1)[0]

        # ====== ギャップ補正 ======
        # flux をインプレースで更新するため、一時コピーを作らない
        for idx in gap_indices:
            # idx+1 以降を「差分だけ減らす」
            flux[idx + 1 :] -= flux[idx + 1] - flux[idx]

        # ====== バッファ追加 ======
        # np.full で先頭・末尾にバッファを追加
        flux_ext = np.hstack([
            np.full(buf_size, flux[0]),
            flux,
            np.full(buf_size, flux[-1]),
        ])
        flux_err_ext = np.hstack([
            np.full(buf_size, 0.0001),
            flux_err,
            np.full(buf_size, 0.0001),
        ])

        # 時刻配列にもバッファを追加
        # 2分間隔を日数に換算 → 2/(24*60)
        dt_min = 2 / (24 * 60)
        a = np.arange(buf_size) * dt_min
        bjd_ext = np.hstack([
            (a - a[-1] - dt_min + bjd[0]),
            bjd,
            (a + dt_min + bjd[-1]),
        ])

        self.gmPDCSAPflux = flux_ext
        self.gmPDCSAPfluxerr = flux_err_ext
        self.gtessBJD = bjd_ext

    def lowpass(self, x, y, fc=3):
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

        # 1) ローパスフィルタ適用
        self.filtered_flux = self.lowpass(time_ext, flux_ext, fc=self.f_cut_lowpass)
        self.s1_flux = flux_ext - self.filtered_flux

        # 2) フレア候補点 (フラックスが誤差 * fac 未満) を除外してスプライン補完
        fac = 3
        ss_flarecan = np.where((self.s1_flux <= flux_err_ext * fac) | (time_ext < time_ext[10]) | (time_ext > time_ext[-11]))[0]

        # スプライン用のベースライン作成
        baseline_spline = interp1d(
            time_ext[ss_flarecan],
            self.lowpass(time_ext[ss_flarecan], flux_ext[ss_flarecan], fc=self.f_cut_spline),
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
        err *= np.mean(self.mPDCSAPfluxerr) / np.mean(err)
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
            while ((i + k + 1) < len(oversigma_idx)) and ((oversigma_idx[i + k + 1] - oversigma_idx[i + k]) == 1):
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
            while (ss_val + k + 1 < len(overonesigma_idx)) and ((overonesigma_idx[ss_val + k + 1] - overonesigma_idx[ss_val + k]) == 1):
                k += 1
            # 左方向
            while (ss_val + j - 1 >= 0) and ((overonesigma_idx[ss_val + j] - overonesigma_idx[ss_val + j - 1]) == 1):
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
            peak_idx = np.where(flux_detrend[(n + j) : (n + k + 1)] == max(flux_detrend[(n + j) : (n + k + 1)]))[0]
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
            while (n_peak + k < len(flux_diff)) and (flux_diff[n_peak + k] >= err[n_peak + k]):
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
            while ((peak_loc + ll) < len(flux_diff)) and (flux_diff[peak_loc + ll] >= peak_flux * np.exp(-1)):
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
            duration_new = np.array(endtime_new) - np.array(starttime_new) + (2 / (24 * 60))
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
        # import os

        #     # 現在のディレクトリのパスを取得
        # current_directory = os.getcwd()
        # print(current_directory)

        #     # ディレクトリ内のすべてのファイルとフォルダを取得
        # files_and_folders = os.listdir(current_directory)

        #     # それぞれのファイルとフォルダの絶対パスを出力
        # for item in files_and_folders:
        #     absolute_path = os.path.join(current_directory, item)
        #     print(absolute_path)

        try:
            # TESSの透過率 (応答関数) を読み込み
            wave, resp = np.loadtxt("../data/tess-response-function-v1.0.csv", delimiter=",").T
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

        fig = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.1)

        # 生の光度曲線
        fig.add_trace(
            go.Scatter(x=self.tessBJD, y=self.mPDCSAPflux, mode="lines", line=dict(color="black", width=1), name="Normalized Flux"),
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="Normalized Flux", row=1, col=1)

        # デトレンド後
        if self.s2mPDCSAPflux is not None:
            fig.add_trace(
                go.Scatter(x=self.tessBJD, y=self.s2mPDCSAPflux, mode="lines", line=dict(color="black", width=1), name="Detrended Flux"),
                row=2,
                col=1,
            )

            # フレアのピーク位置を線で示す
            if self.peaktime is not None:
                for peak in self.peaktime:
                    fig.add_trace(
                        go.Scatter(
                            x=[peak, peak], y=[0.023, 0.0243], mode="lines", line=dict(color="red", width=1, dash="dash"), showlegend=False
                        ),
                        row=2,
                        col=1,
                    )

            fig.update_xaxes(title_text="Time (BJD - 2457000)", row=2, col=1)
            fig.update_yaxes(title_text="Detrended Flux", row=2, col=1)

        fig.add_trace(
            go.Scatter(
                x=self.gtessBJD,
                y=self.filtered_flux,
                mode="lines",
                line=dict(color="black", width=1),
                name="filtered",
            ),
            row=3,
            col=1,
        )
        fig.update_yaxes(title_text="filtered Flux", row=3, col=1)

        fig.add_trace(
            go.Scatter(
                x=self.gtessBJD,
                y=self.s1_flux,
                mode="lines",
                line=dict(color="black", width=1),
                name="after lowpass Flux",
            ),
            row=4,
            col=1,
        )
        fig.update_yaxes(title_text="after lowpass Flux", row=4, col=1)

        fig.add_trace(
            go.Scatter(
                x=self.tessBJD,
                y=self.flux_spline,
                mode="lines",
                line=dict(color="black", width=1),
                name="spline Flux",
            ),
            row=5,
            col=1,
        )
        fig.update_yaxes(title_text="spline Flux", row=5, col=1)

        fig.add_trace(
            go.Scatter(x=self.gtessBJD, y=self.gmPDCSAPflux, mode="lines", line=dict(color="black", width=1), name="None transit Flux"),
            row=6,
            col=1,
        )
        fig.update_yaxes(title_text="None transit Flux", row=6, col=1)

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
                y=cumenergy / (self.precise_obs_time),
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
        fig.update_xaxes(title_text="Flare Energy [erg]", type="log", title_font=dict(size=15))
        fig.update_yaxes(title_text=r"Cumulative Number [day$^{-1}$]", type="log", title_font=dict(size=15))

        # グラフ全体の設定
        fig.update_layout(legend=dict(x=0.05, y=0.95, font=dict(size=11)), width=800, height=600, showlegend=True)

        # グリッド表示
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)

        fig.show()

    def remove(self):
        """
        トランジットデータを除去するメソッド。
        """
        # トランジットデータの除去
        start_time_remove = [
            1332.243,
            1340.381,
            1348.514,
            2040.268,
            2056.539,
            2064.682,
            2080.954,
            3130.791,
            3147.072,
            3155.202,
            3163.342,
            3171.481,
        ]
        end_time_remove = [
            1332.377,
            1340.515,
            1348.646,
            2040.398,
            2056.672,
            2064.818,
            2081.048,
            3130.924,
            3147.202,
            3155.333,
            3163.473,
            3171.613,
        ]

        abjd = self.atessBJD.copy()
        aflux = self.amPDCSAPflux.copy()
        afluxerr = self.amPDCSAPfluxerr.copy()

        for i in range(len(start_time_remove)):
            mask = (abjd < start_time_remove[i]) | (abjd > end_time_remove[i])
            abjd = abjd[mask]
            aflux = aflux[mask]
            afluxerr = afluxerr[mask]

        self.tessBJD = abjd
        self.mPDCSAPflux = aflux
        self.mPDCSAPfluxerr = afluxerr

    def no_remove(self):
        abjd = self.atessBJD.copy()
        aflux = self.amPDCSAPflux.copy()
        afluxerr = self.amPDCSAPfluxerr.copy()

        self.tessBJD = abjd
        self.mPDCSAPflux = aflux
        self.mPDCSAPfluxerr = afluxerr

    def diff(self):
        sorted_flux = sorted(self.mPDCSAPflux)  # mPDCSAPfluxの値をソート

        # brightness_variation_amplitudeを求めるための上下2%を抜く
        lower_bound = int(len(sorted_flux) * 0.02)
        upper_bound = int(len(sorted_flux) * 0.98)
        self.brightness_variation_amplitude = sorted_flux[upper_bound] - sorted_flux[lower_bound]

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

    def flare_energy(self):
        if self.energy is None or len(self.energy) == 0:
            self.flare_number = 0
            self.sum_flare_energy = 0.0
            return

        energy_cor = np.sort(self.energy)  # フレアエネルギーを昇順にソート
        cumenergy = np.array([len(energy_cor) - i for i in range(len(energy_cor))])
        energy4e33_index = np.where(energy_cor >= 4 * 1e33)[0]  # 4*10^33以上のエネルギーを持つフレアのインデックス

        if len(energy4e33_index) > 0:
            self.flare_number = cumenergy[energy4e33_index[0]] - cumenergy[energy4e33_index[-1]] + 1
            # print(f"energy={energy_cor[energy2e33_index[0]:energy2e33_index[-1]+1]}{self.flare_number}、{self.precise_obs_time}")
            self.sum_flare_energy = np.sum(energy_cor[energy4e33_index[0] : energy4e33_index[-1] + 1])
        else:
            self.flare_number = 0
            self.sum_flare_energy = 0.0

    def process_data(self):
        """
        TESS データの読み込みからフレア検出までの一連のプロセスを実行するメソッド。
        """
        # BJD が存在しない or 要素数が少ない場合は処理を中断
        if self.atessBJD is None or len(self.atessBJD) < 2:
            print("Error: BJD が正しく読み込まれていないか、要素数が不足しています。")
            return

        # remove transit data
        self.remove()

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

        # 6) 正確な観測時間を計算
        self.calculate_precise_obs_time()

        # 7) フレアエネルギーに関する計算
        self.flare_energy()

        # sort
        self.diff()

        # 観測時間(最初から最後まで) がゼロでない場合に計算
        # フレア検出の割合を計算し、リストに追加
        if self.tessBJD is not None and len(self.tessBJD) > 1:
            obs_time = self.precise_obs_time
            if obs_time > 0:
                n_flare = len(self.peaktime) if self.peaktime is not None else 0
                self.flare_ratio = n_flare / obs_time
                # リストに追加
                FlareDetector.array_flare_ratio = np.append(FlareDetector.array_flare_ratio, self.flare_ratio)
                # 平均値を再計算
                FlareDetector.average_flare_ratio = np.mean(FlareDetector.array_flare_ratio)
                # 観測時間の中央値をリストに追加 tessBJDの中央値を使用
                FlareDetector.array_observation_time = np.append(FlareDetector.array_observation_time, np.median(self.tessBJD))
                # ライトカーブの変動振幅をリストに追加
                FlareDetector.array_amplitude = np.append(FlareDetector.array_amplitude, self.brightness_variation_amplitude)
                # 観測日数あたりのフレアエネルギーを計算
                sum_flare_energy_ratio = self.sum_flare_energy / obs_time
                FlareDetector.array_energy_ratio = np.append(FlareDetector.array_energy_ratio, sum_flare_energy_ratio)

    def process_data_2(self):
        """
        TESS データの読み込みからフレア検出までの一連のプロセスを実行するメソッド。
        """
        # BJD が存在しない or 要素数が少ない場合は処理を中断
        if self.atessBJD is None or len(self.atessBJD) < 2:
            print("Error: BJD が正しく読み込まれていないか、要素数が不足しています。")
            return

        self.no_remove()

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

        # 6) 正確な観測時間を計算
        self.calculate_precise_obs_time()

        # 7) フレアエネルギーに関する計算
        self.flare_energy()

        # sort
        self.diff()

        # 観測時間(最初から最後まで) がゼロでない場合に計算
        # フレア検出の割合を計算し、リストに追加
        if self.tessBJD is not None and len(self.tessBJD) > 1:
            obs_time = self.precise_obs_time
            if obs_time > 0:
                n_flare = len(self.peaktime) if self.peaktime is not None else 0
                self.flare_ratio = n_flare / obs_time
                # リストに追加
                FlareDetector.array_flare_ratio = np.append(FlareDetector.array_flare_ratio, self.flare_ratio)
                # 平均値を再計算
                FlareDetector.average_flare_ratio = np.mean(FlareDetector.array_flare_ratio)
                # 観測時間の中央値をリストに追加 tessBJDの中央値を使用
                FlareDetector.array_observation_time = np.append(FlareDetector.array_observation_time, np.median(self.tessBJD))
                # ライトカーブの変動振幅をリストに追加
                FlareDetector.array_amplitude = np.append(FlareDetector.array_amplitude, self.brightness_variation_amplitude)
                # 観測日数あたりのフレアエネルギーを計算
                sum_flare_energy_ratio = self.sum_flare_energy / obs_time
                FlareDetector.array_energy_ratio = np.append(FlareDetector.array_energy_ratio, sum_flare_energy_ratio)

    def show_variables(self):
        """
        インスタンス変数とクラス変数の名前、役割、および要素数（配列の場合）を表示するメソッド。
        """
        print("-------- Instance Variables --------")

        # 各インスタンス変数の説明を辞書にまとめる (必要に応じて追加/修正)
        instance_var_info = {
            # "file": "FITSファイルのパス",
            "R_sunstar_ratio": "恒星の半径を太陽半径で割った比",
            # "T_star": "恒星の有効温度 [K]",
            # "tessheader1": "FITSファイルのヘッダ情報",
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
        }
        class_var_info = {
            "array_flare_ratio": "フレア検出割合のリスト",
            "array_observation_time": "観測時間のリスト",
        }

        for var_name, description in instance_var_info.items():
            value = getattr(self, var_name)
            if isinstance(value, np.ndarray):
                # 要素数と簡易情報を表示
                print(f"{var_name}: {description} | type: np.ndarray | length: {value.size}")
            else:
                # ndarray 以外はそのまま出力
                print(f"{var_name}: {description} | value: {value}")

        print("\n-------- Class Variables --------")
        for var_name, description in class_var_info.items():
            value = getattr(FlareDetector, var_name)
            if isinstance(value, np.ndarray):
                # 要素数と簡易情報を表示
                print(f"{var_name}: {description} | type: np.ndarray | length: {value.size}")
            else:
                # ndarray 以外はそのまま出力
                print(f"{var_name}: {description} | value: {value}")
        print("------------------------------------")
