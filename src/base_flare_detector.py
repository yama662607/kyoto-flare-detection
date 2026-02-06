import os
import re
from pathlib import Path

import astropy.io.fits as fits
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from astropy.timeseries import LombScargle
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit


def _ensure_native(array, dtype=None):
    """Plotly などに渡す前にエンディアンをネイティブに揃える。"""
    arr = np.asarray(array, dtype=dtype)
    if arr.dtype.byteorder not in ("=", "|"):
        arr = arr.astype(arr.dtype.newbyteorder("="), copy=False)
    return arr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESS_RESPONSE_PATH = PROJECT_ROOT / "data" / "tess-response-function-v1.0.csv"


def make_rotation_frequency_grid(
    period_min: float = 1.0,
    period_max: float = 8.0,
    n_points: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    frequency = np.linspace(1.0 / period_max, 1.0 / period_min, n_points)
    periods = 1.0 / frequency
    return frequency, periods


ROTATION_FREQUENCY_GRID, _ROTATION_PERIODS = make_rotation_frequency_grid()
_TESS_RESPONSE_CACHE: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
_STAR_INTENSITY_CACHE: dict[float, float] = {}
_REF_INTENSITY: float | None = None


class BaseFlareDetector:
    """
    TESS光度データからのフレア検出とエネルギー推定を行う基本的なクラス。
    """

    array_flare_number = np.array([])
    array_precise_obs_time = np.array([])
    array_flare_ratio = np.array([])
    array_observation_time = np.array([])
    array_energy_ratio = np.array([])
    array_energy_ave = np.array([])
    array_amplitude = np.array([])
    array_starspot = np.array([])
    array_starspot_ratio = np.array([])
    array_data_name = np.array([])
    array_per = np.array([])
    array_per_err = np.array([])
    array_sum_energy = np.array([])
    array_max_energy = np.array([])
    average_flare_ratio = 0.0

    def __init__(
        self,
        file=None,
        R_sunstar_ratio=1.0,
        T_star=5778,
        flux_mean=1.0,
        err_constant_mean=0.0005,
        rot_period=1.0,
        rotation_period_min=1.0,
        rotation_period_max=8.0,
        rotation_n_points=10000,
        buffer_size=10000,
        f_cut_lowpass=3,
        f_cut_spline=6,
        ene_thres_low=5e33,
        ene_thres_high=2e40,
        sector_threshold=None,
        use_sector_mean=False,
        process_data=False,
        run_process_data_2=False,
    ):
        self.file = file
        self.R_sunstar_ratio = R_sunstar_ratio
        self.T_star = T_star
        self.flux_mean = flux_mean
        self.err_constant_mean = err_constant_mean
        self.rot_period = rot_period
        self.rotation_period_min = rotation_period_min
        self.rotation_period_max = rotation_period_max
        self.rotation_n_points = rotation_n_points
        self.sector_threshold = sector_threshold
        self.use_sector_mean = use_sector_mean
        self.d_T_star = 3.58e-5 * self.T_star**2 + 0.249 * self.T_star - 808
        self.buffer_size = buffer_size
        self.f_cut_lowpass = f_cut_lowpass
        self.f_cut_spline = f_cut_spline
        self.ene_thres_low = ene_thres_low
        self.ene_thres_high = ene_thres_high
        self.gap_threshold = 0.1  # デフォルトのギャップ検出閾値
        self.time_offset = 2457000  # For matplotlib plot

        self.debug_print = "hello"

        # Initialize data arrays
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
        self.flux_spline = None
        self.filtered_flux = None

        # Initialize flare detection results
        self.flare_ratio = 0.0
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
        self.power = None
        self.per = None
        self.per_err = 0.0
        self.brightness_variation_amplitude = 0.0
        self.starspot = 0.0
        self.starspot_ratio = 0.0

        self.rotation_ls_method = "auto"

        self._ensure_class_arrays()

        self.load_TESS_data()

        if self.file is not None:
            if run_process_data_2:
                self.process_data(skip_remove=True)
            elif process_data:
                self.process_data()

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
        fname_base = os.path.basename(fname)

        # 正規表現で必要な部分を抽出
        match = re.match(r"(.+)-\d+-\d+-s_lc\.fits$", fname_base)
        if match:
            self.data_name = match.group(1)

        match = re.match(r"[a-z]+\d+-s00(.+)-\d+-\d+-s_lc\.fits$", fname_base)
        data_number = int(match.group(1)) if match else 0

        # FITS データを読み込み
        hdulist = fits.open(fname, memmap=True)
        self.tessheader1 = hdulist[0].header
        data = hdulist[1].data

        # NaN を含む行を除外
        if self.sector_threshold is not None and data_number > self.sector_threshold:
            flux_field, flux_err_field = "SAP_FLUX", "SAP_FLUX_ERR"
        else:
            flux_field, flux_err_field = "PDCSAP_FLUX", "PDCSAP_FLUX_ERR"

        mask = ~np.isnan(data.field(flux_field))
        bjd = data.field("time")[mask]
        pdcsap_flux = data.field(flux_field)[mask]
        pdcsap_flux_err = data.field(flux_err_field)[mask]

        # 光度データの正規化
        flux_mean = np.mean(pdcsap_flux) if self.use_sector_mean else self.flux_mean

        norm_flux = pdcsap_flux / flux_mean
        norm_flux_err = pdcsap_flux_err / flux_mean

        # インスタンス変数へ格納
        self.tessBJD = bjd
        self.mPDCSAPflux = norm_flux
        self.mPDCSAPfluxerr = norm_flux_err
        # Compatibility with some internal methods that expect amPDCSAPflux
        self.atessBJD = bjd
        self.amPDCSAPflux = norm_flux
        self.amPDCSAPfluxerr = norm_flux_err

    def _ensure_class_arrays(self):
        """Ensure per-subclass class arrays exist (legacy-compatible)."""
        cls = self.__class__
        for name in (
            "array_flare_number",
            "array_precise_obs_time",
            "array_flare_ratio",
            "array_observation_time",
            "array_energy_ratio",
            "array_energy_ave",
            "array_amplitude",
            "array_starspot",
            "array_starspot_ratio",
            "array_data_name",
            "array_per",
            "array_per_err",
            "array_sum_energy",
            "array_max_energy",
        ):
            if name not in cls.__dict__:
                setattr(cls, name, np.array([]))
        if "average_flare_ratio" not in cls.__dict__:
            cls.average_flare_ratio = 0.0

    def apply_gap_correction(self):
        """
        時系列データ内のギャップを補正し、データの前後にバッファ領域を追加するメソッド。
        """
        bjd = self.tessBJD.copy()
        flux = self.mPDCSAPflux.copy()
        flux_err = self.mPDCSAPfluxerr.copy()
        buf_size = self.buffer_size

        # ギャップ検出
        diff_bjd = np.diff(bjd)
        gap_indices = np.where(diff_bjd >= self.gap_threshold)[0]

        # ====== ギャップ補正 ======
        for idx in gap_indices:
            flux[idx + 1 :] -= flux[idx + 1] - flux[idx]

        # ====== バッファ追加 ======
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

    def lowpass(self, x, y, fc=3):
        n = len(x)
        dt = 2 / (24 * 60)
        freq = np.linspace(0, 1.0 / dt, n)
        F = np.fft.fft(y) / (n / 2)
        F[0] /= 2
        F2 = F.copy()
        F2[freq > fc] = 0
        filtered = np.fft.ifft(F2)
        return np.real(filtered * n)

    def detrend_flux(self):
        time_ext = self.gtessBJD
        flux_ext = self.gmPDCSAPflux
        flux_err_ext = self.gmPDCSAPfluxerr
        buf_size = self.buffer_size

        self.filtered_flux = self.lowpass(time_ext, flux_ext, fc=self.f_cut_lowpass)
        s1_flux = flux_ext - self.filtered_flux

        fac = 3
        ss_flarecan = np.where(
            (s1_flux <= flux_err_ext * fac)
            | (time_ext < time_ext[10])
            | (time_ext > time_ext[-11])
        )[0]

        baseline_spline = interp1d(
            time_ext[ss_flarecan],
            self.lowpass(
                time_ext[ss_flarecan], flux_ext[ss_flarecan], fc=self.f_cut_spline
            ),
            kind="cubic",
        )

        valid_slice = slice(buf_size, buf_size + len(self.tessBJD))
        self.flux_spline = baseline_spline(time_ext[valid_slice])
        self.s2mPDCSAPflux = flux_ext[valid_slice] - self.flux_spline

    def reestimate_errors(self):
        """
        フラックスの誤差をローカルスキャッターから再推定するメソッド。

        legacy 実装と同等の計算を行うため、各点ごとに
        0.5日窓かつフラックス<=0.005の点で標準偏差を評価する。
        """
        bjd = self.tessBJD
        flux = self.s2mPDCSAPflux
        err = np.ones(len(self.mPDCSAPfluxerr))

        for i in range(len(err)):
            nearby = (np.abs(bjd - bjd[i]) <= 0.5) & (flux <= 0.005)
            err[i] = np.std(flux[nearby])

        # 全体の平均スケールを元のエラーに合わせる
        err *= np.mean(self.mPDCSAPfluxerr) / self.err_constant_mean
        self.mPDCSAPfluxerr_cor = err

    def flaredetect(self):
        flux_detrend = self.s2mPDCSAPflux
        err = self.mPDCSAPfluxerr_cor
        bjd = self.tessBJD

        oversigma_idx = np.where(flux_detrend >= err * 5)[0]
        ss_detect = []
        detecttime = []

        i = 0
        while i <= (len(oversigma_idx) - 3):
            if i >= (len(oversigma_idx) - 1):
                break
            if (oversigma_idx[i + 1] - oversigma_idx[i]) != 1:
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

        starttime, endtime, peaktime, count = [], [], [], []
        diff_bjd = np.diff(bjd)
        overonesigma_idx = np.where(flux_detrend >= err)[0]

        for idx in range(len(ss_detect)):
            n = ss_detect[idx]
            ss_ind = np.where(overonesigma_idx == n)[0]
            if len(endtime) > 0 and np.max(endtime) >= bjd[n]:
                continue
            if len(ss_ind) == 0:
                continue

            ss_val = ss_ind[0]
            k, j = 0, 0
            while (ss_val + k + 1 < len(overonesigma_idx)) and (
                (overonesigma_idx[ss_val + k + 1] - overonesigma_idx[ss_val + k]) == 1
            ):
                k += 1
            while (ss_val + j - 1 >= 0) and (
                (overonesigma_idx[ss_val + j] - overonesigma_idx[ss_val + j - 1]) == 1
            ):
                j -= 1

            if (n + j) <= 30 or (n + k) >= (len(bjd) - 30):
                continue

            a = diff_bjd[(n + j - 10) : (n + k + 10)]
            if len(a) > 0 and np.max(a) >= (2 / (24 * 60)) * 20:
                continue

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

        self.detecttime, self.starttime, self.endtime, self.peaktime = map(
            np.array, [detecttime, starttime, endtime, peaktime]
        )
        self.energy = self.tess_band_energy(np.array(count))

    def flaredetect_check(self):
        N = min(
            len(self.detecttime),
            len(self.starttime),
            len(self.endtime),
            len(self.peaktime),
            len(self.energy) if self.energy is not None else 0,
        )
        (
            detecttime_new,
            starttime_new,
            endtime_new,
            peaktime_new,
            count_new,
            edecay_new,
            a_array,
            b_array,
        ) = [], [], [], [], [], [], [], []

        flux, err, bjd, stime, etime, ptime = (
            self.mPDCSAPflux,
            self.mPDCSAPfluxerr,
            self.tessBJD,
            self.starttime,
            self.endtime,
            self.peaktime,
        )

        flag = 0
        for i in range(N):
            if flag == 1:
                flag = 0
                continue

            ss_pre = np.where(np.abs(bjd - (stime[i] - 0.025)) <= 0.0125)[0]
            val_pre = np.median(flux[ss_pre]) if len(ss_pre) > 0 else np.nan
            ss_post = np.where(np.abs(bjd - (etime[i] + 0.05)) <= 0.025)[0]
            val_post = np.median(flux[ss_post]) if len(ss_post) > 0 else np.nan

            if np.isnan(val_pre) or np.isnan(val_post):
                continue

            t_pre = np.median(bjd[ss_pre]) if len(ss_pre) > 0 else np.nan
            t_post = np.median(bjd[ss_post]) if len(ss_post) > 0 else np.nan
            a_val = (
                (val_post - val_pre) / (t_post - t_pre) if (t_post - t_pre) != 0 else 0
            )
            b_val = val_pre - a_val * t_pre
            flux_diff = flux - (a_val * bjd + b_val)

            peak_idx = np.where(bjd == ptime[i])[0]
            if len(peak_idx) == 0:
                continue
            n_peak = peak_idx[0]

            k, j = 0, 0
            while (n_peak + k < len(flux_diff)) and (
                flux_diff[n_peak + k] >= err[n_peak + k]
            ):
                k += 1
            while (n_peak + j >= 0) and (flux_diff[n_peak + j] >= err[n_peak + j]):
                j -= 1

            n_end, n_start = n_peak + k - 1, n_peak + j + 1
            if n_start < 0 or n_end >= len(flux_diff):
                continue

            ss_flare = np.where((bjd >= bjd[n_start]) & (bjd <= bjd[n_end]))[0]
            if (
                len(ss_flare) <= 1
                or len(np.where((flux_diff[ss_flare] - 3 * err[ss_flare]) >= 0)[0]) <= 1
            ):
                continue

            peak_flux = flux_diff[ss_flare].max()
            peak_loc = ss_flare[np.where(flux_diff[ss_flare] == peak_flux)[0][0]]

            ll = 0
            while ((peak_loc + ll) < len(flux_diff)) and (
                flux_diff[peak_loc + ll] >= peak_flux * np.exp(-1)
            ):
                ll += 1
            if ll == 0:
                continue

            edecay_new.append(bjd[peak_loc + ll] - bjd[peak_loc])
            a_array.append(a_val)
            b_array.append(b_val)
            starttime_new.append(bjd[n_start])
            endtime_new.append(bjd[n_end])
            peaktime_new.append(ptime[i])
            detecttime_new.append(self.detecttime[i])
            count_new.append(np.sum(flux_diff[n_start : n_end + 1]))

            if i < (N - 1) and bjd[n_end] >= stime[i + 1]:
                flag = 1

        (
            self.detecttime,
            self.starttime,
            self.endtime,
            self.peaktime,
            self.a_i,
            self.b_i,
            self.edecay,
        ) = map(
            np.array,
            [
                detecttime_new,
                starttime_new,
                endtime_new,
                peaktime_new,
                a_array,
                b_array,
                edecay_new,
            ],
        )
        self.energy = (
            self.tess_band_energy(np.array(count_new))
            if len(count_new) > 0
            else np.array([])
        )
        self.duration = (
            self.endtime - self.starttime + (2 / (24 * 60))
            if len(starttime_new) > 0
            else np.array([])
        )

    def planck(self, wav, T):
        h, c, k = 6.626e-34, 3.0e8, 1.38e-23
        return (2.0 * h * c**2) / ((wav**5) * (np.exp(h * c / (wav * k * T)) - 1.0))

    def tess_band_energy(self, count):
        """
        TESS帯域でのフレアエネルギー（erg）を推定する。
        T_flare = 10000K の黒体近似を想定。
        """
        response = self._get_tess_response()
        if response is None:
            return np.array([])

        wave, resp, dw = response
        dt = 120.0  # TESS 2-min cadence (seconds)
        Rstar = 695510e5 * self.R_sunstar_ratio  # cm
        sigma_SB = 5.67e-5  # erg cm^-2 s^-1 K^-4

        star_intensity_ratio = self._get_star_intensity_ratio(wave, resp, dw)
        if star_intensity_ratio == 0:
            return np.array([])

        # Eq. 11 in thesis: L_flare = sigma_SB * T_flare^4 * A_flare
        # A_flare = C_flare * pi * R_star^2 * (L_star_TESS / L_flare_TESS)
        # E_flare = sum(L_flare * dt) = sigma_SB * T_flare^4 * pi * R_star^2 * (L_star_TESS / L_flare_TESS) * dt * sum(C_flare)

        area_factor = (np.pi * Rstar**2) * star_intensity_ratio
        return sigma_SB * (10000**4) * area_factor * dt * count

    @staticmethod
    def _get_tess_response():
        global _TESS_RESPONSE_CACHE
        if _TESS_RESPONSE_CACHE is not None:
            return _TESS_RESPONSE_CACHE
        try:
            data = np.loadtxt(TESS_RESPONSE_PATH, delimiter=",")
        except FileNotFoundError:
            print("Error: TESS応答関数のCSVファイルが見つかりません。")
            return None
        wave = data[:, 0]
        resp = data[:, 1]
        dw = np.hstack([np.diff(wave), 0])
        _TESS_RESPONSE_CACHE = (wave, resp, dw)
        return _TESS_RESPONSE_CACHE

    def _get_star_intensity_ratio(self, wave, resp, dw):
        global _STAR_INTENSITY_CACHE, _REF_INTENSITY
        key = float(self.T_star)
        if key not in _STAR_INTENSITY_CACHE:
            star_intensity = np.sum(dw * self.planck(wave * 1e-9, self.T_star) * resp)
            _STAR_INTENSITY_CACHE[key] = star_intensity
        star_intensity = _STAR_INTENSITY_CACHE[key]

        if _REF_INTENSITY is None:
            _REF_INTENSITY = np.sum(dw * self.planck(wave * 1e-9, 10000) * resp)

        if _REF_INTENSITY == 0:
            return 0.0
        return star_intensity / _REF_INTENSITY

    def calculate_precise_obs_time(self):
        """
        データ点数に基づく有効観測時間（日）を計算する。
        2分間隔（120秒）のTESSデータを想定。
        """
        if self.tessBJD is not None:
            # 2分 = 2 / (24 * 60) 日
            cadence_day = 2 / (24 * 60)
            self.precise_obs_time = len(self.tessBJD) * cadence_day
        else:
            self.precise_obs_time = 0.0

    def flare_energy(self, energy_threshold_low, energy_threshold_high):
        if self.energy is None or len(self.energy) == 0:
            self.flare_number, self.sum_flare_energy = 0, 0.0
            return

        energy_cor = np.sort(self.energy)
        energy_mask = (energy_cor >= energy_threshold_low) & (
            energy_cor <= energy_threshold_high
        )

        if np.any(energy_mask):
            self.flare_number = np.sum(energy_mask)
            self.sum_flare_energy = np.sum(energy_cor[energy_mask])
        else:
            self.flare_number, self.sum_flare_energy = 0, 0.0

    def flux_diff(self, min_percent: float = 0.02, max_percent: float = 0.98):
        sorted_flux = sorted(self.mPDCSAPflux)
        lower_bound = int(len(sorted_flux) * min_percent)
        upper_bound = int(len(sorted_flux) * max_percent)
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

    def show_variables(self):
        """インスタンス／クラス変数の概要を表示する。"""
        instance_vars = {
            "file": self.file,
            "tessBJD_length": len(self.tessBJD) if self.tessBJD is not None else 0,
            "flare_number": self.flare_number,
            "sum_flare_energy": self.sum_flare_energy,
            "precise_obs_time": self.precise_obs_time,
            "flare_ratio": getattr(self, "flare_ratio", None),
        }
        print("-------- Instance Summary --------")
        for key, value in instance_vars.items():
            print(f"{key}: {value}")

        print("\n-------- Class Summary --------")
        print(f"array_flare_ratio length: {len(BaseFlareDetector.array_flare_ratio)}")
        print(f"average_flare_ratio: {BaseFlareDetector.average_flare_ratio}")

    def rotation_period(
        self,
        use_gaussian_fit: bool = True,
        show_plot: bool = False,
        w_min: float = 0.010,
        w_max: float = 0.050,
        w_step: float = 0.001,
        main_window: float = 0.02,
        frac_edge: float = 0.13,
    ):
        """
        Lomb-Scargleペリオドグラムから自転周期を推定するメソッド。

        Parameters
        ----------
        use_gaussian_fit : bool, optional
            Trueの場合、ガウスフィッティングによる誤差計算を行う。デフォルトはTrue。
        show_plot : bool, optional
            Trueの場合、プロットを自動表示（legacy互換）。デフォルトはFalse。
        w_min, w_max, w_step : float, optional
            ガウスフィッティングのウィンドウスイープパラメータ
        main_window : float, optional
            メインのフィッティングウィンドウ半幅
        frac_edge : float, optional
            フィット窓の端でのパワー閾値（ピークの何割以下まで落ちているか）
        """
        frequency, periods = make_rotation_frequency_grid(
            period_min=self.rotation_period_min,
            period_max=self.rotation_period_max,
            n_points=self.rotation_n_points,
        )
        lomb = LombScargle(self.tessBJD - self.tessBJD[0], self.mPDCSAPflux)
        method = getattr(self, "rotation_ls_method", "auto")
        self.power = lomb.power(frequency, method=method, assume_regular_frequency=True)
        self.frequency = frequency

        idx_max = int(np.argmax(self.power))
        f0_guess = float(frequency[idx_max])

        if not use_gaussian_fit:
            # 従来のFWHMベースの誤差推定
            self.per = periods[idx_max]
            half_max_power = np.max(self.power) / 2
            aa = np.where(self.power > half_max_power)[0]
            self.per_err = abs(periods[aa[-1]] - periods[aa[0]]) / 2
            self.per_err_minus = self.per_err
            self.per_err_plus = self.per_err
            return

        # ガウスフィッティングによる誤差計算
        def gauss_c0(f, A, f0, sigma):
            """ベースライン0のガウス関数"""
            return A * np.exp(-0.5 * ((f - f0) / sigma) ** 2)

        def fit_gaussian_peak(freq, power, f0_guess, window, frac_edge=0.05):
            """ピーク周辺でガウスフィットを行う"""
            m = (freq > f0_guess - window) & (freq < f0_guess + window)
            f_fit = freq[m]
            p_fit = power[m]
            if f_fit.size < 30:
                return None

            pmax = float(np.max(p_fit))
            if pmax <= 0:
                return None
            # 端が十分落ちていない窓は捨てる
            if (p_fit[0] > frac_edge * pmax) or (p_fit[-1] > frac_edge * pmax):
                return None

            A_guess = pmax
            sigma_guess = (f_fit.max() - f_fit.min()) / 6.0
            p0 = [A_guess, f0_guess, sigma_guess]

            try:
                popt, pcov = curve_fit(
                    gauss_c0,
                    f_fit,
                    p_fit,
                    p0=p0,
                    maxfev=200000,
                    bounds=(
                        [0.0, f0_guess - window, 1e-8],
                        [np.inf, f0_guess + window, window],
                    ),
                )
                perr = np.sqrt(np.diag(pcov))
                resid = p_fit - gauss_c0(f_fit, *popt)
                resid_rms = np.sqrt(np.mean(resid**2))

                return {
                    "window": window,
                    "f_fit": f_fit,
                    "p_fit": p_fit,
                    "popt": popt,
                    "perr": perr,
                    "resid": resid,
                    "resid_rms": resid_rms,
                }
            except Exception:
                return None

        # ウィンドウスイープ
        windows = np.arange(w_min, w_max + 1e-12, w_step)
        sigma_list = np.full_like(windows, np.nan, dtype=float)
        results = {}

        for i, w in enumerate(windows):
            out = fit_gaussian_peak(
                frequency, self.power, f0_guess, float(w), frac_edge=frac_edge
            )
            if out is None:
                continue
            A, f0, sigma_f = out["popt"]
            if (sigma_f <= 0) or (not np.isfinite(sigma_f)):
                continue
            sigma_list[i] = sigma_f
            results[float(w)] = out

        # main_windowを成功しているところに合わせる
        main_window = float(np.round(main_window / w_step) * w_step)
        if main_window not in results:
            ok = np.array(sorted(results.keys()))
            if ok.size == 0:
                # フォールバック: 従来のFWHMベースの誤差推定
                self.per = periods[idx_max]
                half_max_power = np.max(self.power) / 2
                aa = np.where(self.power > half_max_power)[0]
                self.per_err = abs(periods[aa[-1]] - periods[aa[0]]) / 2
                self.per_err_minus = self.per_err
                self.per_err_plus = self.per_err
                return
            main_window = float(ok[np.argmin(np.abs(ok - main_window))])

        main = results[main_window]
        A_fit, f0_fit, sigma_f_fit = main["popt"]

        # 周期と1σ誤差を計算
        self.per = 1.0 / f0_fit
        P0 = self.per
        P_low = 1.0 / (f0_fit + sigma_f_fit)  # f↑ -> P↓
        P_high = 1.0 / (f0_fit - sigma_f_fit)  # f↓ -> P↑

        self.per_err_minus = P0 - P_low
        self.per_err_plus = P_high - P0
        self.per_err = 0.5 * (self.per_err_minus + self.per_err_plus)

        # 診断用属性を保存
        self.f0_fit = f0_fit
        self.sigma_f = sigma_f_fit
        self.main_window = main_window
        self.sigma_list = sigma_list
        self.windows = windows
        self._gaussian_fit_results = results
        self._gaussian_fit_main = main

        # legacy互換: 自動プロット
        if show_plot:
            self.plot_power_spectrum()
            self.plot_rotation_period()

    def plot_power_spectrum(self, save_path: str | None = None, dpi: int = 300):
        """
        Lomb-Scargleパワースペクトル全体図を表示する。

        Parameters
        ----------
        save_path : str, optional
            保存先パス。Noneの場合は保存せず表示のみ
        dpi : int
            解像度（デフォルト300）
        """
        if not hasattr(self, "frequency") or self.frequency is None:
            print("rotation_period()を先に呼び出してください。")
            return

        # 論文品質の設定
        plt.rcParams["xtick.major.width"] = 1.5
        plt.rcParams["ytick.major.width"] = 1.5
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams["xtick.major.size"] = 7
        plt.rcParams["ytick.major.size"] = 7
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42

        plt.figure(figsize=(9, 3))
        plt.plot(self.frequency, self.power, lw=1)

        # ピーク位置に垂直線
        if hasattr(self, "f0_fit"):
            plt.axvline(
                self.f0_fit, ls="--", color="red", label=f"f0 = {self.f0_fit:.5f}"
            )
            plt.legend(loc="upper right")

        plt.xlabel("Frequency [1/day]")
        plt.ylabel("LS Power")
        plt.title(f"Lomb-Scargle Power Spectrum ({self.data_name})")
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"保存しました: {save_path}")

        plt.show()

    def plot_rotation_period(self, save_path: str | None = None, dpi: int = 300):
        """
        自転周期推定の診断プロット（3段）を表示する。

        Parameters
        ----------
        save_path : str, optional
            保存先パス。Noneの場合は保存せず表示のみ
        dpi : int
            解像度（デフォルト300）
        """
        if not hasattr(self, "_gaussian_fit_main") or self._gaussian_fit_main is None:
            print(
                "ガウスフィッティングが実行されていません。rotation_period(use_gaussian_fit=True)を先に呼び出してください。"
            )
            return

        main = self._gaussian_fit_main
        freq = self.frequency
        power = self.power

        # 論文品質の設定
        plt.rcParams["xtick.major.width"] = 1.5
        plt.rcParams["ytick.major.width"] = 1.5
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams["xtick.major.size"] = 7
        plt.rcParams["ytick.major.size"] = 7
        plt.rcParams["xtick.minor.visible"] = True
        plt.rcParams["ytick.minor.visible"] = True
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42

        fig, ax = plt.subplots(3, 1, figsize=(10, 10))

        # ガウス関数
        def gauss_c0(f, A, f0, sigma):
            return A * np.exp(-0.5 * ((f - f0) / sigma) ** 2)

        A_fit, f0_fit, sigma_f_fit = main["popt"]

        # 1) パワースペクトル + ガウスフィット
        ax[0].plot(freq, power, lw=1, label="LS power")
        ax[0].plot(
            main["f_fit"],
            main["p_fit"],
            ".",
            ms=3,
            label=f"Fit region (±{self.main_window:.3f})",
        )
        ax[0].axvline(f0_fit, ls="--", color="red", label=f"f0 = {f0_fit:.5f}")
        ax[0].axvline(f0_fit - sigma_f_fit, ls=":", color="orange")
        ax[0].axvline(f0_fit + sigma_f_fit, ls=":", color="orange", label="f0±σ_f")
        # フィット曲線
        f_dense = np.linspace(f0_fit - 2 * sigma_f_fit, f0_fit + 2 * sigma_f_fit, 200)
        ax[0].plot(
            f_dense,
            gauss_c0(f_dense, A_fit, f0_fit, sigma_f_fit),
            "r-",
            lw=2,
            label="Gaussian fit",
        )
        ax[0].set_xlabel("Frequency [1/day]")
        ax[0].set_ylabel("LS Power")
        ax[0].set_title(
            f"Rotation Period: P = {self.per:.4f} (+{self.per_err_plus:.4f} / -{self.per_err_minus:.4f}) day"
        )
        ax[0].legend(loc="upper right", fontsize=9)

        # 2) 残差
        ax[1].plot(main["f_fit"], main["resid"], lw=1, color="blue")
        ax[1].axhline(0, ls="--", color="gray")
        ax[1].set_xlabel("Frequency [1/day]")
        ax[1].set_ylabel("Residual")
        ax[1].set_title(
            f"Residuals (window=±{self.main_window:.3f}); RMS={main['resid_rms']:.5f}"
        )

        # 3) σ_f vs window
        ax[2].plot(self.windows, self.sigma_list, lw=1, marker="o", ms=3, label="σ_f")
        ax[2].axvline(
            self.main_window,
            ls="--",
            color="red",
            label=f"Main window = {self.main_window:.3f}",
        )
        ax[2].set_xlabel("Window half-width")
        ax[2].set_ylabel("σ_f")
        ax[2].set_title("σ_f vs Window (stability check)")
        ax[2].legend(loc="upper right", fontsize=9)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            print(f"保存しました: {save_path}")

        plt.show()

    def remove(self):
        # This method is intended to be overridden by subclasses for specific data removal.
        pass

    def process_data(self, ene_thres_low=None, ene_thres_high=None, skip_remove=False):
        """
        TESS データの読み込みからフレア検出までの一連のプロセスを実行するメソッド。

        Parameters
        ----------
        ene_thres_low : float, optional
            最小エネルギー閾値 (erg)。指定しない場合はインスタンス変数を使用。
        ene_thres_high : float, optional
            最大エネルギー閾値 (erg)。指定しない場合はインスタンス変数を使用。
        skip_remove : bool, optional
            True にすると `remove()` をスキップし、トランジット除去を行わない処理流を実行します。
        """
        if self.tessBJD is None or len(self.tessBJD) < 2:
            print("Error: BJD が正しく読み込まれていないか、要素数が不足しています。")
            return

        # エネルギー閾値の設定（引数があれば上書き）
        low_threshold = (
            ene_thres_low if ene_thres_low is not None else self.ene_thres_low
        )
        high_threshold = (
            ene_thres_high if ene_thres_high is not None else self.ene_thres_high
        )

        if not skip_remove:
            self.remove()
        self.apply_gap_correction()
        self.detrend_flux()
        self.reestimate_errors()
        self.flaredetect()
        self.flaredetect_check()
        self.calculate_precise_obs_time()
        self.flare_energy(
            energy_threshold_low=low_threshold, energy_threshold_high=high_threshold
        )
        self.flux_diff()
        self.rotation_period()

        if (
            self.tessBJD is not None
            and len(self.tessBJD) > 1
            and self.precise_obs_time > 0
        ):
            self._ensure_class_arrays()
            cls = self.__class__

            flare_ratio = self.flare_number / self.precise_obs_time
            cls.array_flare_number = np.append(
                cls.array_flare_number, self.flare_number
            )
            cls.array_precise_obs_time = np.append(
                cls.array_precise_obs_time, self.precise_obs_time
            )
            cls.array_flare_ratio = np.append(cls.array_flare_ratio, flare_ratio)
            cls.average_flare_ratio = np.mean(cls.array_flare_ratio)

            sum_flare_energy_ratio = self.sum_flare_energy / self.precise_obs_time
            cls.array_energy_ratio = np.append(
                cls.array_energy_ratio, sum_flare_energy_ratio
            )
            cls.array_sum_energy = np.append(
                cls.array_sum_energy, self.sum_flare_energy
            )

            max_flare_energy = (
                float(np.max(self.energy))
                if self.energy is not None and len(self.energy) > 0
                else np.nan
            )
            cls.array_max_energy = np.append(cls.array_max_energy, max_flare_energy)

            cls.array_observation_time = np.append(
                cls.array_observation_time, np.median(self.tessBJD)
            )
            cls.array_amplitude = np.append(
                cls.array_amplitude, self.brightness_variation_amplitude
            )
            cls.array_starspot = np.append(cls.array_starspot, self.starspot)
            cls.array_starspot_ratio = np.append(
                cls.array_starspot_ratio, self.starspot_ratio
            )
            cls.array_data_name = np.append(cls.array_data_name, self.data_name)
            cls.array_per = np.append(cls.array_per, self.per)
            cls.array_per_err = np.append(cls.array_per_err, self.per_err)

    def plt_flare(self, title: str | None = None, save_path: str | None = None):
        """
        データ全体の光度曲線をmatplotlibでプロット（legacy互換）。

        Parameters
        ----------
        title : str, optional
            グラフタイトル。Noneの場合はdata_nameを使用
        save_path : str, optional
            保存先パス。Noneの場合は保存せず表示のみ
        """
        if self.tessBJD is None:
            return

        # legacy完全互換のスタイル設定
        plt.rcParams["xtick.major.width"] = 1.5
        plt.rcParams["ytick.major.width"] = 1.5
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams["xtick.major.size"] = 7
        plt.rcParams["ytick.major.size"] = 7
        plt.rcParams["xtick.minor.visible"] = True
        plt.rcParams["ytick.minor.visible"] = True
        plt.rcParams["xtick.minor.width"] = 1.5
        plt.rcParams["ytick.minor.width"] = 1.5
        plt.rcParams["xtick.minor.size"] = 4
        plt.rcParams["ytick.minor.size"] = 4
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42

        # legacyと同じfigsize
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(14, 7))
        fig.subplots_adjust(hspace=0.1)

        # 1つ目のサブプロット: 生の光度曲線
        axs[0].plot(
            self.tessBJD,
            self.mPDCSAPflux,
            color="black",
            linestyle="-",
            label="Normalized Flux",
            lw=0.5,
        )
        axs[0].set_ylabel("Normalized Flux", fontsize=17)
        axs[0].tick_params(labelsize=13)

        # 2つ目のサブプロット: デトレンド後
        if self.s2mPDCSAPflux is not None:
            axs[1].plot(
                self.tessBJD, self.s2mPDCSAPflux, color="black", linestyle="-", lw=0.5
            )

            # フレアのピーク位置を線で示す
            if self.peaktime is not None:
                for peak in self.peaktime:
                    axs[1].axvline(
                        x=peak,
                        ymin=0.8,
                        ymax=0.85,
                        color="red",
                        linestyle="-",
                        linewidth=1.5,
                    )

            axs[1].set_xlabel(f"Time (day) (BJD - {self.time_offset})", fontsize=17)
            axs[1].set_ylabel("Detrended Flux", fontsize=17)
            axs[1].tick_params(labelsize=13)
            plt.tick_params(labelsize=13)
            leg = plt.legend(loc="upper right", fontsize=11)
            leg.get_frame().set_alpha(0)

        # y軸ラベルの位置を揃える
        axs[0].yaxis.set_label_coords(-0.05, 0.5)
        axs[1].yaxis.set_label_coords(-0.05, 0.5)

        # タイトル
        plot_title = title if title else self.data_name
        fig.suptitle(plot_title, fontsize=17, y=0.93)

        if save_path is not None:
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
            print(f"保存しました: {save_path}")

        plt.show()

    def plot_flare(self):
        if self.tessBJD is None:
            return
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        x_native = np.asarray(self.tessBJD, dtype=np.float64)
        y_flux = np.asarray(self.mPDCSAPflux, dtype=np.float64)
        fig.add_trace(
            go.Scatter(
                x=x_native,
                y=y_flux,
                mode="lines",
                line=dict(color="black", width=1),
                name="Normalized Flux",
            ),
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="Normalized Flux", row=1, col=1)
        if self.s2mPDCSAPflux is not None:
            y_detrended = np.asarray(self.s2mPDCSAPflux, dtype=np.float64)
            fig.add_trace(
                go.Scatter(
                    x=x_native,
                    y=y_detrended,
                    mode="lines",
                    line=dict(color="black", width=1),
                    name="Detrended Flux",
                ),
                row=2,
                col=1,
            )
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
            fig.update_xaxes(
                title_text=f"Time (BJD - {self.time_offset})", row=2, col=1
            )
            fig.update_yaxes(title_text="Detrended Flux", row=2, col=1)
        fig.update_layout(
            title_text=f"Flare Detection Graph ({self.data_name})", height=900
        )
        fig.show()

    def plot_energy(self):
        if self.energy is None or len(self.energy) == 0:
            return
        energy_cor = np.sort(self.energy)
        cumenergy = np.arange(len(energy_cor), 0, -1)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=energy_cor,
                y=cumenergy / self.precise_obs_time,
                mode="lines",
                line=dict(color="gray", width=2),
                name="All Sector",
                line_shape="hv",
            )
        )
        fig.update_layout(title_text=f"Flare Energy Distribution ({self.data_name})")
        fig.update_xaxes(title_text="Flare Energy [erg]", type="log")
        fig.update_yaxes(title_text=r"Cumulative Number [day$^{-1}$]", type="log")

    def plot_flare_matplotlib(self, save_path=None, dpi=300):
        """
        論文投稿用の高品質なmatplotlibプロット

        Parameters
        ----------
        save_path : str, optional
            保存先パス。Noneの場合はデフォルトで "{data_name}_light_curve.pdf" に保存
        dpi : int
            解像度（デフォルト300）
        """
        if self.tessBJD is None:
            return

        # 論文品質の設定を適用
        plt.rcParams["xtick.major.width"] = 1.5
        plt.rcParams["ytick.major.width"] = 1.5
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams["xtick.major.size"] = 7
        plt.rcParams["ytick.major.size"] = 7
        plt.rcParams["xtick.minor.visible"] = True
        plt.rcParams["ytick.minor.visible"] = True
        plt.rcParams["xtick.minor.width"] = 1.5
        plt.rcParams["ytick.minor.width"] = 1.5
        plt.rcParams["xtick.minor.size"] = 4
        plt.rcParams["ytick.minor.size"] = 4
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["pdf.fonttype"] = 42  # PDFフォント埋め込み
        plt.rcParams["ps.fonttype"] = 42

        # 2つのサブプロットを作成 (生の光度曲線とデトレンド後)
        fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(13, 8))
        fig.subplots_adjust(hspace=0.1)  # サブプロット間の垂直方向のスペースを調整

        # 1つ目のサブプロット: 生の光度曲線
        axs[0].plot(
            self.tessBJD,
            self.mPDCSAPflux,
            color="black",
            linestyle="-",
            label="Normalized Flux",
            lw=0.5,
        )
        axs[0].set_ylabel("Normalized Flux", fontsize=17)
        axs[0].tick_params(labelsize=13)

        # 2つ目のサブプロット: デトレンド後
        if self.s2mPDCSAPflux is not None:
            axs[1].plot(
                self.tessBJD,
                self.s2mPDCSAPflux,
                color="black",
                linestyle="-",
                label="Detrended Flux",
                lw=0.5,
            )

            # フレアのピーク位置を線で示す
            if self.peaktime is not None:
                for peak in self.peaktime:
                    axs[1].axvline(
                        x=peak,
                        ymin=0.8,
                        ymax=0.85,
                        color="red",
                        linestyle="-",
                        linewidth=1.5,
                    )

            axs[1].set_xlabel(f"Time (day) (BJD - {self.time_offset})", fontsize=17)
            axs[1].set_ylabel("Detrended Flux", fontsize=17)
            axs[1].tick_params(labelsize=13)
            plt.tick_params(labelsize=11)
            leg = plt.legend(loc="upper right", fontsize=11)
            leg.get_frame().set_alpha(0)  # 背景を完全に透明にする

        # 保存
        if save_path is None:
            save_path = f"{self.data_name}_light_curve.pdf"
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=dpi)

        plt.show()

    def plot_energy_matplotlib(self, save_path=None, dpi=300):
        """
        論文投稿用のエネルギー分布プロット

        Parameters
        ----------
        save_path : str, optional
            保存先パス。Noneの場合はデフォルトで "{data_name}_ffd.pdf" に保存
        dpi : int
            解像度（デフォルト300）
        """
        if self.energy is None or len(self.energy) == 0:
            return

        # 論文品質の設定を適用
        plt.rcParams["xtick.major.width"] = 1.5
        plt.rcParams["ytick.major.width"] = 1.5
        plt.rcParams["axes.linewidth"] = 1.5
        plt.rcParams["xtick.major.size"] = 7
        plt.rcParams["ytick.major.size"] = 7
        plt.rcParams["xtick.minor.visible"] = True
        plt.rcParams["ytick.minor.visible"] = True
        plt.rcParams["xtick.minor.width"] = 1.5
        plt.rcParams["ytick.minor.width"] = 1.5
        plt.rcParams["xtick.minor.size"] = 4
        plt.rcParams["ytick.minor.size"] = 4
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["font.family"] = "Arial"
        plt.rcParams["pdf.fonttype"] = 42
        plt.rcParams["ps.fonttype"] = 42

        energy_cor = np.sort(self.energy)
        cumenergy = np.arange(len(energy_cor), 0, -1)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(
            energy_cor,
            cumenergy / self.precise_obs_time,
            drawstyle="steps-post",
            color="black",
            lw=1.5,
        )
        ax.plot(
            energy_cor,
            cumenergy / self.precise_obs_time,
            marker="o",
            ls="none",
            markerfacecolor="white",
            markeredgecolor="black",
            mew=1.5,
            ms=7,
            label="Observed Flares",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Flare Energy [erg]", fontsize=14)
        ax.set_ylabel(r"Cumulative Frequency [day$^{-1}$]", fontsize=14)
        ax.set_title(f"Flare Frequency Distribution of {self.data_name}", fontsize=16)
        ax.legend(loc="lower left", frameon=False, fontsize=12)
        ax.grid(True, which="both", ls="--", color="lightgray", lw=0.5)

        if save_path is None:
            save_path = f"{self.data_name}_ffd.pdf"
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.show()
