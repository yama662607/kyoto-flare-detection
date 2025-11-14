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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESS_RESPONSE_PATH = PROJECT_ROOT / "data" / "tess-response-function-v1.0.csv"
ROTATION_FREQUENCY_GRID = np.linspace(1 / 8.0, 1.0, 10000)
_ROTATION_PERIODS = 1 / ROTATION_FREQUENCY_GRID
_TESS_RESPONSE_CACHE: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
_STAR_INTENSITY_CACHE: dict[float, float] = {}
_REF_INTENSITY: float | None = None


class BaseFlareDetector:
    """
    TESS光度データからのフレア検出とエネルギー推定を行う基本的なクラス。
    """

    array_flare_ratio = np.array([])
    array_observation_time = np.array([])
    array_energy_ratio = np.array([])
    array_amplitude = np.array([])
    array_starspot = np.array([])
    array_starspot_ratio = np.array([])
    array_data_name = np.array([])
    array_per = np.array([])
    array_per_err = np.array([])
    average_flare_ratio = 0.0

    def __init__(
        self,
        file=None,
        R_sunstar_ratio=1.0,
        T_star=5778,
        flux_mean=1.0,
        err_constant_mean=0.0005,
        rot_period=1.0,
        buffer_size=10000,
        f_cut_lowpass=3,
        f_cut_spline=6,
        ene_thres_low=5e33,
        ene_thres_high=2e40,
        sector_threshold=None,
        process_data=False,
        run_process_data_2=False,
    ):
        self.file = file
        self.R_sunstar_ratio = R_sunstar_ratio
        self.T_star = T_star
        self.flux_mean = flux_mean
        self.err_constant_mean = err_constant_mean
        self.rot_period = rot_period
        self.sector_threshold = sector_threshold
        self.d_T_star = 3.58e-5 * self.T_star**2 + 0.249 * self.T_star - 808
        self.buffer_size = buffer_size
        self.f_cut_lowpass = f_cut_lowpass
        self.f_cut_spline = f_cut_spline
        self.ene_thres_low = ene_thres_low
        self.ene_thres_high = ene_thres_high
        self.gap_threshold = 0.1  # デフォルトのギャップ検出閾値
        self.time_offset = 2457000  # For matplotlib plot

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
        self.per = None
        self.per_err = 0.0
        self.brightness_variation_amplitude = 0.0
        self.starspot = 0.0
        self.starspot_ratio = 0.0

        self.load_TESS_data()

        if self.file is not None:
            if run_process_data_2:
                self.process_data(skip_remove=True)
            elif process_data:
                self.process_data()

    def load_TESS_data(self):
        if self.atessBJD is not None:
            return
        if self.file is None:
            print("Error: ファイルパスが指定されていません。")
            return

        fname = self.file
        fname_base = os.path.basename(fname)
        match = re.match(r"(.+)-\d+-\d+-s_lc\.fits$", fname_base)
        if match:
            self.data_name = match.group(1)

        # セクタ番号を抽出
        match = re.match(r"[a-z]+\d+-s00(.+)-\d+-\d+-s_lc\.fits$", fname_base)
        data_number = int(match.group(1)) if match else 0

        hdulist = fits.open(fname, memmap=True)
        self.tessheader1 = hdulist[0].header
        data = hdulist[1].data

        # セクタ分岐: sector_thresholdが設定されている場合のみ分岐
        if self.sector_threshold is not None and data_number > self.sector_threshold:
            flux_field = "SAP_FLUX"
            flux_err_field = "SAP_FLUX_ERR"
        else:
            flux_field = "PDCSAP_FLUX"
            flux_err_field = "PDCSAP_FLUX_ERR"

        mask = ~np.isnan(data.field(flux_field))
        bjd = data.field("time")[mask]
        pdcsap_flux = data.field(flux_field)[mask]
        pdcsap_flux_err = data.field(flux_err_field)[mask]

        norm_flux = pdcsap_flux / self.flux_mean
        norm_flux_err = pdcsap_flux_err / self.flux_mean

        self.atessBJD = bjd
        self.amPDCSAPflux = norm_flux
        self.amPDCSAPfluxerr = norm_flux_err

        # Initially, set tessBJD to the loaded data. It can be overridden by child classes.
        self.tessBJD = self.atessBJD
        self.mPDCSAPflux = self.amPDCSAPflux
        self.mPDCSAPfluxerr = self.amPDCSAPfluxerr

    def apply_gap_correction(self):
        bjd = self.tessBJD.copy()
        flux = self.mPDCSAPflux.copy()
        flux_err = self.mPDCSAPfluxerr.copy()
        buf_size = self.buffer_size

        diff_bjd = np.diff(bjd)
        gap_indices = np.where(diff_bjd >= self.gap_threshold)[0]

        for idx in gap_indices:
            flux[idx + 1 :] -= flux[idx + 1] - flux[idx]

        flux_ext = np.hstack([np.full(buf_size, flux[0]), flux, np.full(buf_size, flux[-1])])
        flux_err_ext = np.hstack([np.full(buf_size, 0.0001), flux_err, np.full(buf_size, 0.0001)])

        dt_min = 2 / (24 * 60)
        a = np.arange(buf_size) * dt_min
        bjd_ext = np.hstack([(a - a[-1] - dt_min + bjd[0]), bjd, (a + dt_min + bjd[-1])])

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
        ss_flarecan = np.where((s1_flux <= flux_err_ext * fac) | (time_ext < time_ext[10]) | (time_ext > time_ext[-11]))[0]

        baseline_spline = interp1d(
            time_ext[ss_flarecan],
            self.lowpass(time_ext[ss_flarecan], flux_ext[ss_flarecan], fc=self.f_cut_spline),
            kind="cubic",
        )

        valid_slice = slice(buf_size, buf_size + len(self.tessBJD))
        self.flux_spline = baseline_spline(time_ext[valid_slice])
        self.s2mPDCSAPflux = flux_ext[valid_slice] - self.flux_spline

    def reestimate_errors(self):
        bjd = self.tessBJD
        flux = self.s2mPDCSAPflux
        err = np.ones(len(self.mPDCSAPfluxerr))
        quiet_mask = flux <= 0.005
        quiet_bjd = bjd[quiet_mask]
        quiet_flux = flux[quiet_mask]

        if len(quiet_bjd) == 0:
            err[:] = np.nan
        else:
            window = 0.5
            n_quiet = len(quiet_bjd)
            start_idx = 0
            end_idx = 0
            prefix = np.concatenate(([0.0], np.cumsum(quiet_flux, dtype=float)))
            prefix_sq = np.concatenate(([0.0], np.cumsum(quiet_flux**2, dtype=float)))
            for i, center in enumerate(bjd):
                left = center - window
                right = center + window

                while start_idx < n_quiet and quiet_bjd[start_idx] < left:
                    start_idx += 1
                if end_idx < start_idx:
                    end_idx = start_idx
                while end_idx < n_quiet and quiet_bjd[end_idx] <= right:
                    end_idx += 1

                if start_idx == end_idx:
                    err[i] = np.nan
                    continue

                samples = end_idx - start_idx
                if samples == 1:
                    err[i] = 0.0
                    continue
                sum_val = prefix[end_idx] - prefix[start_idx]
                sum_sq = prefix_sq[end_idx] - prefix_sq[start_idx]
                mean_val = sum_val / samples
                var = max((sum_sq / samples) / 1.0 - mean_val**2, 0.0)
                err[i] = np.sqrt(var)  # [perf] prefix-sum variance to avoid slicing

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
            while ((i + k + 1) < len(oversigma_idx)) and ((oversigma_idx[i + k + 1] - oversigma_idx[i + k]) == 1):
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
            while (ss_val + k + 1 < len(overonesigma_idx)) and ((overonesigma_idx[ss_val + k + 1] - overonesigma_idx[ss_val + k]) == 1):
                k += 1
            while (ss_val + j - 1 >= 0) and ((overonesigma_idx[ss_val + j] - overonesigma_idx[ss_val + j - 1]) == 1):
                j -= 1

            if (n + j) <= 30 or (n + k) >= (len(bjd) - 30):
                continue

            a = diff_bjd[(n + j - 10) : (n + k + 10)]
            if len(a) > 0 and np.max(a) >= (2 / (24 * 60)) * 20:
                continue

            starttime.append(bjd[n + j])
            endtime.append(bjd[n + k])
            subbjd = bjd[(n + j) : (n + k + 1)]
            peak_idx = np.where(flux_detrend[(n + j) : (n + k + 1)] == max(flux_detrend[(n + j) : (n + k + 1)]))[0]
            if len(peak_idx) == 0:
                continue
            peaktime.append(subbjd[peak_idx[0]])
            count.append(np.sum(flux_detrend[(n + j) : (n + k + 1)]))

        self.detecttime, self.starttime, self.endtime, self.peaktime = map(np.array, [detecttime, starttime, endtime, peaktime])
        self.energy = self.tess_band_energy(np.array(count))

    def flaredetect_check(self):
        N = min(
            len(self.detecttime),
            len(self.starttime),
            len(self.endtime),
            len(self.peaktime),
            len(self.energy) if self.energy is not None else 0,
        )
        detecttime_new, starttime_new, endtime_new, peaktime_new, count_new, edecay_new, a_array, b_array = [], [], [], [], [], [], [], []

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
            a_val = (val_post - val_pre) / (t_post - t_pre) if (t_post - t_pre) != 0 else 0
            b_val = val_pre - a_val * t_pre
            flux_diff = flux - (a_val * bjd + b_val)

            peak_idx = np.where(bjd == ptime[i])[0]
            if len(peak_idx) == 0:
                continue
            n_peak = peak_idx[0]

            k, j = 0, 0
            while (n_peak + k < len(flux_diff)) and (flux_diff[n_peak + k] >= err[n_peak + k]):
                k += 1
            while (n_peak + j >= 0) and (flux_diff[n_peak + j] >= err[n_peak + j]):
                j -= 1

            n_end, n_start = n_peak + k - 1, n_peak + j + 1
            if n_start < 0 or n_end >= len(flux_diff):
                continue

            ss_flare = np.where((bjd >= bjd[n_start]) & (bjd <= bjd[n_end]))[0]
            if len(ss_flare) <= 1 or len(np.where((flux_diff[ss_flare] - 3 * err[ss_flare]) >= 0)[0]) <= 1:
                continue

            peak_flux = flux_diff[ss_flare].max()
            peak_loc = ss_flare[np.where(flux_diff[ss_flare] == peak_flux)[0][0]]

            ll = 0
            while ((peak_loc + ll) < len(flux_diff)) and (flux_diff[peak_loc + ll] >= peak_flux * np.exp(-1)):
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

        self.detecttime, self.starttime, self.endtime, self.peaktime, self.a_i, self.b_i, self.edecay = map(
            np.array, [detecttime_new, starttime_new, endtime_new, peaktime_new, a_array, b_array, edecay_new]
        )
        self.energy = self.tess_band_energy(np.array(count_new)) if len(count_new) > 0 else np.array([])
        self.duration = self.endtime - self.starttime + (2 / (24 * 60)) if len(starttime_new) > 0 else np.array([])

    def planck(self, wav, T):
        h, c, k = 6.626e-34, 3.0e8, 1.38e-23
        return (2.0 * h * c**2) / ((wav**5) * (np.exp(h * c / (wav * k * T)) - 1.0))

    def tess_band_energy(self, count):
        response = self._get_tess_response()
        if response is None:
            return np.array([])

        wave, resp, dw = response
        dt = 120.0
        Rstar = 695510e5 * self.R_sunstar_ratio
        sigma = 5.67e-5

        star_intensity_ratio = self._get_star_intensity_ratio(wave, resp, dw)  # [perf] cached ratio keeps math identical but avoids recompute
        if star_intensity_ratio == 0:
            return np.array([])

        area_factor = (np.pi * Rstar**2) * star_intensity_ratio
        return sigma * (10000**4) * area_factor * dt * count

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
        bjd = self.tessBJD
        diff_bjd = np.diff(bjd)
        gap_indices = np.where(diff_bjd >= 0.2)[0]
        gap_time = sum(bjd[idx + 1] - bjd[idx] for idx in gap_indices)
        self.precise_obs_time = bjd[-1] - bjd[0] - gap_time

    def flare_energy(self, energy_threshold_low, energy_threshold_high):
        if self.energy is None or len(self.energy) == 0:
            self.flare_number, self.sum_flare_energy = 0, 0.0
            return

        energy_cor = np.sort(self.energy)
        cumenergy = np.arange(len(energy_cor), 0, -1)
        energy_mask = (energy_cor >= energy_threshold_low) & (energy_cor <= energy_threshold_high)

        if np.any(energy_mask):
            self.flare_number = np.sum(energy_mask)
            self.sum_flare_energy = np.sum(energy_cor[energy_mask])
        else:
            self.flare_number, self.sum_flare_energy = 0, 0.0

    def flux_diff(self, min_percent: float = 0.02, max_percent: float = 0.98):
        sorted_flux = sorted(self.mPDCSAPflux)
        lower_bound = int(len(sorted_flux) * min_percent)
        upper_bound = int(len(sorted_flux) * max_percent)
        self.brightness_variation_amplitude = sorted_flux[upper_bound] - sorted_flux[lower_bound]
        self.starspot = (
            2
            * np.pi
            * (self.R_sunstar_ratio * 695510e3) ** 2
            * (self.T_star**4 / (self.T_star**4 - (self.T_star - self.d_T_star) ** 4))
            * self.brightness_variation_amplitude
        )
        self.starspot_ratio = (self.T_star**4 / (self.T_star**4 - (self.T_star - self.d_T_star) ** 4)) * self.brightness_variation_amplitude

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

    def rotation_period(self):  # [perf] regular frequency grid enables LombScargle fast solver
        frequency = ROTATION_FREQUENCY_GRID
        periods = _ROTATION_PERIODS
        lomb = LombScargle(self.tessBJD - self.tessBJD[0], self.mPDCSAPflux)
        power = lomb.power(frequency, method="fast")
        idx_max = int(np.argmax(power))
        self.per = periods[idx_max]
        half_max_power = np.max(power) / 2
        aa = np.where(power > half_max_power)[0]
        self.per_err = abs(periods[aa[-1]] - periods[aa[0]]) / 2

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
        low_threshold = ene_thres_low if ene_thres_low is not None else self.ene_thres_low
        high_threshold = ene_thres_high if ene_thres_high is not None else self.ene_thres_high

        if not skip_remove:
            self.remove()
        self.apply_gap_correction()
        self.detrend_flux()
        self.reestimate_errors()
        self.flaredetect()
        self.flaredetect_check()
        self.calculate_precise_obs_time()
        self.flare_energy(energy_threshold_low=low_threshold, energy_threshold_high=high_threshold)
        self.flux_diff()
        self.rotation_period()

        if self.tessBJD is not None and len(self.tessBJD) > 1 and self.precise_obs_time > 0:
            flare_ratio = self.flare_number / self.precise_obs_time
            BaseFlareDetector.array_flare_ratio = np.append(BaseFlareDetector.array_flare_ratio, flare_ratio)
            BaseFlareDetector.average_flare_ratio = np.mean(BaseFlareDetector.array_flare_ratio)
            sum_flare_energy_ratio = self.sum_flare_energy / self.precise_obs_time
            BaseFlareDetector.array_energy_ratio = np.append(BaseFlareDetector.array_energy_ratio, sum_flare_energy_ratio)
            BaseFlareDetector.array_observation_time = np.append(BaseFlareDetector.array_observation_time, np.median(self.tessBJD))
            BaseFlareDetector.array_amplitude = np.append(BaseFlareDetector.array_amplitude, self.brightness_variation_amplitude)
            BaseFlareDetector.array_starspot = np.append(BaseFlareDetector.array_starspot, self.starspot)
            BaseFlareDetector.array_starspot_ratio = np.append(BaseFlareDetector.array_starspot_ratio, self.starspot_ratio)
            BaseFlareDetector.array_data_name = np.append(BaseFlareDetector.array_data_name, self.data_name)
            BaseFlareDetector.array_per = np.append(BaseFlareDetector.array_per, self.per)
            BaseFlareDetector.array_per_err = np.append(BaseFlareDetector.array_per_err, self.per_err)

    def plot_flare(self):
        if self.tessBJD is None:
            return
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
        fig.add_trace(
            go.Scatter(x=self.tessBJD, y=self.mPDCSAPflux, mode="lines", line=dict(color="black", width=1), name="Normalized Flux"),
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="Normalized Flux", row=1, col=1)
        if self.s2mPDCSAPflux is not None:
            fig.add_trace(
                go.Scatter(x=self.tessBJD, y=self.s2mPDCSAPflux, mode="lines", line=dict(color="black", width=1), name="Detrended Flux"),
                row=2,
                col=1,
            )
            if self.peaktime is not None:
                for peak in self.peaktime:
                    fig.add_trace(
                        go.Scatter(
                            x=[peak, peak], y=[0.023, 0.0243], mode="lines", line=dict(color="red", width=1, dash="dash"), showlegend=False
                        ),
                        row=2,
                        col=1,
                    )
            fig.update_xaxes(title_text=f"Time (BJD - {self.time_offset})", row=2, col=1)
            fig.update_yaxes(title_text="Detrended Flux", row=2, col=1)
        fig.update_layout(title_text=f"Flare Detection Graph ({self.data_name})", height=900)
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
        axs[0].plot(self.tessBJD, self.mPDCSAPflux, color="black", linestyle="-", label="Normalized Flux", lw=0.5)
        axs[0].set_ylabel("Normalized Flux", fontsize=17)
        axs[0].tick_params(labelsize=13)

        # 2つ目のサブプロット: デトレンド後
        if self.s2mPDCSAPflux is not None:
            axs[1].plot(self.tessBJD, self.s2mPDCSAPflux, color="black", linestyle="-", label="Detrended Flux", lw=0.5)

            # フレアのピーク位置を線で示す
            if self.peaktime is not None:
                for peak in self.peaktime:
                    axs[1].axvline(x=peak, ymin=0.8, ymax=0.85, color="red", linestyle="-", linewidth=1.5)

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
        ax.plot(energy_cor, cumenergy / self.precise_obs_time, drawstyle="steps-post", color="black", lw=1.5)
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

    def show_variables(self):
        # ... (Implementation is the same as before)
        pass
