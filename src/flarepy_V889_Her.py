import numpy as np
from scipy.interpolate import interp1d

from .base_flare_detector import BaseFlareDetector


class FlareDetector_V889_Her(BaseFlareDetector):
    """
    Flare detector for V889 Her, with a custom detrending method.
    """

    def __init__(
        self,
        file=None,
        process_data=False,
        ene_thres_low=5e33,
        ene_thres_high=2e40,
    ):
        super().__init__(
            file=file,
            R_sunstar_ratio=1.0,
            T_star=6550,
            flux_mean=300710.62334465684,
            err_constant_mean=np.mean(
                [
                    0.000327511843591592,
                    0.0002949516133656024,
                    0.0005806540022277836,
                    0.0003847150867966205,
                ]
            ),
            rot_period=0.4398277479138892,
            rotation_period_min=0.3,
            rotation_period_max=2.0,
            rotation_n_points=10000,
            f_cut_lowpass=30,
            f_cut_spline=40,
            sector_threshold=90,  # Use SAP_FLUX for sectors > 90
            process_data=False,  # Handle processing manually after setting attributes
            ene_thres_low=ene_thres_low,
            ene_thres_high=ene_thres_high,
            use_sector_mean=True,
        )
        # V889 Her固有のギャップ検出閾値を設定
        self.gap_threshold = 0.004

        if process_data:
            self.process_data()

    def difference_at_lag(self, arr, n=1):
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        if n <= 0 or n >= len(arr):
            raise ValueError(f"n (lag) {n} is invalid for array of length {len(arr)}.")
        return arr[n:] - arr[:-n]

    def detrend_flux(self):
        # ローカル変数を短く
        time_ext = self.gtessBJD
        flux_ext = self.gmPDCSAPflux
        flux_err_ext = self.gmPDCSAPfluxerr
        buf_size = self.buffer_size

        # flare_canにstart～endにあるindex全てを入れる
        diff_time = np.diff(time_ext)
        diff_flux = np.diff(flux_ext)

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

        flare_can_start_candidates = np.where(
            (
                (diff_flux > 0.01)
                | (flux_diff_lag2_appended > 0.01)
                | (flux_diff_lag3_appended > 0.01)
                | (flux_diff_lag4_appended > 0.01)
                | (flux_diff_lag5_appended > 0.01)
            )
            & (diff_time < 0.005)
        )[0]

        before_low_flare_list = []
        valid_flare_starts = []

        for j_start_candidate in flare_can_start_candidates:
            for i_end_candidate in range(j_start_candidate + 5, len(flux_ext)):
                if abs(flux_ext[j_start_candidate] - flux_ext[i_end_candidate]) < 0.008:
                    before_low_flare_list.append(i_end_candidate)
                    valid_flare_starts.append(j_start_candidate)
                    break

        flare_starts = np.array(valid_flare_starts, dtype=int)
        flare_ends = np.array(before_low_flare_list, dtype=int)

        flux_to_interpolate = np.copy(flux_ext)
        mask = np.zeros(len(flux_ext), dtype=bool)
        for start_idx, end_idx in zip(flare_starts, flare_ends, strict=False):
            if start_idx < end_idx:
                mask[start_idx : end_idx + 1] = True

        time_valid = time_ext[~mask]
        flux_valid = flux_ext[~mask]

        spline_func = interp1d(time_valid, flux_valid, kind="cubic")
        time_flare_intervals = time_ext[mask]
        if len(time_flare_intervals) > 0:
            flux_to_interpolate[mask] = spline_func(time_flare_intervals)

        self.flux_splined = flux_to_interpolate
        self.filtered_flux = self.lowpass(
            time_ext, self.flux_splined, fc=self.f_cut_lowpass
        )
        self.s1_flux = flux_ext - self.filtered_flux

        fac = 3
        ss_flarecan = np.where(
            (self.s1_flux <= flux_err_ext * fac)
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
