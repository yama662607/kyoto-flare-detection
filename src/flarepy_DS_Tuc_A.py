import numpy as np

from .base_flare_detector import BaseFlareDetector


class FlareDetector_DS_Tuc_A(BaseFlareDetector):
    """
    Flare detector for DS Tuc A, inheriting from BaseFlareDetector.
    """

    def __init__(
        self,
        file=None,
        process_data=False,
        run_process_data_2=False,
        ene_thres_low=5e33,
        ene_thres_high=2e40,
    ):
        super().__init__(
            file=file,
            R_sunstar_ratio=0.964,
            T_star=5428,
            flux_mean=119633.99533564149,
            err_constant_mean=np.mean(
                [
                    0.0005519882005171186,
                    0.0005789426404572392,
                    0.0005975980849639832,
                    0.0005009879446873866,
                    0.0005225201442198351,
                ]
            ),
            rot_period=0.3672257916463397,
            rotation_period_min=1.0,
            rotation_period_max=8.0,
            rotation_n_points=10000,
            f_cut_lowpass=3,
            f_cut_spline=6,
            sector_threshold=74,  # Use SAP_FLUX for sectors > 74
            process_data=process_data,
            run_process_data_2=run_process_data_2,
            ene_thres_low=ene_thres_low,
            ene_thres_high=ene_thres_high,
            use_sector_mean=True,
        )
        # DS Tuc A-specific gap detection threshold.
        self.gap_threshold = 0.05

    def remove(self):
        """
        Removes transit data specific to DS Tuc A.
        """
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

        for start, end in zip(start_time_remove, end_time_remove, strict=False):
            mask = (abjd < start) | (abjd > end)
            abjd = abjd[mask]
            aflux = aflux[mask]
            afluxerr = afluxerr[mask]

        self.tessBJD = abjd
        self.mPDCSAPflux = aflux
        self.mPDCSAPfluxerr = afluxerr

    def tess_band_energy(self, count):
        """
        Estimate TESS-band energy including contributions from the DS Tuc A
        primary and its companion.
        """
        response = self._get_tess_response()
        if response is None:
            return np.array([])

        wave, resp, dw = response
        dt = 120.0
        Rsun_cm = 695510e5
        sigma_SB = 5.67e-5

        R_primary = Rsun_cm * self.R_sunstar_ratio  # 0.87 Rsun
        R_companion = Rsun_cm * 0.864  # Rsun

        # Integrate intensities in the TESS band for each component.
        planck_star = self.planck(wave * 1e-9, self.T_star)
        planck_companion = self.planck(wave * 1e-9, 4700)
        planck_flare = self.planck(wave * 1e-9, 10000)

        main_intensity = np.sum(dw * planck_star * resp)
        companion_intensity = np.sum(dw * planck_companion * resp)
        ref_intensity = np.sum(dw * planck_flare * resp)

        if ref_intensity == 0:
            return np.array([])

        # Total stellar intensity (area-weighted).
        total_star_intensity = (
            main_intensity * R_primary**2 + companion_intensity * R_companion**2
        )

        # A_flare = C_flare * pi * R_primary^2 * (L_total_star_TESS / L_primary_flare_TESS)
        # Assume flares occur on the primary star.
        area_factor = np.pi * (total_star_intensity / ref_intensity)

        return sigma_SB * (10000**4) * area_factor * dt * count

    def flux_diff(self, min_flux=0.02, max_flux=0.98):
        """Update starspot area using the primary + companion areas."""
        super().flux_diff(min_percent=min_flux, max_percent=max_flux)
        primary_area = (self.R_sunstar_ratio * 695510e3) ** 2
        companion_area = (0.864 * 695510e3) ** 2
        area_scale = (
            2
            * np.pi
            * (primary_area + companion_area)
            * (self.T_star**4 / (self.T_star**4 - (self.T_star - self.d_T_star) ** 4))
        )
        self.starspot = area_scale * self.brightness_variation_amplitude
