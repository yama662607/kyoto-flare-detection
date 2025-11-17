from .base_flare_detector import BaseFlareDetector


class FlareDetector_EK_Dra(BaseFlareDetector):
    """
    Flare detector for EK Dra, inheriting from BaseFlareDetector.
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
            R_sunstar_ratio=0.94,
            T_star=5700,
            flux_mean=249320.35370300722,
            err_constant_mean=0.0004111604805261475,  # Mean of the 12 values
            rot_period=0.2094793179536128,
            f_cut_lowpass=3,
            f_cut_spline=6,
            sector_threshold=74,  # Use SAP_FLUX for sectors > 74
            process_data=process_data,
            ene_thres_low=ene_thres_low,
            ene_thres_high=ene_thres_high,
        )
        # EK Dra固有のギャップ検出閾値を設定
        self.gap_threshold = 0.2
