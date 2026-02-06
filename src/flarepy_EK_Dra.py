import numpy as np

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
            err_constant_mean=np.mean(
                [
                    0.000384864314943702,
                    0.0004424048336559645,
                    0.00040872798534543,
                    0.0004216541882493638,
                    0.0004073989969316659,
                    0.0003649385833756368,
                    0.0004276275012618665,
                    0.0003423306959085268,
                    0.00041515198565203514,
                    0.00037956035591004555,
                    0.000576264495105358,
                    0.00037297277185969937,
                ]
            ),
            rot_period=0.2094793179536128,
            rotation_period_min=1.5,
            rotation_period_max=5.0,
            rotation_n_points=10000,
            f_cut_lowpass=3,
            f_cut_spline=6,
            sector_threshold=74,  # Use SAP_FLUX for sectors > 74
            process_data=process_data,
            ene_thres_low=ene_thres_low,
            ene_thres_high=ene_thres_high,
            use_sector_mean=True,
        )
        # EK Dra-specific gap detection threshold.
        self.gap_threshold = 0.2
