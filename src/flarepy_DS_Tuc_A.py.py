
from .base_flare_detector import BaseFlareDetector

class FlareDetector_DS_Tuc_A(BaseFlareDetector):
    """
    Flare detector for DS Tuc A, inheriting from BaseFlareDetector.
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
            R_sunstar_ratio=0.87,
            T_star=5428,
            flux_mean=119633.99533564149,
            err_constant_mean=0.0005505874029881446, # Mean of the 5 values
            rot_period=0.3672257916463397,
            f_cut_lowpass=3,
            f_cut_spline=6,
            process_data=process_data,
            ene_thres_low=ene_thres_low,
            ene_thres_high=ene_thres_high,
        )

    def remove(self):
        """
        Removes transit data specific to DS Tuc A.
        """
        start_time_remove = [1332.243, 1340.381, 1348.514, 2040.268, 2056.539, 2064.682, 2080.954, 3130.791, 3147.072, 3155.202, 3163.342, 3171.481]
        end_time_remove = [1332.377, 1340.515, 1348.646, 2040.398, 2056.672, 2064.818, 2081.048, 3130.924, 3147.202, 3155.333, 3163.473, 3171.613]

        abjd = self.atessBJD.copy()
        aflux = self.amPDCSAPflux.copy()
        afluxerr = self.amPDCSAPfluxerr.copy()

        for start, end in zip(start_time_remove, end_time_remove):
            mask = (abjd < start) | (abjd > end)
            abjd = abjd[mask]
            aflux = aflux[mask]
            afluxerr = afluxerr[mask]

        self.tessBJD = abjd
        self.mPDCSAPflux = aflux
        self.mPDCSAPfluxerr = afluxerr
