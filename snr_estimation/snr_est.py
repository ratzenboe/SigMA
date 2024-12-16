import numpy as np
from snr_estimation.hist_3d import Hist3D


class SNR(Hist3D):
    def __init__(self, data_sig, data_bg, limits, bin_size):
        super().__init__(limits, bin_size)
        self.data_sig = data_sig
        self.data_bg = data_bg
        self.snr = None
        self.snr_std = None

    def update_data(self, data_sig, data_bg):
        self.data_sig = data_sig
        self.data_bg = data_bg
        return

    def snr(self, radius=10, ref_pt=None):
        """Compute SNR of a signal in a 3D (velocity) histogram."""
        if ref_pt is None:
            ref_pt = np.median(self.data_sig, axis=0)
        # Get the number of sources in the neighborhood (within some km/s)
        n_sources_bg_all = []
        if isinstance(radius, (int, float)):
            radius = [radius]
        # Compute the number of sources in the background for different radii
        for r in radius:
            n_sources_bg = self.voxel_counts(r, data=self.data_bg, ref_pt=ref_pt)
            n_sources_bg_all.append(np.mean(n_sources_bg))
        # Compute the number of sources in the signal in the "central" voxel
        n_sources_sig = self.voxel_counts(self.bin_size/2, data=self.data_sig, ref_pt=ref_pt)
        # Compute SNR
        self.snr = n_sources_sig / np.median(n_sources_bg_all)
        if len(radius) > 1:
            self.snr_std = np.std(n_sources_sig / n_sources_bg_all)
        return self.snr, self.snr_std
