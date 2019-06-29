import numpy as np
from scipy import constants as ct
from tqdm import tqdm

from GCE import GCE

import time

MTSUN = 1.989e30*ct.G/ct.c**3


class ConditionalEntropy:
    def __init__(self, **kwargs):
        """
        """
        prop_defaults = {
            'use_double': False,  # not implemented yet
            'phase_bins': 15,
            'mag_bins': 10
        }

        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        self.gce = GCE(self.phase_bins, self.mag_bins)

    def batched_run_const_nfreq(self, lightcurves, batch_size, freqs, show_progress=True):
        """
        lightcurves should be list of light curves (number of lcs, number of points for lc, 2) 2-> time, mag
        """

        split_inds = []
        i = 0
        while(i < len(lightcurves)):
            i += batch_size
            if i >= len(lightcurves):
                break
            split_inds.append(i)

        lightcurves_split_all = np.split(lightcurves, split_inds)

        iterator = enumerate(lightcurves_split_all)
        iterator = tqdm(iterator) if show_progress else iterator

        ce_vals_out = []
        for i, light_curve_split in iterator:
            max_length = 0
            number_of_pts = np.zeros((len(light_curve_split),)).astype(int)
            light_curve_mag_max = np.zeros((len(light_curve_split),))
            light_curve_mag_min = np.zeros((len(light_curve_split),))
            for j, lc in enumerate(light_curve_split):
                number_of_pts[j] = len(lc)
                max_length = max_length if len(lc) < max_length else len(lc)
                light_curve_mag_max[j] = np.max(lc[:,1])
                light_curve_mag_min[j] = np.min(lc[:,1])
            light_curve_arr = np.zeros((len(light_curve_split), max_length, 2))

            for j, lc in enumerate(light_curve_split):
                light_curve_arr[j, :len(lc)] = np.asarray(lc)

            light_curve_times = light_curve_arr[:, :, 0]
            light_curve_mags = light_curve_arr[:, :, 1]

            light_curve_mag_bin_edges = np.asarray([np.linspace(min_val*0.999, max_val*1.001, self.mag_bins+1) for min_val, max_val in zip(light_curve_mag_min, light_curve_mag_max)])

            # flatten everything
            light_curve_times = light_curve_times.flatten()
            light_curve_mags = light_curve_mags.flatten()
            light_curve_mag_bin_edges = light_curve_mag_bin_edges.flatten()

            ce_vals_out.append( self.gce.conditional_entropy(light_curve_times.astype(np.float64),
                                                   light_curve_mags.astype(np.float64),
                                                   light_curve_mag_bin_edges.astype(np.float64),
                                                   number_of_pts.astype(np.int32),
                                                   freqs.astype(np.float64)))

        return np.concatenate(ce_vals_out, axis=0)
