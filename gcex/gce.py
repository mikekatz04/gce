import numpy as np
from scipy import constants as ct
from tqdm import tqdm
import time

from GCE import run_gce_wrap

try:
    import cupy as xp

except ImportError:
    print("Need to install cupy.")
    exit()

import time

MTSUN = 1.989e30 * ct.G / ct.c ** 3


class ConditionalEntropy:
    def __init__(self, **kwargs):
        """
        """
        prop_defaults = {
            "use_double": False,  # not implemented yet
            "phase_bins": 15,
            "mag_bins": 10,  # overlap 50 %
        }

        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        if self.use_double:
            raise NotImplementedError

        else:
            self.dtype = xp.float32

        self.phase_bin_edges = xp.zeros((2 * self.phase_bins,), dtype=self.dtype)

        bin_start = 0.0
        bin_end = 2.0 / (self.phase_bins + 1)
        dbin = 1.0 / (self.phase_bins + 1)
        for i in range(self.phase_bins):
            self.phase_bin_edges[2 * i] = bin_start + dbin * i
            self.phase_bin_edges[2 * i + 1] = bin_end + dbin * i

        self.phase_bin_edges[-1] = 1.000001
        self.phase_bin_edges[0] = -0.000001

        self.half_dbins = dbin

        print(
            "\nGCE initialized with {} phase bins and {} mag bins.".format(
                self.phase_bins, self.mag_bins
            ),
            "\n",
        )

    def batched_run_const_nfreq(
        self,
        lightcurves,
        batch_size,
        pdot_batch_size,
        freqs,
        pdots,
        return_best_only=False,
        show_progress=True,
    ):
        """
        lightcurves should be list of light curves (number of lcs, number of points for lc, 2) 2-> time, mag
        """

        # setup mag bins
        bin_start = 0.0
        bin_end = 2.0 / (self.mag_bins + 1)
        dbin = 1.0 / (self.mag_bins + 1)
        mag_bin_template = np.zeros(self.mag_bins * 2)
        for i in range(self.mag_bins):
            mag_bin_template[2 * i] = bin_start + i * dbin
            mag_bin_template[2 * i + 1] = bin_end + i * dbin

        split_inds = []
        i = 0
        while i < len(lightcurves):
            i += batch_size
            if i >= len(lightcurves):
                break
            split_inds.append(i)

        lightcurves_split_all = np.split(lightcurves, split_inds)

        iterator = enumerate(lightcurves_split_all)
        iterator = tqdm(iterator) if show_progress else iterator

        ce_vals_out = []
        for i, light_curve_split in iterator:
            # st = time.perf_counter()
            max_length = 0
            number_of_pts = np.zeros((len(light_curve_split),)).astype(int)
            light_curve_mag_max = np.zeros((len(light_curve_split),))
            light_curve_mag_min = np.zeros((len(light_curve_split),))
            for j, lc in enumerate(light_curve_split):
                number_of_pts[j] = len(lc)
                max_length = max_length if len(lc) < max_length else len(lc)
                light_curve_mag_max[j] = np.max(lc[:, 1])
                light_curve_mag_min[j] = np.min(lc[:, 1])
            light_curve_arr = np.zeros((len(light_curve_split), max_length, 3))

            for j, lc in enumerate(light_curve_split):
                light_curve_arr[j, : len(lc)] = np.asarray(lc)

            light_curve_times = light_curve_arr[:, :, 0]
            min_light_curve_times = light_curve_times[:, 0]

            light_curve_mags = light_curve_arr[:, :, 1]
            light_curve_mags_inds = (
                np.ones(light_curve_mags.shape + (2,)).astype(int) * -1
            )

            light_curve_mag_bin_edges = []
            for min_val, max_val in zip(light_curve_mag_min, light_curve_mag_max):
                if min_val < 0.0:
                    min_val = min_val * 1.001
                else:
                    min_val = min_val * 0.999

                if max_val < 0.0:
                    max_val = max_val * 0.999
                else:
                    max_val = max_val * 1.001

                light_curve_mag_bin_edges.append(
                    min_val + (max_val - min_val) * mag_bin_template
                )

            light_curve_mag_bin_edges = np.asarray(light_curve_mag_bin_edges).reshape(
                -1, self.mag_bins, 2
            )

            # figure out index of mag bin (can put on gpu for speed up)
            for lc_i, lc_mag in enumerate(light_curve_mags):
                for kk in range(number_of_pts[lc_i]):
                    mag = lc_mag[kk]
                    num_bin = 0
                    k = 0
                    while num_bin < 2 and k < self.mag_bins:
                        bin_min, bin_max = light_curve_mag_bin_edges[lc_i, k]
                        if mag >= bin_min and mag < bin_max:
                            light_curve_mags_inds[lc_i, kk, num_bin] = k
                            num_bin += 1
                        k += 1

            light_curve_times = np.repeat(light_curve_times, 2, axis=1)
            light_curve_mags_inds = light_curve_mags_inds.reshape(
                light_curve_mags_inds.shape[0], -1
            )

            sort = np.argsort(light_curve_mags_inds, axis=1)[:, ::-1]
            for i, sort_i in enumerate(sort):
                light_curve_mags_inds[i] = light_curve_mags_inds[i][sort_i]
                light_curve_times[i] = light_curve_times[i][sort_i]

                num_pts_temp = np.where(light_curve_mags_inds[i] != -1)[0][-1]
                number_of_pts[i] = num_pts_temp

            # flatten everything
            light_curve_times_in = (
                xp.asarray(light_curve_times).flatten().astype(self.dtype)
            )
            light_curve_mags_in = xp.asarray(light_curve_mags.flatten()).astype(
                self.dtype
            )
            light_curve_mag_bin_edges_in = xp.asarray(
                light_curve_mag_bin_edges.flatten()
            ).astype(self.dtype)

            light_curve_mags_inds_in = xp.asarray(
                light_curve_mags_inds.flatten()
            ).astype(self.dtype)

            number_of_pts_in = xp.asarray(number_of_pts).astype(xp.int32)

            freqs_in = xp.asarray(freqs).astype(self.dtype)
            pdots_in = xp.asarray(pdots).astype(self.dtype)
            min_light_curve_times_in = xp.asarray(min_light_curve_times).astype(
                self.dtype
            )

            ce_vals_out_temp = xp.zeros(
                (len(freqs_in) * len(pdots_in) * len(light_curve_times),),
                dtype=self.dtype,
            )

            run_gce_wrap(
                ce_vals_out_temp,
                freqs_in,
                len(freqs_in),
                pdots_in,
                len(pdots_in),
                self.phase_bin_edges,
                light_curve_mags_inds_in,
                light_curve_times_in,
                number_of_pts_in,
                number_of_pts_in.max().item(),
                self.mag_bins,
                self.phase_bins,
                len(light_curve_times),
                min_light_curve_times_in,
                self.half_dbins,
            )
            ce_vals_out.append(ce_vals_out_temp.get())

        return np.concatenate(ce_vals_out, axis=0)
