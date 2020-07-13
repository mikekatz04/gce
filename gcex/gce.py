#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Michael Katz (2020)
#
# This file is part of gce
#
# gce is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gce is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gce.  If not, see <http://www.gnu.org/licenses/>

import numpy as np
from scipy import constants as ct
from tqdm import tqdm
import time
import pickle

try:
    from GCE import run_gce_wrap, run_long_lc_gce_wrap
    run_gpu = True

except ImportError:
    print("No gce. Running python ce version.")
    run_gpu = False

if run_gpu:
    try:
        import cupy as xp

    except ImportError:
        import numpy as xp

        print("Need to install cupy.")
        exit()

else:
    import numpy as xp

import time

MTSUN = 1.989e30 * ct.G / ct.c ** 3

# module for python usage of the ce calculation
# mainly for prototyping
def py_check_ce(
    ce_vals,
    freqs_in,
    num_freqs,
    pdots_in,
    num_pdots,
    light_curve_mags_inds,
    time_vals_in,
    number_of_pts_in,
    max_points,
    mag_bins,
    phase_bins,
    num_lcs,
    half_dbins,
):

    for lc_i in range(num_lcs):
        mags = light_curve_mags_inds[lc_i * max_points : (lc_i + 1) * max_points]
        time_vals = time_vals_in[lc_i * max_points : (lc_i + 1) * max_points]
        for pdot_i, pdot in enumerate(pdots_in):
            for i, frequency in enumerate(freqs_in):
                p_ij = np.zeros((phase_bins, mag_bins))
                period = 1.0 / frequency
                folded = (
                    np.fmod(
                        time_vals - 0.5 * pdot * frequency * (time_vals * time_vals),
                        period,
                    )
                    * frequency
                )

                phase_bin_inds = (folded // half_dbins).astype(int) % phase_bins

                for pbi, mbi in zip(phase_bin_inds, mags):
                    p_ij[pbi][mbi] += 1.0

                # 1 overlap
                phase_bin_inds = ((folded // half_dbins).astype(int) - 1) % phase_bins

                for pbi, mbi in zip(phase_bin_inds, mags):
                    p_ij[pbi][mbi] += 1.0

                total_points = np.sum(p_ij)
                p_j = np.tile(np.sum(p_ij, axis=1), (10, 1)).T

                inds = np.where(p_ij != 0.0)
                Hc = (
                    1
                    / total_points
                    * np.sum(p_ij[inds] * np.log(p_j[inds] / p_ij[inds]))
                )
                ce_vals[(lc_i * num_pdots + pdot_i) * num_freqs + i] = Hc

    return ce_vals


class ConditionalEntropy:
    """Calculate the Conditional Entropy

    This class computes the Conditional Entropy for a list of light curves
    along pdot and frequency axes determined by the user. It will use GPU
    resources if available. Otherwise, it will run a slow Python version of
    the Conditional Entropy. This Python version allows for prototyping code
    and testing code before using the GPUs.

    Args:
        phase_bins (int, optional): Number of phase bins. Default is 30.
        mag_bins (int, optional): Number of magnitude bins. Default is 10.
        long_limit (int, optional): Length of a light curve where the code
            forces the use of the longer light curve version of gce.
            Default is 2,000.
        use_long (bool, optional): If True, use the long gce version of the
            code. Default is False.

    Attributes:
        phase_bins (int): Number of phase bins.
        mag_bins (int): Number of magnitude bins.
        half_dbins (double): Half of the phase bin width.
        ce (array): Numpy ndarray of Conditional Entropy values. Shape is
            (light curves, pdots, frequencyies).
        min_ce (double): Minimum Conditional Entropy value.
        mean_ce (double): Mean of the Conditional Entropy values.
        std_ce (double): Standard deviation of the Conditional Entropy values.
        significance (double): Returns the significance of the observations.
            The formula we use for the significance is (mean - min)/std.
        best_params (tuple of lists): Best parameters determined by the minimum
            Conditional Entropy. The tuple will contain lists of the best
            parameters, which may be more than one configuration.

    """

    def __init__(
        self, phase_bins=30, mag_bins=10, long_limit=2000, use_long=False, **kwargs
    ):

        prop_defaults = {"use_double": False}  # not implemented yet

        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        self.phase_bins = phase_bins
        self.mag_bins = mag_bins
        self.long_limit, self.use_long = long_limit, use_long

        if self.use_double:
            raise NotImplementedError

        else:
            self.dtype = xp.float32

        # get bin widths
        dbin = 1.0 / (self.phase_bins + 1)
        self.half_dbins = dbin

        self.half_d_mag_bins = 1.0 / (self.mag_bins + 1)

        print(
            "\nGCE initialized with {} phase bins and {} mag bins.".format(
                self.phase_bins, self.mag_bins
            ),
            "\n",
        )

        # set output function based on what is requested from user
        self.corresponding_func = dict(
            all="ce", best="min_ce", best_params="best_params"
        )

    def _get_best_params(self, ce, run_gpu=True):
        best_freqs = []
        best_pdots = []

        for sub in ce:

            # find the indices where ce is minimum
            if run_gpu:
                inds = np.where(sub.get() == sub.get().min())
            else:
                inds = np.where(sub == np.min(sub))

            best_freqs.append(self.test_freqs[inds[1]])
            best_pdots.append(self.test_pdots[inds[0]])

        return best_freqs, best_pdots

    @property
    def best_params(self):
        if self.convert_f_to_p:
            best_periods = [1.0 / bf for bf in self.best_freqs]
            return best_periods, self.best_pdots
        else:
            return self.best_freqs, self.best_pdots

    @property
    def significance(self):
        # calculate significance of observation
        return (self.mean_ce - self.min_ce) / self.std_ce

    def _single_light_curve_batch(
        self, light_curve_split, pdot_batch_size, freqs, pdots, show_progress=False
    ):

        # determine maximum length of light curves in batch
        # also find minimum and maximum magnitudes for each light curve
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

        # populate light_curve_arr
        for j, lc in enumerate(light_curve_split):
            light_curve_arr[j, : len(lc)] = np.asarray(lc)

        # separate time and mag info
        light_curve_times = light_curve_arr[:, :, 0]

        light_curve_mags = light_curve_arr[:, :, 1]
        light_curve_mags_inds = np.ones(light_curve_mags.shape + (2,)).astype(int) * -1

        # setup light curve bin magnitudes
        # adjust maximum and minimum to include all values of interest
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

            light_curve_mag_bin_edges.append([min_val, max_val])

        light_curve_mag_bin_edges = np.asarray(light_curve_mag_bin_edges)

        # figure out index of mag bin
        # If first mag_bin_ind is self.mag_bins or 0, then this point will
        # not count twice
        # find mag values that occur in overlap and record twice
        # also makes sure mag_ind of self.mag_bins is in the
        # index (self.mag_bins-1) bin
        for lc_i, (lc_mag, min_max) in enumerate(
            zip(light_curve_mags, light_curve_mag_bin_edges)
        ):
            min = min_max[0]
            diff = min_max[1] - min_max[0]

            mag_ind = np.floor((lc_mag - min) / diff / self.half_d_mag_bins).astype(int)
            mag_ind_2 = -1 * ((mag_ind == 0) | (mag_ind == self.mag_bins)) + (
                mag_ind - 1
            ) * ((mag_ind > 0) & (mag_ind < self.mag_bins))

            mag_ind = mag_ind * (mag_ind != self.mag_bins) + (mag_ind - 1) * (
                mag_ind == self.mag_bins
            )

            light_curve_mags_inds[lc_i, :, 0] = mag_ind
            light_curve_mags_inds[lc_i, :, 1] = mag_ind_2

        light_curve_mags_inds = light_curve_mags_inds.reshape(
            light_curve_mags_inds.shape[0], -1
        )

        # duplicate times for multiple magnitude bins
        light_curve_times = np.repeat(light_curve_times, 2, axis=1)

        # sort so that max mag_ind is first and
        # -1 values for no second mag bin are last
        sort = np.argsort(light_curve_mags_inds, axis=1)[:, ::-1]
        for i, sort_i in enumerate(sort):
            light_curve_mags_inds[i] = light_curve_mags_inds[i][sort_i]
            light_curve_times[i] = light_curve_times[i][sort_i]
            number_of_pts[i] = np.where(light_curve_mags_inds[i] != -1)[0][-1] + 1

        # find new max number of points
        max_num_pts_in = number_of_pts.max()

        # remove and values that go beyond max number of points
        light_curve_times = light_curve_times[:, :max_num_pts_in]
        light_curve_mags_inds = light_curve_mags_inds[:, :max_num_pts_in]

        # split into pdot batches
        split_inds = []
        i = 0
        while i < len(pdots):
            i += pdot_batch_size
            if i >= len(pdots):
                break
            split_inds.append(i)

        pdots_split_all = np.split(pdots, split_inds)

        iterator = enumerate(pdots_split_all)
        iterator = tqdm(iterator, desc="pdot batch") if show_progress else iterator

        # maintain statistics of each batch
        mins = []
        means = []
        ns = []
        stds = []
        bf = []
        bp = []
        ce_vals = []
        for i, pdots_split in iterator:
            out = self._single_pdot_batch(
                light_curve_times,
                light_curve_mags_inds,
                number_of_pts,
                freqs,
                pdots_split,
                show_progress=show_progress,
            )

            mins.append(out.get("min"))
            means.append(out.get("mean"))
            stds.append(out.get("std"))
            ns.append(out.get("n"))
            bf.append(out.get("best_freqs"))
            bp.append(out.get("best_pdots"))
            ce_vals.append(out.get("ce_vals"))

        mins = np.vstack(mins).T
        means = np.vstack(means).T
        ns = np.vstack(ns).T
        stds = np.vstack(stds).T

        ce_vals = np.concatenate(ce_vals, axis=1)

        min_ce = np.min(mins, axis=1)
        mean_ce = np.sum(means * ns, axis=1) / self.n
        std_ce = np.sqrt(np.sum(stds ** 2 * ns, axis=1) / self.n)

        # find best ce values and account for batch statistics
        ind_mins = np.argmin(mins, axis=1)

        best_freqs = []
        best_pdots = []

        for lc_i in range(len(light_curve_times)):
            ind_min = ind_mins[lc_i]
            best_freqs.append(bf[ind_min][lc_i])
            best_pdots.append(bp[ind_min][lc_i])

        temp = dict(
            mean=mean_ce,
            std=std_ce,
            min=min_ce,
            best_freqs=best_freqs,
            best_pdots=best_pdots,
            ce_vals=ce_vals,
        )

        return temp

    def _single_pdot_batch(
        self,
        light_curve_times,
        light_curve_mags_inds,
        number_of_pts,
        freqs,
        pdots,
        return_type="all",
        show_progress=False,
    ):
        # flatten everything
        light_curve_times_in = (
            xp.asarray(light_curve_times).flatten().astype(self.dtype)
        )

        light_curve_mags_inds_in = xp.asarray(light_curve_mags_inds.flatten()).astype(
            xp.int32
        )

        number_of_pts_in = xp.asarray(number_of_pts).astype(xp.int32)

        freqs_in = xp.asarray(freqs).astype(self.dtype)
        pdots_in = xp.asarray(pdots).astype(self.dtype)

        ce_vals_out_temp = xp.zeros(
            (len(freqs_in) * len(pdots_in) * len(light_curve_times),), dtype=self.dtype
        )

        max_num_pts_in = number_of_pts_in.max().item()

        # determine which function to use in gce
        if run_gpu:
            if self.use_long or max_num_pts_in > self.long_limit:
                # TODO: adjust magnitude/time inputs specific to long
                self.gce_func = run_long_lc_gce_wrap

            else:
                self.gce_func = run_gce_wrap

        else:
            self.gce_func = py_check_ce

        # run a single pdot and single light curve batch
        self.gce_func(
            ce_vals_out_temp,
            freqs_in,
            len(freqs_in),
            pdots_in,
            len(pdots_in),
            light_curve_mags_inds_in,
            light_curve_times_in,
            number_of_pts_in,
            max_num_pts_in,
            self.mag_bins,
            self.phase_bins,
            len(light_curve_times),
            self.half_dbins,
        )

        ce_vals_out_temp = ce_vals_out_temp.reshape(
            len(light_curve_times), len(pdots_in), len(freqs_in)
        )
 
        bf, bp = self._get_best_params(ce_vals_out_temp,run_gpu=run_gpu)

        temp = dict(
            mean=xp.mean(
                ce_vals_out_temp.reshape(ce_vals_out_temp.shape[0], -1), axis=1
            ).get(),
            n=ce_vals_out_temp.shape[1] * ce_vals_out_temp.shape[2],
            std=xp.std(
                ce_vals_out_temp.reshape(ce_vals_out_temp.shape[0], -1), axis=1
            ).get(),
            min=xp.min(
                ce_vals_out_temp.reshape(ce_vals_out_temp.shape[0], -1), axis=1
            ).get(),
            ce_vals=ce_vals_out_temp.get(),
            best_freqs=bf,
            best_pdots=bp,
        )

        return temp

    def batched_run_const_nfreq(
        self,
        lightcurves,
        batch_size,
        freqs,
        pdots=None,
        pdot_batch_size=None,
        return_type="all",
        show_progress=False,
        pickle_out=[],
        pickle_string="ce_out",
        true_vals=None,
        convert_f_to_p=False,
        store_ce=False,
    ):
        """Run gce on lights curves.

        This method actually performs the gce calculation.

        args:
            lightcurves (list of numpy ndarrays): Light curves to be analyzed
                given as a list of numpy arrays containing the light curve
                information. The light curve arrays have shape
                (2, light curve length), where 2 is the time and magnitdue.

            batch_size (int): Max number of light curves to run through gce for each
                computation. Usually, the larger the number, the faster the
                algorithm will perform, but with diminishing returns. However,
                there is an upper limit due to memory on the GPU.

                Choice here affects the memory usage on the GPU and splits up
                the calculation as needed to stay within memory limits.

            freqs (1D array): Array with the frequencies to be tested.
            pdots (1D array, optional): Array with pdots to be tested. Default
                is `None` indicating a test with 0.0 for the pdot.

            pdot_batch_size (int, optional): Number of pdots to run for each computation.
                Choice here affects the memory usage on the GPU and splits up
                the calculation as needed to stay within memory limits. When
                len(pdots) > pdot_batch_size, the calculations are run
                separately and then combined, including statistics of the
                Conditional Entropy values. Defaults to `None`, which means gce
                will use the length of the pdot array as the `pdot_batch_size`.

            return_type (str, optional): User chosen return type.

                Options:
                    `all`: Returned Conditional Entropy values for every grid point.
                    `best`: Return best Conditional Entropy value.
                    `best_params`: Return best frequency and pdot values.

            show_progress (bool, optional): Shows progress of batch calculations
                using tqdm. Default is False.

        Returns:
            array or lists of arrays: ce, best, best_params
                Return all Conditional Entropy values, the best Conditional
                Entropy values, or the best parameters for each light curve.

        Raises:
            TypeError: `return_type` is set to a value that is not in the
                options list.
            ValueError: `pdot_batch_size` is less than one.
            ValueError: Key in `pickle_out` is not available.

        """
        self.convert_f_to_p = convert_f_to_p

        if return_type not in ["all", "best", "best_params"]:
            raise TypeError(
                "Variable `return_type` must be set to either 'all', 'best', or 'best_params'"
            )

        # defaults for pdots
        if pdots is None:
            self.test_pdots = xp.asarray([0.0])
            pdot_batch_size = 1

        # defaults for pdot_batch_size
        if pdot_batch_size is None:
            pdot_batch_size = 1

        if pdot_batch_size < 1:
            raise ValueError(
                "pdot_batch_size must be greater than or equal to one. Setting it to None defaults to 1."
            )

        # store for later computations
        self.test_pdots = pdots
        self.test_freqs = freqs
        self.n = len(pdots) * len(freqs)

        # split by light curve batches
        split_inds = []
        i = 0
        while i < len(lightcurves):
            i += batch_size
            if i >= len(lightcurves):
                break
            split_inds.append(i)

        lightcurves_split_all = np.split(lightcurves, split_inds)

        iterator = enumerate(lightcurves_split_all)
        iterator = tqdm(iterator, desc="lc batch") if show_progress else iterator

        # keep track of statistics
        mins = []
        means = []
        stds = []
        bf = []
        bp = []
        ce_vals = []
        for i, light_curve_split in iterator:

            # run one light curve batch
            out = self._single_light_curve_batch(
                light_curve_split,
                pdot_batch_size,
                freqs,
                pdots,
                show_progress=show_progress,
            )

            # go through statistics of each light curve in the batch
            for lc_i in range(len(light_curve_split)):
                mins.append(out.get("min")[lc_i])
                means.append(out.get("mean")[lc_i])
                stds.append(out.get("std")[lc_i])
                bf.append(out.get("best_freqs")[lc_i])
                bp.append(out.get("best_pdots")[lc_i])
                ce_vals.append(out.get("ce_vals")[lc_i])

        # set overall quantities of interest
        self.min_ce = np.asarray(mins)
        self.mean_ce = np.asarray(means)
        self.std_ce = np.asarray(stds)
        self.best_freqs = bf
        self.best_pdots = bp
        self.ce = np.asarray(ce_vals)

        if pickle_out != []:
            out_dict = {}
            for key in pickle_out:
                if key not in ["truth", "lcs"]:
                    try:
                        out_dict[key] = getattr(self, key)
                    except AttributeError:
                        raise ValueError(
                            "Keys for pickling must be attributes. See list in documentation."
                        )
                elif key == "truth":
                    out_dict[key] = true_vals

                elif key == "lcs":
                    out_dict["lcs"] = lightcurves

                else:
                    raise ValueError(
                        "Keys for pickling must be attributes. See list in documentation."
                    )

            with open(pickle_string + ".pickle", "wb") as fp:
                pickle.dump(out_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

        return getattr(self, self.corresponding_func.get(return_type))
