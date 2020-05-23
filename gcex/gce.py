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

try:
    from GCE import run_gce_wrap
except ImportError:
    pass

try:
    import cupy as xp

except ImportError:
    print("Need to install cupy.")

import time

MTSUN = 1.989e30 * ct.G / ct.c ** 3


class ConditionalEntropy:
    """Calculate the Conitional Entropy

    This class computes the conditional entropy for a list of light curves
    along pdot and frequency axes determined by the user.

    Args:
        phase_bins (int, optional): Number of phase bins. Default is 30.
        mag_bins (int, optional): Number of magnitude bins. Default is 10.

    Attributes:
        phase_bins (int): Number of phase bins.
        mag_bins (int): Number of magnitude bins.
        half_dbins (double): Half of the phase bin width.
        ce (array): Cupy array of conditional entropy values. Shape is
            (light curves, pdots, frequencyies).
        ce_cpu (array): Numpy array of conditional entropy values. Shape is
            (light curves, pdots, frequencyies).
        min_ce (double): Minimum conditional entropy value.
        mean_ce (double): Mean of the conditional entropy values.
        std_ce (double): Standard deviation of the conditional entropy values.
        significance (double): Returns the significance of the observations.
            The formula we use for the significance is (mean - min)/std.
        best_params (tuple of lists): Best parameters determined by the minimum
            conditional entropy. The tuple will contain lists of the best
            parameters, which may be more than one configuration. The tuple will
            be length 1 if only one pdot is tested (tuple=(**freqs**,)).
            It will be length 2 if multiple pdots are tested (tuple=(**freqs**, **pdots**)).



    """

    def __init__(self, phase_bins=30, mag_bins=10, **kwargs):

        prop_defaults = {"use_double": False}  # not implemented yet

        for prop, default in prop_defaults.items():
            setattr(self, prop, kwargs.get(prop, default))

        self.phase_bins = phase_bins
        self.mag_bins = mag_bins

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

    @property
    def ce(self):
        return self.ce_vals_out

    @property
    def ce_cpu(self):
        return np.asarray(self.ce_vals_out)

    @property
    def min_ce(self):
        shape = self.ce_vals_out.shape
        return xp.min(self.ce_vals_out.reshape(shape[0], -1), axis=1)

    def best_params(self):
        best_freqs = []
        best_pdots = []
        for sub in self.ce_vals_out:
            inds = xp.where(sub == sub.min())

            best_freqs.append(self.freqs[inds])
            best_pdots.append(self.pdots[inds])
        if len(self.pdots) == 1:
            return best_freqs

        else:
            return best_freqs, best_pdots

    def min_indices(self):
        return xp.where(self.ce_vals_out == xp.min(self.ce_vals_out))

    @property
    def mean_ce(self):
        shape = self.ce_vals_out.shape
        return xp.mean(self.ce_vals_out.reshape(shape[0], -1), axis=1)

    @property
    def std_ce(self):
        shape = self.ce_vals_out.shape
        return xp.std(self.ce_vals_out.reshape(shape[0], -1), axis=1)

    @property
    def significance(self):
        return (self.mean_ce - self.min_ce) / self.std

    def batched_run_const_nfreq(
        self,
        lightcurves,
        batch_size,
        freqs,
        pdots=None,
        return_type="all",
        pdot_batch_size=1,
        show_progress=True,
    ):
        """Run gce on lights curves.

        This method actually performs the gce calculation.

        args:
            lightcurves (list of numpy ndarrays): Light curves to be analyzed
                given as a list of numpy arrays containing the light curve
                information. The light curve arrays have shape
                (2, light curve length), where 2 is the time and magnitdue.

            batch_size (int): Number of light curves to run through gce for each
                computation. Usually, the larger the number, the faster the
                algorithm will perform, but with diminishing returns. However,
                there is an upper limit due to memory on the GPU.

            pdot_batch_size (int): Number of pdots to run for each computation.
                Choice here affects the memory usage and the GPU and splits up
                the calculation as needed to stay within memory limits. When
                len(pdots) > pdot_batch_size, the calculations are run
                separately and then combined, including statistics of the
                conditional entropy values.

            freqs (1D array): Array with the frequencies to be tested.
            pdots (1D array, optional): Array with pdots to be tested. Default
                is `None` indicating a test with 0.0 for the pdot.

            return_type (str, optional): User chosen return type.

                Options:
                    `all`: Returned conditional entropy values for every grid
                        point.
                    `best`: Return best conditional entropy value.
                    `best_params`: Return best frequency and pdot values.

            show_progress (bool, optional): Shows progress of batch calculations
                using tqdm. Default is True.

        Returns:
            array or lists of arrays: Return all conditional entropy values, the best conditional
                entropy values, or the best parameters for each light curve.

        Raises:
            TypeError: `return_type` is set to a value that is not in the
                options list.


        """
        if return_type not in ["all", "best", "best_params"]:
            raise TypeError(
                "Variable `return_type` must be set to either 'all', 'best', or 'best_params'"
            )

        if pdots is None:
            self.pdots = xp.asarray([0.0])

        self.pdots = pdots
        self.freqs = freqs

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
            ).astype(xp.int32)

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

        self.ce_vals_out = np.concatenate(ce_vals_out, axis=0)
        return self.ce_vals_out
