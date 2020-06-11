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
import scipy.constants as ct
import pickle


def read_in_for_paper(
    fp,
    true_mag=True,
    subtract_median=True,
    mag_limit=20.5,
    rand_phase_shift=True,
    pdot_cut=-1e-13,
    mag_diff_cut=0.1,
    num_max=None,
):
    """Read in light curves from file.

    Read in light curves saved in the format used for the gce paper.
    The light curves are saved in a dictionary object.

    Args:
        fp (str): File name (.pickle) with the light curves.
        true_mag (bool, optional): If True, return magnitude with errors in
            measurements. If False, return idealized light curves
            with no errors. Default is True.
        subtract_median (str): If True, subtract median light curve value
            from each value. This normalization can help when using
            different bands. Default is True.
        mag_limit (float, optional): Magnitude limit above which observations
            are removed. When sampling with errors, observation points
            may extend beyong the sensitivity of a telescope when examining
            fringe sources. Default is 20.5.
        rand_phase_shift (bool, optional): If True, assign a random initial
            phase to each light curve from 0 to 2pi. Default is True.
        pdot_cut (float, optional): Make cut at a specific pdot_gw.
            Default is -1e-13 (). This indicates values less than -1e-13.
        mag_diff_cut (float, optional): Make a cut on the difference between
            the minimum observation and median observation. Default is 0.1.
            This will cut any values below 0.1.

    Returns:
        tuple: (list of Light curve arrays, dictionary of true parameters)

    """
    with open(fp, "rb") as f:
        in_vals = pickle.load(f)

    pdot = []
    period = []
    keys = []
    mag_check = []

    # read in binaries
    for i, (key, binary) in enumerate(in_vals.items()):

        pdot.append(binary.get("params").get("Pdot"))
        period.append(binary.get("params").get("period"))
        keys.append(key)

        mag = binary.get("true_mag")
        mag_check.append(mag.max() - np.median(mag))

    # choose what to keep
    if pdot_cut is None:
        pdot_cut = -1e-20

    if mag_diff_cut is None:
        mag_diff_cut > 1e-10

    keep = np.where(
        (np.asarray(pdot) < pdot_cut) & (np.asarray(mag_check) > mag_diff_cut)
    )[0]

    pdot = np.asarray(pdot)[keep]
    period = np.asarray(period)[keep]
    keys = np.asarray(keys)[keep]

    sort = np.argsort(pdot)

    pdot = np.asarray(pdot)[sort]
    period = np.asarray(period)[sort]
    keys = np.asarray(keys)[sort]

    # prepare actual outputs
    out_list = []
    params = []
    for key in keys:
        temp = in_vals[key]
        t = temp.get("t")
        if true_mag:
            mag = temp.get("true_mag")

        else:
            mag = temp.get("mag")

        if subtract_median:
            mag = mag - np.median(mag)

        inds = np.where(mag <= mag_limit)
        t = t - t.min()
        period = temp.get("params").get("period")

        # add random phase shift
        if rand_phase_shift:
            t += np.random.rand() * period

        out_list.append(np.array([t[inds], mag[inds], mag[inds]]).T)
        params.append(temp.get("params"))

    # cut number of light curves
    if num_max is not None:
        if num_max < len(out_list):
            out_list = out_list[:num_max]
            params = params[:num_max]

    return out_list, params
