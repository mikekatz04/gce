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
cimport numpy as np

from gcex.utils.pointer_adjust import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/kernel.hh":
    ctypedef void* fod 'fod'
    void run_gce(fod *d_ce_vals, fod *d_freqs, int num_freqs, fod *d_pdots, int num_pdots, int *d_mag_bin_inds, fod *d_time_vals,
                 int *d_num_pts_arr, int num_pts_max, int mag_bins, int phase_bins, int num_lcs, float half_dbins);

    void run_long_lc_gce(fod *d_ce_vals, fod *d_freqs, int num_freqs, fod *d_pdots, int num_pdots, int *d_mag_bin_inds, fod *d_time_vals,
                 int *d_num_pts_arr, int num_pts_max, int mag_bins, int phase_bins, int num_lcs, float half_dbins);


@pointer_adjust
def run_gce_wrap(ce_vals, freqs, num_freqs, pdots,
                 num_pdots, mag_bin_inds,
                 time_vals, num_pts_arr, num_pts_max, mag_bins,
                 phase_bins, num_lcs, half_dbins):

    cdef size_t ce_vals_in = ce_vals
    cdef size_t freqs_in = freqs
    cdef size_t pdots_in = pdots
    cdef size_t mag_bin_inds_in = mag_bin_inds
    cdef size_t time_vals_in = time_vals
    cdef size_t num_pts_arr_in = num_pts_arr

    run_gce(<fod*> ce_vals_in, <fod*> freqs_in, num_freqs, <fod*> pdots_in, num_pdots,
            <int*> mag_bin_inds_in, <fod*> time_vals_in,
            <int*> num_pts_arr_in, num_pts_max, mag_bins,
            phase_bins, num_lcs, half_dbins)


@pointer_adjust
def run_long_lc_gce_wrap(ce_vals, freqs, num_freqs, pdots,
                 num_pdots, mag_bin_inds,
                 time_vals, num_pts_arr, num_pts_max, mag_bins,
                 phase_bins, num_lcs, half_dbins):

    cdef size_t ce_vals_in = ce_vals
    cdef size_t freqs_in = freqs
    cdef size_t pdots_in = pdots
    cdef size_t mag_bin_inds_in = mag_bin_inds
    cdef size_t time_vals_in = time_vals
    cdef size_t num_pts_arr_in = num_pts_arr

    run_long_lc_gce(<fod*> ce_vals_in, <fod*> freqs_in, num_freqs, <fod*> pdots_in, num_pdots,
            <int*> mag_bin_inds_in, <fod*> time_vals_in,
            <int*> num_pts_arr_in, num_pts_max, mag_bins,
            phase_bins, num_lcs, half_dbins)
