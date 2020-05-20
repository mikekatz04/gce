import numpy as np
cimport numpy as np

from gcex.utils.pointer_adjust import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/kernel.hh":
    ctypedef void* fod 'fod'
    void run_gce(fod *d_ce_vals, fod *d_freqs, int num_freqs, fod *d_pdots, int num_pdots, fod *d_phase_bin_edges, int *d_mag_bin_inds, fod *d_time_vals,
                 int *d_num_pts_arr, int num_pts_max, int mag_bins, int phase_bins, int num_lcs, fod *d_min_light_curve_times, float half_dbins);


@pointer_adjust
def run_gce_wrap(ce_vals, freqs, num_freqs, pdots,
                 num_pdots, phase_bin_edges, mag_bin_inds,
                 time_vals, num_pts_arr, num_pts_max, mag_bins,
                 phase_bins, num_lcs, min_light_curve_times, half_dbins):

    cdef size_t ce_vals_in = ce_vals
    cdef size_t freqs_in = freqs
    cdef size_t pdots_in = pdots
    cdef size_t phase_bin_edges_in = phase_bin_edges
    cdef size_t mag_bin_inds_in = mag_bin_inds
    cdef size_t time_vals_in = time_vals
    cdef size_t num_pts_arr_in = num_pts_arr
    cdef size_t min_light_curve_times_in = min_light_curve_times

    run_gce(<fod*> ce_vals_in, <fod*> freqs_in, num_freqs, <fod*> pdots_in, num_pdots,
            <fod*> phase_bin_edges_in, <int*> mag_bin_inds_in, <fod*> time_vals_in,
            <int*> num_pts_arr_in, num_pts_max, mag_bins,
            phase_bins, num_lcs, <fod*> min_light_curve_times_in, half_dbins)
