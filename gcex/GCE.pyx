import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/manager.hh":
    cdef cppclass GCEWrap "GCE":
        GCEWrap(int, int)
        void conditional_entropy(np.float32_t*, int, np.float32_t*, np.int32_t*, np.int32_t*, int, np.float32_t*, int, np.float32_t*, int, np.float32_t*)

cdef class GCE:
    cdef GCEWrap* g
    cdef int num_lcs
    cdef int num_freqs

    def __cinit__(self, phase_bins, mag_bins):
        self.g = new GCEWrap(phase_bins, mag_bins)

    def conditional_entropy(self,
                            np.ndarray[ndim=1, dtype=np.float32_t] time_vals,
                            np.ndarray[ndim=1, dtype=np.int32_t] mag_bin_inds,
                            np.ndarray[ndim=1, dtype=np.int32_t] num_pts_arr,
                            np.ndarray[ndim=1, dtype=np.float32_t] freqs,
                            np.ndarray[ndim=1, dtype=np.float32_t] pdots,
                            np.ndarray[ndim=1, dtype=np.float32_t] min_light_curve_times):

        cdef np.ndarray[ndim=1, dtype=np.float32_t] ce_vals = np.zeros((len(num_pts_arr)*len(freqs)*len(pdots),), dtype=np.float32)

        num_pts_max = np.max(num_pts_arr)
        num_lcs = len(num_pts_arr)
        num_freqs = len(freqs)
        num_pdots = len(pdots)
        self.g.conditional_entropy(&ce_vals[0], num_lcs, &time_vals[0], &mag_bin_inds[0], &num_pts_arr[0], num_pts_max, &freqs[0], num_freqs, &pdots[0], num_pdots, &min_light_curve_times[0])

        return ce_vals.reshape(num_lcs, num_pdots, num_freqs)
