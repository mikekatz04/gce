import numpy as np
cimport numpy as np

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "src/manager.hh":
    cdef cppclass GCEWrap "GCE":
        GCEWrap(int, int)
        void conditional_entropy(np.float64_t*, int, np.float64_t*, np.float64_t*, np.float64_t*, np.int32_t*, int, np.float64_t*, int)

cdef class GCE:
    cdef GCEWrap* g
    cdef int num_lcs
    cdef int num_freqs

    def __cinit__(self, phase_bins, mag_bins):
        self.g = new GCEWrap(phase_bins, mag_bins)

    def conditional_entropy(self,
                            np.ndarray[ndim=1, dtype=np.float64_t] time_vals,
                            np.ndarray[ndim=1, dtype=np.float64_t] mag_vals,
                            np.ndarray[ndim=1, dtype=np.float64_t] mag_bin_edges,
                            np.ndarray[ndim=1, dtype=np.int32_t] num_pts_arr,
                            np.ndarray[ndim=1, dtype=np.float64_t] freqs):

        cdef np.ndarray[ndim=1, dtype=np.float64_t] ce_vals = np.zeros((len(num_pts_arr)*len(freqs),), dtype=np.float64)

        num_pts_max = np.max(num_pts_arr)
        self.num_lcs = len(num_pts_arr)
        self.num_freqs = len(freqs)
        self.g.conditional_entropy(&ce_vals[0], self.num_lcs, &time_vals[0], &mag_vals[0], &mag_bin_edges[0], &num_pts_arr[0], num_pts_max, &freqs[0], self.num_freqs)

        return ce_vals.reshape(self.num_lcs, self.num_freqs)
