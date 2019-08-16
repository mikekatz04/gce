import numpy as np


def py_check_ce(freqs, time_vals, magnitude_vals, mag_bins=10, phase_bins=15, verbose=False):
    ce_vals = np.zeros_like(freqs)
    for i, freq in enumerate(freqs):
        period = 1./freq

        folded = (time_vals % period)*freq
        bin_edges_mag = np.linspace(magnitude_vals.min()*0.999, magnitude_vals.max()*1.001, mag_bins+1)
        bins_edges_phase = np.linspace(0.000001, 1.000001, phase_bins+1)
        H, x, y = np.histogram2d(folded, magnitude_vals, bins=[bins_edges_phase, bin_edges_mag])

        p_ij_all = H/len(time_vals)
        p_j_all = np.tile(np.sum(p_ij_all, axis=1), (mag_bins,1)).T
        inds = np.where((p_ij_all != 0.0) & (p_j_all != 0.0))
        p_ij = p_ij_all[inds]
        p_j = p_j_all[inds]
        ce = np.sum(p_ij * np.log(p_j/p_ij))
        ce_vals[i] = ce
        if False: #i == 255 and verbose:
            print(freq)
            print(bin_edges_mag)
            print(p_j_all)
            print(p_ij_all)
            print(ce ,'\n\n\n')

    return ce_vals
