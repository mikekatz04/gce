import numpy as np
import pdb
import scipy.constants as ct
import time

from gce.gce import ConditionalEntropy

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

def test():

    num_lcs = 1000
    num_freqs = int(2**16)
    min_period = 3 * 60.0  # 3 minutes
    max_period = 50.0*24*3600.0  # 50 days

    min_freq = 1./max_period
    max_freq = 1./min_period

    print('%.18e'%min_freq)

    baseline = 1*ct.Julian_year # 30 days

    test_freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), num_freqs)

    actual_freqs = np.random.uniform(low=min_freq, high=max_freq, size=num_lcs)

    number_of_pts = np.random.random_integers(250, 300, size=num_lcs)

    max_mag_factor = 10.0
    min_mag_factor = 1.0
    mag_factors = np.random.uniform(low=min_mag_factor, high=max_mag_factor, size=num_lcs)

    lcs = []
    ce_checks = []
    for i, (num_pts, freq, mag_fac) in enumerate(zip(number_of_pts, actual_freqs, mag_factors)):
        time_vals = np.random.uniform(low=0.0, high=baseline, size=num_pts)
        initial_phase = np.random.uniform(low=0.0, high=2*np.pi)
        vert_shift = np.random.uniform(low=mag_fac, high=3*mag_fac)
        mags = mag_fac*np.sin(2*np.pi*freq*time_vals + initial_phase) + vert_shift
        lcs.append(np.array([time_vals, mags]).T)
        verbose = True if i == 1 else False
        #check = py_check_ce(test_freqs, time_vals, mags, mag_bins=10, phase_bins=15, verbose=verbose)
        #ce_checks.append(check)

    #pyce_checks = np.asarray(ce_checks)

    ce = ConditionalEntropy()
    batch_size = 200

    st = time.perf_counter()
    check = ce.batched_run_const_nfreq(lcs, batch_size, test_freqs, show_progress=True)
    et = time.perf_counter()
    print((et - st)/(num_lcs*num_freqs))
    checker = actual_freqs/test_freqs[np.argmin(check, axis=1)]

    sig = (np.min(check, axis=1) - np.mean(check, axis=1))/np.std(check, axis=1)

    print('num > 10:', len(np.where(sig<-10.0)[0])/num_lcs)

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    test()
