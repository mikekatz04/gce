import numpy as np
import pdb
import scipy.constants as ct
import time
from astropy.io import ascii

from gcex.gce import ConditionalEntropy
from ztfperiodic import simulate

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


def read_helper(fp):
    # add new routines as necessary
    if fp[-4:] == '.txt':
        data = np.asarray(ascii.read(fp))
        keys = data.dtype.names

    params = {key: data[key] for key in keys if key not in ['m1', 'm2']}
    if 'q' not in keys:
        m1 = data['m1']
        m2 = data['m2']
        params['q'] = m1/m2 * (m1 >= m2) + m2/m1 * (m1 < m2)
    return params


def test(input_dict):
    keys = list(input_dict.keys())
    num_lcs = len(input_dict[keys[0]])
    num_freqs = int(2**13)
    num_pdots = int(2**8)
    min_period = 3 * 60.0  # 3 minutes
    max_period = 50.0*24*3600.0  # 50 days

    max_pdot = 1e-6
    min_pdot = 1e-8

    min_freq = 1./max_period
    max_freq = 1./min_period

    baseline = 1*ct.Julian_year # 30 days

    test_freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), num_freqs)
    test_pdots = np.logspace(np.log10(min_pdot), np.log10(max_pdot), num_pdots)

    lcs = []
    ce_checks = []
    number_of_pts = np.random.random_integers(80, 97, size=num_lcs)
    for lc_i, n in zip(np.arange(num_lcs), number_of_pts):
        # form dictionary
        params = {key: input_dict[key][lc_i] for key in keys}
        import pdb; pdb.set_trace()

        t_obs = simulate.time(n=n, mean_dt=3, sig_t=2)
        mag, phase, err = simulate.pdot_lc(t_obs, **params)
        """mag=None, absmag=True, d=None, Pdot=Pdot, radius_1=r1/a, radius_2=r2/a, sbratio=sbratio, incl=i,
           light_3 = 0, t_zero = 0, period = P0, a = a, q = m1/m2,
           f_c = None, f_s = None,
           ldc_1 = None, ldc_2 = None,
           gdc_1 = None, gdc_2 = None,
           didt = None, domdt = None,
           rotfac_1 = 1, rotfac_2 = 1,
           hf_1 = 1.5, hf_2 = 1.5,
           bfac_1 = None, bfac_2 = None,
           heat_1 = None, heat_2 = None,
           lambda_1 = None, lambda_2 = None,
           vsini_1 = None, vsini_2 = None,
           t_exp=None, n_int=None,
           grid_1='default', grid_2='default',
           ld_1=None, ld_2=None,
           shape_1='sphere', shape_2='sphere',
           spots_1=None, spots_2=None,
           exact_grav=False, verbose=1, plot_nopdot=True,savefig=False)"""

        lcs.append(np.array([t_obs, mag]).T)


        #check = py_check_ce(test_freqs, time_vals, mags, mag_bins=10, phase_bins=15, verbose=verbose)
        #ce_checks.append(check)

    #pyce_checks = np.asarray(ce_checks)

    ce = ConditionalEntropy()
    batch_size = 200

    st = time.perf_counter()

    check = ce.batched_run_const_nfreq(lcs, batch_size, test_freqs, test_pdots, show_progress=True)
    et = time.perf_counter()
    print('Time per frequency per pdot per light curve:', (et - st)/(num_lcs*num_freqs*num_pdots))
    print('Total time for {} light curves and {} frequencies and {} pdots per lc:'.format(num_lcs, num_freqs, num_pdots), et - st)
    #checker = actual_freqs/test_freqs[np.argmin(check, axis=1)]

    #sig = (np.min(check, axis=1) - np.mean(check, axis=1))/np.std(check, axis=1)

    #print('num > 10:', len(np.where(sig<-10.0)[0])/num_lcs)

    import pdb; pdb.set_trace()


if __name__ == "__main__":
    input_dict = read_helper('test_params.txt')
    test(input_dict)
