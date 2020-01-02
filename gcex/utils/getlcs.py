import numpy as np
import pickle
import multiprocessing as mp

from ztfperiodic import simulate


def parallel_func(
    input_dict, lc_i, n, mean_dt, sig_t, keys, limiting_inc, verbose, kwargs
):
    # form dictionary
    name = input_dict["name"][lc_i]
    params = {key: input_dict[key][lc_i] for key in keys}

    import pdb

    pdb.set_trace()

    params["incl"] = 89.5
    if params["incl"] <= limiting_inc or params["incl"] >= 180.0 - limiting_inc:
        return (None, None)

    t_obs = simulate.time(n=n, mean_dt=mean_dt, sig_t=sig_t)

    mag, phase, err = simulate.pdot_lc(t_obs, plot_nopdot=None, **{**params, **kwargs})
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

    if verbose:
        if lc_i % verbose == 0:
            print(lc_i)
    # check = py_check_ce(test_freqs, time_vals, mags, mag_bins=10, phase_bins=15, verbose=verbose)
    # ce_checks.append(check)
    if np.all(mag == 1.0):
        return (None, None)
    return (name, np.array([t_obs, mag, err]))


def get_lcs(
    input_dict,
    min_pts=95,
    max_pts=105,
    verbose=25,
    mean_dt=20,
    sig_t=10,
    pickle_out=False,
    file_name_out="light_curves",
    limiting_inc=80.0,  # degrees
    num_procs=None,
    **kwargs
):
    keys = list(input_dict.keys())
    num_lcs = len(input_dict[keys[0]])

    out_dict = {}
    # ce_checks = []
    number_of_pts = np.random.random_integers(min_pts, max_pts, size=num_lcs)[0:100]

    args = [
        (input_dict, lc_i, n, mean_dt, sig_t, keys, limiting_inc, verbose, kwargs)
        for lc_i, n, in zip(np.arange(num_lcs), number_of_pts)
    ]

    if num_procs is None:
        num_procs = mp.cpu_count()

    # test
    check = parallel_func(*args[0])
    # import pdb; pdb.set_trace()
    """print(num_procs)
    with mp.Pool(num_procs) as pool:
        results = [pool.apply_async(parallel_func, arg) for arg in args]
        results = [res.get() for res in results]
    """
    results = [parallel_func(*arg) for arg in args]
    import pdb

    pdb.set_trace()

    if pickle_out:
        with open(file_name_out + ".pickle", "wb") as handle:
            pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return lcs


def get_lcs_test(
    input_dict, min_pts=95, max_pts=105, verbose=25, mean_dt=20, sig_t=10, **kwargs
):
    keys = list(input_dict.keys())
    num_lcs = len(input_dict[keys[0]])

    lcs = []
    # ce_checks = []
    number_of_pts = np.random.random_integers(min_pts, max_pts, size=num_lcs)
    for lc_i, n in zip(np.arange(num_lcs), number_of_pts):
        # form dictionary
        params = {key: input_dict[key][lc_i] for key in keys}

        t_obs = simulate.time(n=n, mean_dt=mean_dt, sig_t=sig_t)

        mag, phase, err = simulate.pdot_lc(
            t_obs, plot_nopdot=None, **{**params, **kwargs}
        )
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

        lcs.append(np.array([t_obs, mag, err]).T)
        if verbose:
            if lc_i % verbose == 0:
                print(lc_i)
        # check = py_check_ce(test_freqs, time_vals, mags, mag_bins=10, phase_bins=15, verbose=verbose)
        # ce_checks.append(check)
    return lcs
