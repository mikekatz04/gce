import numpy as np
import pdb
import time

from gcex.utils.io import read_out_to_hdf5, cosmic_read_helper, read_helper
from gcex.utils.getlcs import get_lcs

try:
    from gcex.gce import ConditionalEntropy
except ImportError:
    print("GCE not found.")

M_sun = 1.989e30


def test(input_dict, output_string):

    num_pdots = int(2 ** 8)
    max_pdot = 1e-10
    min_pdot = 1e-12
    test_pdots = np.logspace(np.log10(min_pdot), np.log10(max_pdot), num_pdots)

    num_freqs = int(2 ** 13)
    min_period = 3 / (24 * 60.0)  # 3 minutes
    max_period = 50.0  # 50 days
    min_freq = 1.0 / max_period
    max_freq = 1.0 / min_period
    test_freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), num_freqs)

    input_dict = {key: input_dict[key][0:3] for key in input_dict}
    lcs = get_lcs(input_dict, min_pts=95, max_pts=105, verbose=25, mean_dt=3, sig_t=2)

    num_lcs = len(lcs)

    # pyce_checks = np.asarray(ce_checks)
    ce = ConditionalEntropy(phase_bins=50)
    batch_size = 200

    st = time.perf_counter()

    output = ce.batched_run_const_nfreq(
        lcs, batch_size, test_freqs, test_pdots, show_progress=True
    )
    et = time.perf_counter()
    print(
        "Time per frequency per pdot per light curve:",
        (et - st) / (num_lcs * num_freqs * num_pdots),
    )
    print(
        "Total time for {} light curves and {} frequencies and {} pdots:".format(
            num_lcs, num_freqs, num_pdots
        ),
        et - st,
    )

    read_out_to_hdf5(output_string, input_dict, output, test_freqs, test_pdots)


if __name__ == "__main__":
    # input_dict = read_helper('test_params.txt')
    input_dict = cosmic_read_helper(
        "input/gx_save_lambda_var_alpha_025.csv",
        x_sun=0.0,
        y_sun=0.0,
        z_sun=0.0,
        use_gr=False,
    )
    print("Read data complete.")
    test(input_dict, "gce_output_small_test")
