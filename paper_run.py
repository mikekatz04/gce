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


def run_search(lcs, true_vals):

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

    num_lcs = len(lcs)

    # pyce_checks = np.asarray(ce_checks)
    ce = ConditionalEntropy(phase_bins=50)
    batch_size = 200

    output = ce.batched_run_const_nfreq(
        lcs, batch_size, test_freqs, test_pdots, show_progress=True
    )

    import pdb

    pdb.set_trace()
    # read_out_to_hdf5(output_string, input_dict, output, test_freqs, test_pdots)


if __name__ == "__main__":
    # input_dict = read_helper('test_params.txt')
    lcs, true_vals = read_in_for_paper("input/light_curves.pickle")
    print("Read data complete.")
    run_search(lcs, true_vals)
