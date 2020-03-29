import numpy as np
import pdb
import time
import pickle

from gcex.utils.input_output import (
    read_out_to_hdf5,
    cosmic_read_helper,
    read_helper,
    read_in_for_paper,
)
from gcex.utils.getlcs import get_lcs

try:
    from gcex.gce import ConditionalEntropy
except ImportError:
    print("GCE not found.")

M_sun = 1.989e30


def run_search(lcs, true_vals):

    num_pdots = int(2 ** 7)
    min_pdot = 1e-14
    max_pdot = 1e-11
    test_pdots = -1 * np.logspace(np.log10(min_pdot), np.log10(max_pdot), num_pdots)

    # test_pdots = np.array([0.0])

    baseline = 1200.0
    if baseline < 10:
        fmin, fmax = 18, 1440
    else:
        fmin, fmax = 2 / baseline, 480
        # fmin, fmax = 1.0, 1e3

    samples_per_peak = 1

    df = 1.0 / (samples_per_peak * baseline)
    nf = int(np.ceil((fmax - fmin) / df))
    test_freqs = fmin + df * np.arange(nf)

    num_lcs = len(lcs)

    # pyce_checks = np.asarray(ce_checks)
    ce = ConditionalEntropy(phase_bins=50)
    batch_size = 1

    out = {}
    out["df"] = df
    out["test_freqs"] = test_freqs
    out["test_pdots"] = test_pdots
    out["nf"] = nf
    out["res"] = []
    out["truth"] = true_vals

    for i in range(len(lcs)):
        temp = {}
        output = ce.batched_run_const_nfreq(
            lcs[i : i + 1], batch_size, test_freqs, test_pdots, show_progress=True
        )
        ch = output[0]
        sig = (np.min(ch) - np.mean(ch)) / np.std(ch)
        inds_best = np.where(ch == ch.min())
        temp["sig"] = sig
        temp["inds_best"] = inds_best
        out["res"].append(temp)

    with open(
        "/projects/b1095/mkatz/gce/check_data_cosmic_pdot.pickle", "wb"
    ) as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # read_out_to_hdf5(output_string, input_dict, output, test_freqs, test_pdots)


if __name__ == "__main__":
    # input_dict = read_helper('test_params.txt')
    lcs, true_vals = read_in_for_paper("gce/input/light_curves.pickle", true_mag=True)

    print("Read data complete.")
    run_search(lcs, true_vals)
