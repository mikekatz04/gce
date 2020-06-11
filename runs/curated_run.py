import numpy as np
import pdb
import time
import pickle

from gcex.utils.input_output import read_in_for_paper

from gcex.gce import ConditionalEntropy

M_sun = 1.989e30

num_pdots = int(2 ** 9)
min_pdot = 1e-10
max_pdot = 1e-12
test_pdots = -1 * np.logspace(np.log10(min_pdot), np.log10(max_pdot), num_pdots)
# test_pdots = np.array([0.0])

baseline = 3600
# test_pdots = test_pdots[0:1]
if baseline < 10:
    fmin, fmax = 18, 1440
else:
    fmin, fmax = 2 / baseline, 480
    # fmin, fmax = 1.0, 1e3

samples_per_peak = 3

df = 1.0 / (samples_per_peak * baseline)
nf = int(np.ceil((fmax - fmin) / df))
test_freqs = fmin + df * np.arange(nf)
test_freqs = test_freqs

try:
    fill = len(test_pdots)

except TypeError:
    fill = 0


ce = ConditionalEntropy(phase_bins=50, mag_bins=10, use_long=False)

batch_size = 20


num_light_curve_sets = 3
for i in range(0, num_light_curve_sets):  # num_light_curve_sets):
    lcs, true_vals = read_in_for_paper(
        "input/curated_data_{}_16.pickle".format(i), true_mag=True
    )

    print("Read data complete.")

    ce.batched_run_const_nfreq(
        lcs,
        batch_size,
        test_freqs,
        pdots=test_pdots,
        pdot_batch_size=4,
        return_type="best_params",
        show_progress=True,
        pickle_out=[
            "truth",
            "lcs",
            "significance",
            "best_params",
            "test_freqs",
            "test_pdots",
        ],
        pickle_string="/projects/b1095/mkatz/gce/ce_curated_50_bins_512_{}".format(i),
        true_vals=true_vals,
        convert_f_to_p=True,
    )

    print("finished ", i)
