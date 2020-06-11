import numpy as np
import pdb
import time
import pickle

from gcex.utils.input_output import read_in_for_paper

from gcex.gce import ConditionalEntropy

M_sun = 1.989e30

num_pdots = int(2 ** 2)
min_pdot = 1e-10
max_pdot = 1e-12
test_pdots = -1 * np.logspace(np.log10(min_pdot), np.log10(max_pdot), num_pdots)
# test_pdots = np.array([0.0])

baseline = 3600
if baseline < 10:
    fmin, fmax = 18, 1440
else:
    fmin, fmax = 2 / baseline, 480

samples_per_peak = 3

df = 1.0 / (samples_per_peak * baseline)
nf = int(np.ceil((fmax - fmin) / df))
test_freqs = fmin + df * np.arange(nf)

lcs, true_vals = read_in_for_paper(
    "input/cosmic_data.pickle", true_mag=True, num_max=100
)

print("Read data complete.")

try:
    fill = len(test_pdots)

except TypeError:
    fill = 0

output_string = "paper/check_data_cosmic_0_pdot.pickle".format(fill)

ce = ConditionalEntropy(phase_bins=20, mag_bins=10, use_long=False)

batch_size = 20

ce.batched_run_const_nfreq(
    lcs[0:5],
    batch_size,
    test_freqs[0::10],
    pdots=test_pdots,
    pdot_batch_size=2,
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
    pickle_string="test_out_file".format(fill),
    true_vals=true_vals[0:5],
    convert_f_to_p=True,
)
