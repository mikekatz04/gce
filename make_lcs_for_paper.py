import numpy as np
from gcex.utils.getlcs import get_lcs
from gcex.utils.input_output import cosmic_read_helper
import pdb
import matplotlib.pyplot as plt

input_dict = cosmic_read_helper(
    ["ThinDisk", "ThickDisk", "Bulge"],
    "input/",
    "_out_file.hdf5",
    x_sun=8.5,
    y_sun=0.0,
    z_sun=0.0,
    use_gr=True,
    limiting_inc=0.0,
    limiting_period=1 / 24.0,
    limiting_magnitude=22.0,
)

"""
plt.loglog(input_dict["radius_1"], input_dict["radius_2"], ".")
plt.xlabel("r1/a")
plt.ylabel("r2/a")
plt.savefig("r1_vs_r2.pdf", dpi=200)
plt.show()
"""

print("num lcs:", input_dict["radius_1"].shape)
lcs = get_lcs(
    input_dict,
    min_pts=500,
    max_pts=520,
    verbose=25,
    mean_dt=7,
    sig_t=2,
    pickle_out=True,
    file_name_out="/projects/b1095/mkatz/gce/light_curves_more_7day_cadence.pickle",
    limiting_inc=0.0,  # degrees
    num_procs=None,
)
