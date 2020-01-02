import numpy as np
from gcex.utils.getlcs import get_lcs
from gcex.utils.io import cosmic_read_helper
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
    limiting_inc=70.0,
)

"""plt.loglog(input_dict["radius_1"], input_dict["radius_2"], ".")
plt.xlabel("r1/a")
plt.ylabel("r2/a")
plt.savefig("r1_vs_r2.pdf", dpi=200)
plt.show()
"""

print("num lcs:", input_dict["radius_1"].shape)
lcs = get_lcs(
    input_dict,
    min_pts=275,
    max_pts=325,
    verbose=25,
    mean_dt=3,
    sig_t=2,
    pickle_out=True,
    file_name_out="input/light_curves",
    limiting_inc=70.0,  # degrees
    num_procs=None,
)
