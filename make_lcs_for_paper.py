import numpy as np
from gcex.utils.getlcs import get_lcs
from gcex.utils.io import cosmic_read_helper
import pdb

input_dict = cosmic_read_helper(
    ["ThinDisk", "ThickDisk", "Bulge"],
    "input/",
    "_out_file.hdf5",
    x_sun=8.5,
    y_sun=0.0,
    z_sun=0.0,
    use_gr=True,
    limiting_inc=80.0,
)

import pdb

pdb.set_trace()

lcs = get_lcs(
    input_dict,
    min_pts=275,
    max_pts=325,
    verbose=25,
    mean_dt=3,
    sig_t=2,
    pickle_out=True,
    file_name_out="input/light_curves",
    limiting_inc=80.0,  # degrees
    num_procs=None,
)

import pdb

pdb.set_trace()
