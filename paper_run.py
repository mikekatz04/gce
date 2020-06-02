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
