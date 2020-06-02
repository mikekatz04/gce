// Copyright (C) Michael Katz (2020)
//
// This file is part of gce
//
// gce is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// gce is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with gce.  If not, see <http://www.gnu.org/licenses/>



#ifndef __KERNEL__HH__
#define __KERNEL__HH__

#include "global.h"


void run_gce(fod *d_ce_vals, fod *d_freqs, int num_freqs, fod *d_pdots, int num_pdots, int *d_mag_bin_inds, fod *d_time_vals,
             int *d_num_pts_arr, int num_pts_max, int mag_bins, int phase_bins, int num_lcs, fod half_dbins);

void run_long_lc_gce(fod *d_ce_vals, fod *d_freqs, int num_freqs, fod *d_pdots, int num_pdots, int *d_mag_bin_inds, fod *d_time_vals,
                          int *d_num_pts_arr, int num_pts_max, int mag_bins, int phase_bins, int num_lcs, fod half_dbins);

#endif // __KERNEL__HH__
