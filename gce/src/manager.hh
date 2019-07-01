#ifndef __MANAGER_H__
#define __MANAGER_H__
#include "global.h"

class GCE {
  // pointer to the GPU memory where the array is stored
  int phase_bins;
  int mag_bins;

  fod *phase_bin_edges;
  fod *d_phase_bin_edges;

  int num_lcs;
  int num_freqs;
  int num_pdots;

  fod *d_time_vals;
  fod *d_mag_vals;
  fod *d_mag_bin_edges;
  int *d_num_pts_arr;
  fod *d_freqs;
  fod *d_pdots;
  fod *d_ce_vals;
  fod *d_min_light_curve_times;


public:
  /* By using the swig default names INPLACE_ARRAY1, DIM1 in the header
     file (these aren't the names in the implementation file), we're giving
     swig the info it needs to cast to and from numpy arrays.

     If instead the constructor line said
       GPUAdder(int* myarray, int length);

     We would need a line like this in the swig.i file
       %apply (int* ARGOUT_ARRAY1, int DIM1) {(int* myarray, int length)}
   */

  GCE(int phase_bins_, int mag_bins_); // constructor (copies to GPU)

  ~GCE(); // destructor

  void conditional_entropy(fod *ce_vals, int num_lcs_, fod *time_vals, fod *mag_vals, fod *mag_bin_edges, int *num_pts_arr, int num_pts_max, fod *freqs, int num_freqs_, fod *pdots, int num_pdots, fod *min_light_curve_times);
};

#endif //__MANAGER_H__
