#include <math.h>
#include <random>
#include "global.h"

#define NUM_THREADS 64

__device__
double mod_func(double x, double y)
{
    int num = (int) (x/y);

    double check = x - (num*y);

    return check;
}

__device__
int get_phase_bin(fod t_val, fod pdot, fod frequency, fod period, fod half_dbins, int phase_bins){
    double folded_val = mod_func((double) t_val-0.5*pdot*frequency*(t_val*t_val), (double)period)*((double)frequency); // between 0 and 1
    if (folded_val < 0.0) folded_val *= -1.0;

    int j = (int) abs((folded_val/(double)half_dbins));
    //printf("1: %d %d %lf %lf\n", j, phase_bins, folded_val, half_dbins);
    if ((j<0)) j = 0;
    else if (j>phase_bins)  j = phase_bins;
    //printf("2: %d %d %lf %lf\n", j, phase_bins, folded_val, half_dbins);
    return j;
}

__device__ fod ce (fod frequency, fod pdot, fod* phase_bin_edges,
                   int* mag_bin_inds, fod* time_vals, int npoints,
                   int mag_bins, int phase_bins,
                   int offset, int lc_i, fod lc_start_time,
                   fod * temp_phase_prob, fod *overall_phase_prob, fod half_dbins){
    fod period = 1./frequency;
    fod folded_val = 0.0;
    int mag_ind_1 = -1, mag_ind_2=-1;
    int l = 1;
    fod sum_ij = 0.0;
    fod t_val = 0.0;
    fod total_points = 0.0;
    int j = 0;

    for (int jj = 0; jj<phase_bins; jj+=1){
            overall_phase_prob[jj] = 0.0;
            temp_phase_prob[jj] = 0.0;
        }

    //for (int i=0; i<mag_bins; i++){
    //    mag_bin_vals[i] = 0;
    //}

    //fod current_phase_prob = 0.0;
    int temp_mag_points = 0;
    short index=0;
    int ind_mag=0;

    int temp_ind = 0;
    int gap = 0;
    for (int k=0; k<npoints; k++){

        /*# if __CUDA_ARCH__>=200
        if ((offset == 0)  && (lc_i == 0)){
            printf("(%d %d %d): %lf - %lf\n", lc_i, j, offset, phase_bin_edges[2*j], phase_bin_edges[2*j+1]);
        }
        #endif //*/
        //int k = 0;
        //while(k < npoints - gap){

            ///index = indicies[k];
            //if (index == -1) break;

            t_val = time_vals[k] - lc_start_time;
            j = get_phase_bin(t_val, pdot, frequency, period, half_dbins, phase_bins);

            temp_ind = j;
            if (j == phase_bins) temp_ind = j - 1;

            overall_phase_prob[temp_ind] += 1.0;
            total_points += 1.0;

            //if (k == 10) printf("%lf %lf %lf %lf, %d %d\n", t_val, pdot, period, half_dbins, j, phase_bins);

            if ((j > 0) && (j < phase_bins)){
                j-=1;
                overall_phase_prob[j] += 1.0;
                total_points += 1.0;
            }
        }

    ind_mag = mag_bin_inds[0];
    int kk = 0;
    while (kk < npoints){
        while ((mag_bin_inds[kk] == ind_mag)){
            t_val = time_vals[kk] - lc_start_time;
            j = get_phase_bin(t_val, pdot, frequency, period, half_dbins, phase_bins);
            temp_ind = j;
            if (j == phase_bins) temp_ind = j - 1;

            temp_phase_prob[temp_ind] += 1.0;
            //printf("%lf %lf %lf %lf, %d %d %d\n", t_val, pdot, period, half_dbins, j, mag_bin_inds[kk], ind_mag);
            if ((j > 0) && (j < phase_bins)){
                j-=1;
                temp_phase_prob[j] += 1.0;
            }

            kk += 1;
            if (kk == npoints) break;
        }

        for (int jj = 0; jj<phase_bins; jj+=1){
            if ((overall_phase_prob[jj] > 0.0) && (temp_phase_prob[jj] > 0.0)){
                 //printf("%d %d %lf %lf \n", ind_mag, jj, overall_phase_prob[jj], temp_phase_prob[jj]);
                sum_ij += temp_phase_prob[jj]*log(overall_phase_prob[jj]/temp_phase_prob[jj]);
                temp_phase_prob[jj] = 0.0;
            }
        }
        if (kk < npoints) ind_mag = mag_bin_inds[kk];
    }
                //printf("%d, %lf, %lf\n", l, mag_bins[l-1], current_phase_prob);

            //        indicies[k] = indicies[(npoints - 1) - gap];
            //        gap += 1;
            //}
            //else k++;
        //}
        /*# if __CUDA_ARCH__>=200
        if ((offset == mag_bins*255)  && (lc_i == 0)){
            printf("(%d %d %d %.18e %e)\n", lc_i, j, offset, frequency, current_phase_prob);
            for (int jj=0; jj<mag_bins; jj++) printf("%e, ", mag_bin_vals[offset + jj]);
            printf("\n");
            for (int jj=0; jj<mag_bins+1; jj++) printf("%e, ", mag_bin_edges[jj]);
            printf("\n\n");
        }
        #endif //*/


    //for (int j=0; j<phase_bins*mag_bins; j++){
        //if (current_phase_prob[j] == 0.0) continue;
        //fod curr_phase_p = current_phase_prob[j];
        //for (int i=0; i<mag_bins; i++){
            //if (bin_counter[j*mag_bins + i] > 0.0){
        //        continue;
                //if (j < 2) printf("%d, %d, %lf, %lf\n", j, i,bin_counter[j*mag_bins + i], curr_phase_p);
            //    sum_ij += (bin_counter[j*mag_bins + i])*log((curr_phase_p)/(bin_counter[j*mag_bins + i]));
        //    }

        //}
    /*# if __CUDA_ARCH__>=200
    if ((offset == mag_bins*255)  && (lc_i == 0)){
        printf("(%d %d %.18e CE: %.18e)\n", lc_i, offset, frequency, sum_ij);
    }
    #endif //*/
    //printf("%lf %lf\n", sum_ij, total_points);
    return  sum_ij/((fod) total_points);
    //return 0.0;
}

__global__ void kernel(fod* ce_vals, fod* freqs, int num_freqs, fod* pdots,
                       int num_pdots, fod* phase_bin_edges, int* mag_bin_inds,
                       fod* time_vals, int *num_pts_arr, int num_pts_max,
                       const int mag_bins, int phase_bins, int num_lcs,
                       fod* min_light_curve_times, fod half_dbins){


    // __shared__ fod share_mag_bin_vals[NUM_THREADS*10];
    extern __shared__ fod share_bins_phase[];
    __shared__ fod time_vals_share[2000];
    __shared__ int mag_bin_inds_share[2000];
    __shared__ fod temp_phase_prob[51*NUM_THREADS];
    __shared__ fod overall_phase_prob[51*NUM_THREADS];


    //for (int j=threadIdx.x; j<=phase_bins; j+=blockDim.x){
    //    share_bins_phase[2*j] = phase_bin_edges[2*j];
    //    share_bins_phase[2*j+1] = phase_bin_edges[2*j+1];
  // }

    __syncthreads();

    for (int lc_i = blockIdx.y;
         lc_i < num_lcs;
         lc_i += gridDim.y) {

     int num_pts_this_lc = num_pts_arr[lc_i];
     fod lc_start_time = min_light_curve_times[lc_i];

     for (int j=threadIdx.x; j<num_pts_this_lc; j+=blockDim.x){
         time_vals_share[j] = time_vals[lc_i*num_pts_max + j];
         mag_bin_inds_share[j] = mag_bin_inds[lc_i*num_pts_max + j];
     }

     __syncthreads();

   for (int pdot_i = blockIdx.z;
        pdot_i < num_pdots;
        pdot_i += gridDim.z) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_freqs;
         i += blockDim.x * gridDim.x) {

    // TODO: make this adjustable: https://devblogs.nvidia.com/using-shared-memory-cuda-cc/  !!!!
    int offset = mag_bins*threadIdx.x;


    //__shared__ fod share_t_vals[100];
    //__shared__ fod share_magnitude[100];

        //for (int j=0; j<num_pts_this_lc; j++) share_t_vals[j] = time_vals[lc_i*num_pts_max + j];
        //for (int j=0; j<num_pts_this_lc; j++) share_magnitude[j] = mag_vals[lc_i*num_pts_max + j];
        /*# if __CUDA_ARCH__>=200
        if ((freqs[i] < 2.314814814814814839e-07) && (lc_i == 1)){
            printf("%.18e\n", freqs[i]);
            for (int jj=0; jj<=mag_bins; jj++) printf("%e, ", share_bins_mag[jj]);
            printf("\n\n");
        }


        #endif //*/

    ce_vals[(lc_i*num_pdots + pdot_i)*num_freqs + i] = ce(freqs[i], pdots[pdot_i], share_bins_phase, &mag_bin_inds[0],
                                                          &time_vals_share[0], num_pts_this_lc, mag_bins, phase_bins,
                                                          offset, lc_i, lc_start_time,
                                                          &temp_phase_prob[threadIdx.x*51], &overall_phase_prob[threadIdx.x*51], half_dbins);
  }
}
}
}

/*
int main(){

    int npoints=100;
    int phase_bins=15;
    int mag_bins=10;
    int num_freqs= 1 << 13;
    fod divisor = ((fod) num_freqs)/2.0;
    int num_lcs = (int) 1000000;

    fod* freqs = new fod[num_freqs];
    fod* ce_vals = new fod[num_freqs];

    fod *d_freqs, *d_ce_vals;

    cudaMalloc(&d_freqs, sizeof(fod)*num_freqs);
    cudaMalloc(&d_ce_vals, sizeof(fod)*num_freqs);

    for (int i=0; i<num_freqs; i++){
        freqs[i] = (i+1)*1./12./divisor;
    }

    cudaMemcpy(d_freqs, freqs, num_freqs*sizeof(fod), cudaMemcpyHostToDevice);

    fod* bins_mag = new fod[mag_bins + 1];
    fod* bins_phase = new fod[phase_bins + 1];
    fod *d_bins_mag, *d_bins_phase;

    cudaMalloc(&d_bins_mag, sizeof(fod)*(mag_bins + 1));
    cudaMalloc(&d_bins_phase, sizeof(fod)*(phase_bins + 1));

    fod* time_vals = new fod[npoints*num_lcs];
    fod* mag_vals = new fod[npoints*num_lcs];
    fod *d_t_vals, *d_magnitude;

    cudaMalloc(&d_t_vals, sizeof(fod)*npoints*num_lcs);
    cudaMalloc(&d_magnitude, sizeof(fod)*npoints*num_lcs);

    fod ovrl_mag = 2.12323;

    for (int i=0; i<=mag_bins; i++){
        bins_mag[i] = 2.0*i*ovrl_mag/((fod) mag_bins) - ovrl_mag;
    }

    for (int i=0; i<=phase_bins; i++){
        bins_phase[i] = (fod) i/((fod) phase_bins);
    }

    cudaMemcpy(d_bins_mag, bins_mag, (mag_bins + 1)*sizeof(fod), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bins_phase, bins_phase, (phase_bins + 1)*sizeof(fod), cudaMemcpyHostToDevice);


    std::default_random_engine generator;
    std::uniform_real_distribution<fod> distribution(0.0, 1000.0);

    for (int i=0; i< npoints*num_lcs; i++){
        time_vals[i] = distribution(generator);
        mag_vals[i] = ovrl_mag*sin(2.0*3.1415*0.083333*time_vals[i]); // period of 12 minutes
    }

    cudaMemcpy(d_t_vals, time_vals, npoints*num_lcs*sizeof(fod), cudaMemcpyHostToDevice);
    cudaMemcpy(d_magnitude, mag_vals, npoints*num_lcs*sizeof(fod), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int nblocks = (int)ceil((num_freqs + NUM_THREADS - 1)/NUM_THREADS);
    dim3 griddim(num_lcs, nblocks);
    cudaEventRecord(start);
    kernel<<<griddim, NUM_THREADS>>>(d_ce_vals, d_freqs,  num_freqs, d_bins_phase, d_bins_mag, d_t_vals, d_magnitude,  npoints,  mag_bins,  phase_bins, num_lcs);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    fod milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%e, %e\n", milliseconds/1.0e3, milliseconds/1.0e3/((fod) num_freqs*num_lcs));

    cudaMemcpy(ce_vals, d_ce_vals, num_freqs*sizeof(fod), cudaMemcpyDeviceToHost);

    printf("%d\n", nblocks);
    //for (int i=0; i<num_freqs; i++){
    //     if (i % (1 << 22) == 0) printf("%lf, %lf\n", 1./freqs[i], ce_vals[i]);
    //}

    delete[] freqs;
    delete[] ce_vals;
    delete[] time_vals;
    delete[] mag_vals;
    delete[] bins_mag;
    delete[] bins_phase;

    cudaFree(d_ce_vals);
    cudaFree(d_freqs);
    cudaFree(d_magnitude);
    cudaFree(d_t_vals);
    cudaFree(d_bins_mag);
    cudaFree(d_bins_phase);
}*/
