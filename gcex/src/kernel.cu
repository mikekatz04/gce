#include <math.h>
#include <random>
#include "global.h"

#define NUM_THREADS 256

__device__ fod ce (fod frequency, fod pdot, fod* phase_bin_edges,
                   int* mag_bin_inds, fod* time_vals, int npoints,
                   int mag_bins, int phase_bins, fod* mag_bin_vals,
                   int offset, int lc_i, fod lc_start_time){
    fod period = 1./frequency;
    fod folded_val = 0.0;
    int mag_ind_1 = -1, mag_ind_2=-1;
    int l = 1;
    fod sum_ij = 0.0;
    fod t_val = 0.0;
    fod total_points = 0.0;

    for (int i=0; i<mag_bins; i++){
        mag_bin_vals[offset + i] = 0;
    }

    fod current_phase_prob = 0.0;
    int temp_mag_points = 0;

    for (int j=0; j<phase_bins; j++){
        current_phase_prob = 0;
        /*# if __CUDA_ARCH__>=200
        if ((offset == 0)  && (lc_i == 0)){
            printf("(%d %d %d): %lf - %lf\n", lc_i, j, offset, phase_bin_edges[2*j], phase_bin_edges[2*j+1]);
        }
        #endif //*/
        for (int k=0; k<npoints; k++){
            t_val = time_vals[k] - lc_start_time;

            folded_val = fmodf(t_val-0.5*pdot*frequency*(t_val*t_val), period)*frequency; // between 0 and 1
            if (folded_val < 0) {
                folded_val = 1 + folded_val;
            }

            if ((folded_val >= phase_bin_edges[2*j]) && (folded_val < phase_bin_edges[2*j+1])){
                mag_ind_1 = mag_bin_inds[2*k];
                mag_bin_vals[offset + mag_ind_1] += 1.0;
                current_phase_prob += 1.0;
                total_points += 1.0;
                mag_ind_2 = mag_bin_inds[2*k + 1];
                if (mag_ind_2 != -1){
                    mag_bin_vals[offset + mag_ind_2] += 1.0;
                    current_phase_prob += 1.0;
                    total_points += 1.0;
                }
                //printf("%d, %lf, %lf\n", l, mag_bins[l-1], current_phase_prob);
            }
        }
        /*# if __CUDA_ARCH__>=200
        if ((offset == mag_bins*255)  && (lc_i == 0)){
            printf("(%d %d %d %.18e %e)\n", lc_i, j, offset, frequency, current_phase_prob);
            for (int jj=0; jj<mag_bins; jj++) printf("%e, ", mag_bin_vals[offset + jj]);
            printf("\n");
            for (int jj=0; jj<mag_bins+1; jj++) printf("%e, ", mag_bin_edges[jj]);
            printf("\n\n");
        }
        #endif //*/

        for (int i=0; i<mag_bins; i++){
            if ((current_phase_prob > 0.0) && (mag_bin_vals[offset + i] > 0.0)){
                //printf("%d, %lf\n", i,log(current_phase_prob/mag_bins[i]));
                sum_ij += (mag_bin_vals[offset + i])*log((current_phase_prob)/(mag_bin_vals[offset + i]));
            }
            mag_bin_vals[offset + i] = 0.0;

        }
    }
    /*# if __CUDA_ARCH__>=200
    if ((offset == mag_bins*255)  && (lc_i == 0)){
        printf("(%d %d %.18e CE: %.18e)\n", lc_i, offset, frequency, sum_ij);
    }
    #endif //*/

    return  sum_ij/((fod) total_points);
}

__global__ void kernel(fod* ce_vals, fod* freqs, int num_freqs, fod* pdots, int num_pdots, fod* phase_bin_edges, int* mag_bin_inds, fod* time_vals, int *num_pts_arr, int num_pts_max, int mag_bins, int phase_bins, int num_lcs, fod* min_light_curve_times){
    int i = blockIdx.y*blockDim.x + threadIdx.x;
    int lc_i = blockIdx.x;
    int pdot_i = blockIdx.z;


    for (int lc_i = blockIdx.x;
         lc_i < num_lcs;
         lc_i += gridDim.x) {

   for (int pdot_i = blockIdx.z;
        pdot_i < num_pdots;
        pdot_i += gridDim.z) {

    for (int i = blockIdx.y * blockDim.x + threadIdx.x;
         i < num_freqs;
         i += blockDim.x * gridDim.y) {

    int num_pts_this_lc = num_pts_arr[lc_i];
    fod lc_start_time = min_light_curve_times[lc_i];

    // TODO: make this adjustable: https://devblogs.nvidia.com/using-shared-memory-cuda-cc/  !!!!
    int offset = mag_bins*threadIdx.x;
    __shared__ fod share_mag_bin_vals[NUM_THREADS*10];
    extern __shared__ fod share_bins_phase[];
    //__shared__ fod share_t_vals[100];
    //__shared__ fod share_magnitude[100];

    if (threadIdx.x == 0){
        for (int j=0; j<=phase_bins; j++){
            share_bins_phase[2*j] = phase_bin_edges[2*j];
            share_bins_phase[2*j+1] = phase_bin_edges[2*j+1];
        }
        //for (int j=0; j<num_pts_this_lc; j++) share_t_vals[j] = time_vals[lc_i*num_pts_max + j];
        //for (int j=0; j<num_pts_this_lc; j++) share_magnitude[j] = mag_vals[lc_i*num_pts_max + j];
        /*# if __CUDA_ARCH__>=200
        if ((freqs[i] < 2.314814814814814839e-07) && (lc_i == 1)){
            printf("%.18e\n", freqs[i]);
            for (int jj=0; jj<=mag_bins; jj++) printf("%e, ", share_bins_mag[jj]);
            printf("\n\n");
        }


        #endif //*/
    }
    __syncthreads();

    ce_vals[(lc_i*num_pdots + pdot_i)*num_freqs + i] = ce(freqs[i], pdots[pdot_i], share_bins_phase, &(mag_bin_inds[lc_i*num_pts_max*2]), &(time_vals[lc_i*num_pts_max]), num_pts_this_lc, mag_bins, phase_bins, share_mag_bin_vals, offset, lc_i, lc_start_time);
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
