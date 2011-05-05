// This is the matrix multiply kernel you are to replace.
#include <math.h>
#include <stdio.h>


/* single-level blocking @ bsize N/2 with and without prefetch

NO PREFETCH

Dim.   MFLOPS     Runtime     Tot. Instr.  Tot. Cycles    L1 D-Miss      L2 Miss
   2  625.000      0.0000             110           60            0            0
   4  166.668      0.0008            2391         1793            0            0
   8  188.688      0.0054           18091        12513            0            0
  16  192.386      0.0426          141747        99184            0            0
  32  198.620      0.3300         1124740       768511           32            0
  64  199.026      2.6343         8997513      6154776         3416            2
 128  195.963     21.4035        71979389     49618748       178494          135
 256  195.084    172.0000       575833361    399952567      2539796        11211
 512  192.427   1395.0000      4606661371   3242082454     24518784       608854
1024  190.887  11250.0000     36853271726  26219937154    197983479     16155011
	


*/

void matmul (int N, const double* A, const double* B, double* C) {
	
	int i, j, k;
	
	/* One-level blocking variables */
	int kmax, jmax, kk, jj;
	int bsize = 32;
	if (N <= bsize) bsize = N/4;
	
	/* Two-level blocking variables */
	int bj_s, bj_l, bk_s, bk_l, bi_l;
	int kmax_s, kmax_l, jmax_s, jmax_l, imax_l;
	int bsize_l = 720;			// Large block size (L2 cache)
	if (N <= bsize_l) bsize_l = N/2;
	int bsize_s = 32;				// Small block size (L1 cache)
	if (N <= bsize) bsize = N/4;
	
	if (N == 2) {
		C[0] += (A[0] * B[0] + A[1] * B[2]);
		C[1] += (A[0] * B[1] + A[1] * B[3]);
		C[2] += (A[2] * B[0] + A[3] * B[2]);
		C[3] += (A[2] * B[1] + A[3] * B[3]);
	}
	
	
	else if (N < 32) {
		
		for (i = 0; i < N; i++)
	    for (j = 0; j < N; j++)
	      for (k = 0; k < N; k++)
	        C[i*N + j] += A[i*N+k]*B[k*N+j];
	
	}
	
	else {
		
		/* I-K-J blocking */
		
			for (kk = 0; kk < N; kk += bsize) {
				kmax = kk + bsize;
				if (kmax > N) kmax = N;

				for (jj = 0; jj < N; jj += bsize) {
					jmax = jj + bsize;
					if (jmax > N) jmax = N;

					for (i = 0; i < N; i++) {
						
						__builtin_prefetch (&A[(i+2)*N + kk], 0, 1);
						__builtin_prefetch (&C[(i+2)*N + jj], 1, 1);
						
						for (k = kk; k < kmax; k++) {							
							for (j = jj; j < jmax; j++) {
								C[i*N + j] += A[i*N + k] * B[k*N + j];
							}
						}
					}
				}
			}		
		
		
		/* Two-level blocking */
		/*
		for (bk_l = 0; bk_l < N; bk_l += bsize_l) {
			kmax_l = bk_l + bsize_l;
			if (kmax_l > N) kmax_l = N;
			
			for (bj_l = 0; bj_l < N; bj_l += bsize_l) {
				jmax_l = bj_l + bsize_l;
				if (jmax_l > N) jmax_l = N;
				
				for (bi_l = 0; bi_l < N; bi_l += bsize_l) {
					imax_l = bi_l + bsize_l;
					if (imax_l > N) imax_l = N;
					
					for (bk_s = bk_l; bk_s < kmax_l; bk_s += bsize_s) {
						kmax_s = bk_s + bsize_s;
						if (kmax_s > kmax_l) kmax_s = kmax_l;
	
						for (bj_s = bj_l; bj_s < jmax_l; bj_s += bsize_s) {
							jmax_s = bj_s + bsize_s;
							if (jmax_s > jmax_l) jmax_s = jmax_l;
	
							for (i = bi_l; i < imax_l; i++) {
								for (k = bk_s; k < kmax_s; k++) {
									for (j = bj_s; j < jmax_s; j++) {
										C[i*N + j] += A[i*N + k] * B[k*N + j];
									}
								}
							}
						}
					}
				}
			}
		}*/
		
	}
	
}