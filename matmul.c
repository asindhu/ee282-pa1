// This is the matrix multiply kernel you are to replace.
#include <math.h>
#include <stdio.h>

void matmul (int N, const double* A, const double* B, double* C) {
  int i, j, k;
	int bj_s, bj_l, bk_s, bk_l, bi_l;
	int kmax_s, kmax_l, jmax_s, jmax_l, imax_l;
	
	/* BLOCK SIZES */
	int bsize_l = 128;			// Large block size (L2 cache)
	int bsize_s = 32;				// Small block size (L1 cache)
	
	/* If the size is less than 32, there is no benefit from blocking
		 (i.e. no cache misses) – for now, just use naive implementation. */
	if (N < 32) {
		
		for (i = 0; i < N; i++)
	    for (j = 0; j < N; j++)
	      for (k = 0; k < N; k++)
	        C[i*N + j] += A[i*N+k]*B[k*N+j];
	
	} else {
		
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
		}
		
	}
	
}