// This is the matrix multiply kernel you are to replace.
#include <math.h>
#include <stdio.h>
#include <xmmintrin.h>


/* single-level blocking @ bsize N/2 with and without prefetch

before vector

	Dim.   MFLOPS     Runtime     Tot. Instr.  Tot. Cycles    L1 D-Miss      L2 Miss
	   2 1538.463      0.0000              68           26            0            0
	   4  800.005      0.0002             961          376            0            0
	   8 1538.536      0.0007            3881         1442            0            0
	  16 2501.018      0.0033           19737         7672            0            0
	  32 1432.898      0.0457          298147       107083           12            0
	  64 1576.897      0.3325         2164768       782211         3163            0
	 128 1593.836      2.6316        17316634      6444034       150468            4
	 256 1360.315     24.6667       138529423     57054500      3089164          504
	 512 1335.500    201.0000      1108223827    468773993     25174242       256620
	1024 1217.855   1763.3333      8865749211   4102503239    202332856     12805109

after vector

Dim.   MFLOPS     Runtime     Tot. Instr.  Tot. Cycles    L1 D-Miss      L2 Miss
   2 1538.463      0.0000              65           26            0            0
   4  769.236      0.0002             959          385            0            0
   8 1666.748      0.0006            3879         1446            0            0
  16 2501.018      0.0033           19735         7475            0            0
  32 1823.688      0.0359          208972        92651           76            0
  64 1576.897      0.3325         1548234       789416         3680            0
 128 1707.681      2.4561        12384810      5842782       155207           17
 256 1480.343     22.6667        99075967     51804038      3147948         2624
 512 1451.002    185.0000       792599598    433785461     25089467       405222
1024 1322.110   1624.2857      6340765554   3775259579    201410294     12382560

*/

void matmul (int N, const double* A, const double* B, double* C) {
	
	int i, j, k;
	
	/*
	double d1 = 4.0;
	double d2 = 3.0;
	
	__m128d a = _mm_set_sd(d1);
	__m128d b = _mm_set_sd(d2);
	
	__m128d res = _mm_mul_pd(a, b);
	
	double r;
	_mm_store_sd(&r, res);
	
	*/
	
	if (N == 2) {		
		C[0] += (A[0] * B[0] + A[1] * B[2]);
		C[1] += (A[0] * B[1] + A[1] * B[3]);
		C[2] += (A[2] * B[0] + A[3] * B[2]);
		C[3] += (A[2] * B[1] + A[3] * B[3]);
	}
	
	else if (N == 4) {
	
		C[0] += (A[0] * B[0] + A[1] * B[4] + A[2] * B[8] + A[3] * B[12]);
		C[1] += (A[0] * B[1] + A[1] * B[5] + A[2] * B[9] + A[3] * B[13]);
		C[2] += (A[0] * B[2] + A[1] * B[6] + A[2] * B[10] + A[3] * B[14]);
		C[3] += (A[0] * B[3] + A[1] * B[7] + A[2] * B[11] + A[3] * B[15]);

		C[4] += (A[4] * B[0] + A[5] * B[4] + A[6] * B[8] + A[7] * B[12]);
		C[5] += (A[4] * B[1] + A[5] * B[5] + A[6] * B[9] + A[7] * B[13]);
		C[6] += (A[4] * B[2] + A[5] * B[6] + A[6] * B[10] + A[7] * B[14]);
		C[7] += (A[4] * B[3] + A[5] * B[7] + A[6] * B[11] + A[7] * B[15]);
		
		C[8] += (A[8] * B[0] + A[9] * B[4] + A[10] * B[8] + A[11] * B[12]);
		C[9] += (A[8] * B[1] + A[9] * B[5] + A[10] * B[9] + A[11] * B[13]);
		C[10] += (A[8] * B[2] + A[9] * B[6] + A[10] * B[10] + A[11] * B[14]);
		C[11] += (A[8] * B[3] + A[9] * B[7] + A[10] * B[11] + A[11] * B[15]);
		
		C[12] += (A[12] * B[0] + A[13] * B[4] + A[14] * B[8] + A[15] * B[12]);
		C[13] += (A[12] * B[1] + A[13] * B[5] + A[14] * B[9] + A[15] * B[13]);
		C[14] += (A[12] * B[2] + A[13] * B[6] + A[14] * B[10] + A[15] * B[14]);
		C[15] += (A[12] * B[3] + A[13] * B[7] + A[14] * B[11] + A[15] * B[15]);
		
	}
	
	else if (N < 64) {
		
		for (i = 0; i < N; i++) {
	    for (j = 0; j < N; j++) {	
				for (k = 0; k < N; k++) {
					//__builtin_prefetch (&A[i*N + k + 8], 0, 1);
					//__builtin_prefetch (&B[(k+4)*N + j], 0, 1);
	        C[i*N + j] += A[i*N+k]*B[k*N+j];
				}
			}
		}
	}
	
	else {
		
	/* One-level blocking variables */
	int kmax, jmax, kk, jj;
	int bsize = 32;
	if (N <= bsize) bsize = N/2;
	
	/* Two-level blocking variables */
	int bj_s, bj_l, bk_s, bk_l, bi_l;
	int kmax_s, kmax_l, jmax_s, jmax_l, imax_l;
	int bsize_l = 256;			// Large block size (L2 cache)
	if (N <= bsize_l) bsize_l = N/4;
	int bsize_s = 16;				// Small block size (L1 cache)
	if (N <= bsize) bsize = N/8;
	
		/* I-K-J blocking */
		
		double result;
		
		__m128d a, b1, b2, c1, c2;
		__m128d mulres1, mulres2, addres1, addres2;
		
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
							
							a = _mm_set_pd(A[i*N + k], A[i*N + k]);
							
							for (j = jj; j < jmax; j += 4) {
								
								b1 = _mm_set_pd(B[k*N + j], B[k*N + j + 1]);
								b2 = _mm_set_pd(B[k*N + j + 2], B[k*N + j + 3]);
								c1 = _mm_set_pd(C[i*N + j], C[i*N + j + 1]);
								c2 = _mm_set_pd(C[i*N + j + 2], C[i*N + j + 3]);
								
								mulres1 = _mm_mul_pd(a, b1);
								addres1 = _mm_add_pd(c1, mulres1);
								mulres2 = _mm_mul_pd(a, b2);
								addres2 = _mm_add_pd(c2, mulres2);
								
								_mm_storeh_pd(&C[i*N + j], addres1);
								_mm_storel_pd(&C[i*N + j + 1], addres1);
								_mm_storeh_pd(&C[i*N + j + 2], addres2);
								_mm_storel_pd(&C[i*N + j + 3], addres2);
								
							}
						}
						/*
						for (k = kk; k < kmax; k++) {							
							for (j = jj; j < jmax; j++) {
								C[i*N + j] += A[i*N + k] * B[k*N + j];
							}
						} */
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
								
								__builtin_prefetch (&A[(i+2)*N + bk_s], 0, 1);
								__builtin_prefetch (&C[(i+2)*N + bj_s], 1, 1);
								
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
