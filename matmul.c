// This is the matrix multiply kernel you are to replace.
#include <math.h>
#include <stdio.h>

void matmul (int N, const double* A, const double* B, double* C) {
  int kk, jj, i, j, k;
	int bsize = 64;
	int kmax, jmax;

	for (kk = 0; kk < N; kk += bsize) {
		for (jj = 0; jj < N; jj += bsize) {
			for (i = 0; i < N; i++) {
				kmax = kk + bsize;
				if (kmax > N) kmax = N;
				
				for (k = kk; k < kmax; k++) {
					jmax = jj + bsize;
					if (jmax > N) jmax = N;
					
					for (j = jj; j < jmax; j++) {
						C[i*N + j] += A[i*N + k] * B[k*N + j];
					}
				}
			}
		}
	}
	
}


/*
	ORIGINAL CODE:
	
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      for (k = 0; k < N; k++)
        C[i*N + j] += A[i*N+k]*B[k*N+j];

	PERFORMANCE RESULTS OF ORIGINAL CODE: 

	Dim.   MFLOPS     Runtime     Tot. Instr.  Tot. Cycles    L1 D-Miss      L2 Miss
	   2  125.786      0.0001             354          299            0            0
	   4  183.487      0.0007            2384         1616            0            0
	   8  194.184      0.0053           18084        12303            0            0
	  16  196.158      0.0418          141740        96548            0            0
	  32  196.672      0.3332         1123644       773485           39            0
	  64  188.070      2.7877         8950365      6484409        70549            0
	 128  168.363     24.9123        71451811     58015737      2124310           28
	 256  162.360    206.6667       571017540    480574057     16851267         3523
	 512  141.406   1898.3333      4565766927   4419674326    134860979        96558
	1024   28.614  75050.0000     36516676354 174576179219   1094538583   1706050215
	

*/