#include <getopt.h>
#include <strings.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <unistd.h>
#include <signal.h>

#ifdef PAPI
#include <papi.h>
#endif

#include "utils.h"

// matmul() can be found in matmul.c.
void matmul (int i_matdim, const double* pd_A, const double* pd_B, double* pd_C);

// For simplicity, test sizes are restricted to be power of 2.
//int test_sizes[] = { 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}; 
int test_sizes[] = { 4 }; 


#ifdef PAPI
// You may replace these events with others of your choice. Refer to the PAPI documentation.

#define NUM_EVENTS 4
static char *event_names[] = { "Tot. Instr.", "Tot. Cycles", "L1 D-Miss", "L2 Miss", 0 };
int               events[] = {  PAPI_TOT_INS, PAPI_TOT_CYC, PAPI_L1_DCM, PAPI_L2_TCM, 0 };

#endif

volatile int expired = 0;

#define MAX_ERROR     (2.0)
#define CALC_ITERS(n) (10 + 1e8 / CUBE ((long long) n))
#define NUM_TESTS     (sizeof(test_sizes) / sizeof(int))

void alarm_handler(int signum) {
  expired = 1;
}

// DO NOT MODIFY THIS FUNCTION! This is to check your matmul() for CORRECTNESS.
void naive_matmul(int n, const double *A, const double *B, double *C) {
  int i, j, k;

  for (i = 0; i < n; i++)
    for (j = 0; j < n; j++)
      for (k = 0; k < n; k++)
        C[i*n + j] += A[i*n+k]*B[k*n+j];
}


void check_correct () {
  double *A, *B, *C, *cA, *cB, *cC;

  int maxnbytes = sizeof(double) * SQR(test_sizes[NUM_TESTS - 1]);
  int i;

  A  = (double*) malloc(maxnbytes);
  B  = (double*) malloc(maxnbytes);
  C  = (double*) malloc(maxnbytes);
  cA = (double*) malloc(maxnbytes);
  cB = (double*) malloc(maxnbytes);
  cC = (double*) malloc(maxnbytes);
 
  if (A  == NULL || B  == NULL || C  == NULL || 
      cA == NULL || cB == NULL || cC == NULL) {
    fprintf(stderr,"check_correct(): memory allocation failed\n");
    exit(-1);
  }
 
  fprintf(stderr, "Checking for correctness:"); 

  for (i = 0; i < NUM_TESTS; i++)  {
    int matdim = test_sizes[i];
    double err;
    int nbytes = sizeof(double) * SQR(matdim);

    fprintf(stderr, " %dx%d", matdim, matdim); 

    mat_init(A, matdim, matdim);
    mat_init(B, matdim, matdim);
    mat_init(C, matdim, matdim);

    bcopy((void*)A, (void*)cA, nbytes);
    bcopy((void*)B, (void*)cB, nbytes);
    bcopy((void*)C, (void*)cC, nbytes);

    naive_matmul(matdim, cA, cB, cC);
    matmul(matdim, A, B, C);

    if (bcmp((void*)A, (void*)cA, nbytes) != 0 ||
        bcmp((void*)B, (void*)cB, nbytes) != 0) {
      fprintf(stderr, "\nSource matrices have changed.\nFAILED.\n");
      exit(1);
    }

    if ((err = error(C, cC, matdim, matdim)) > MAX_ERROR) {
      fprintf(stderr,
              "\nCalculated error for test case %dx%d is %f > %f.\nFAILED.\n",
              matdim, matdim, err, MAX_ERROR);
      exit(1);
    }
  }

  fprintf(stderr, "\nPASSED.\n");

  free(A);
  free(B);
  free(C); 
  free(cA);
  free(cB);
  free(cC);
}

void measure_performance () {
  struct rusage ru_start, ru_end;

  double *A,  *B,  *C;
  double *oA, *oB, *oC;
  int i, j, k;
  int test;
  int papi = 0;

  long long values[4];

  printf("\n");
  printf("Each measurement is average per iteration. Runtime is given in milliseconds.\n");
  printf("MFLOPS is estimated assuming a naive matmul().\n");
  printf("\n");
  printf("Dim.   MFLOPS     Runtime   ");

#ifdef PAPI
  for (i = 0; events[i] != 0; i++)
    printf("%13s", event_names[i]);
#endif

  printf("\n");

  for (test = 0; test < NUM_TESTS; test++) {
    int matdim = test_sizes[test];
    int num_iters = (int)(CALC_ITERS(matdim));
    int iter;

    double mflops;
    double utime;

    // Allocate matrices. We use different matricies for each trial so
    // that the OS page mapping won't affect the results. rrand()
    // defeats the buddy allocator.
    A = oA = (double*) malloc ((SQR(matdim) + 1 + rrand(1,10)) * sizeof(double));
    B = oB = (double*) malloc ((SQR(matdim) + 1 + rrand(1,10)) * sizeof(double));
    C = oC = (double*) malloc ((SQR(matdim) + 1 + rrand(1,10)) * sizeof(double));

    if (A == NULL || B == NULL || C == NULL) {
      fprintf(stderr, "measure_performance(): memory allocation failed\n");
      exit(-1);
    }
  
    // Make sure the pointers are aligned.
    if (((unsigned long) A) & 0x8) A = (double*)(((unsigned long) A) + 0x8);
    if (((unsigned long) B) & 0x8) B = (double*)(((unsigned long) B) + 0x8);
    if (((unsigned long) C) & 0x8) C = (double*)(((unsigned long) C) + 0x8);

    // Fill matrices with random data.
    mat_init(A, matdim, matdim);
    mat_init(B, matdim, matdim);
    mat_init(C, matdim, matdim);

    // Mark the current time.
    getrusage(RUSAGE_SELF, &ru_start);

#ifdef PAPI     
    // Enable performance counters.
    if (PAPI_start_counters(events, NUM_EVENTS) == PAPI_OK) {
      papi = 1;
    } else {
      fprintf(stderr, "measure_performance(): PAPI_start_counters() failed\n");
      //      exit(-1);
    }
#endif

    expired = 0;
    alarm(10); // 10 second time limit for each size (not strictly enforced)

    // Iteratively run matmul(). Note: The output will accumulate in *C.
    for (iter = 0; iter < num_iters && !expired; iter++) {
      matmul(matdim, A, B, C);
    }

#ifdef PAPI
    // Stop and read performance counters.
    if (papi) {
      if (PAPI_stop_counters(values, 4) != PAPI_OK) {
        fprintf(stderr, "measure_performance(): PAPI_stop_counters() failed\n");
        exit(-1);
      }
    }
#endif

    // Mark the current time.
    getrusage(RUSAGE_SELF, &ru_end);

    // Calculate the measured user time.
    utime = timeval_diff(ru_start.ru_utime, ru_end.ru_utime);

    // A silly estimate of MFLOPS..
    mflops = 2.0 * CUBE((long long) matdim) * iter * 1e-6 / utime;

    printf("%4d %8.3f % 11.4f   ", matdim, mflops, utime / iter * 1e3);

#ifdef PAPI
    if (papi) {
      for (i = 0; i < NUM_EVENTS; i++)
        printf("% 13.0f", ((double) values[i]) / iter);
    }
#endif

    printf("\n");

    free(oA); free(oB); free(oC);
  }
}

int main(int argc, char ** argv) {
  int check = 0, measure = 0, help = 0;

  while (1) {
    int c = getopt(argc, argv, "cph");

    if (c == -1) break;

    switch (c) {
    case 'c': check = 1;   break;
    case 'p': measure = 1; break;
    case 'h': help = 1;    break;
    default: break;
    }
  }

  if (help) {
    char help_text[] =
      "EE282 Programming Assignment 1 -- Matrix Multiplication\n\
\n\
Usage: matmul [-c] [-p] [-h]\n\
           -c    Check matmul() for correctness.\n\
           -p    Measure matmul() performance. (default)\n\
           -h    Display this help text.\n\
\n\
";
    printf(help_text);
    exit(0);
  }

  signal(SIGALRM, alarm_handler);

  rseed();

  if (check) check_correct();

  if (!check || measure) measure_performance();

  return 0;
}
