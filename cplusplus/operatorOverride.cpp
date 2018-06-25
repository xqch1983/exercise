#define A(i,j) a[ (j)*lda + (i) ]
#define B(i,j) b[ (j)*ldb + (i) ]
#define C(i,j) c[ (j)*ldc + (i) ]
/* Block sizes */
#define mc 256
#define kc 128
#define nb 1000
#define min( i, j ) ( (i)<(j) ? (i): (j) )

/* Routine for computing C = A * B + C */

void AddDot4x4(int, double *, int, double *, int, double *, int);
void PackMatrixA(int, double *, int, double *);
void PackMatrixB(int, double *, int, double *);
void InnerKernel(int, int, int, double *, int, double *, int, double *, int, int);

void MY_MMult(int m, int n, int k, double *a, int lda,
	double *b, int ldb,
	double *c, int ldc)
{
	int i, p, pb, ib;

	/* This time, we compute a mc x n block of C by a call to the InnerKernel */

	for (p = 0; p<k; p += kc) {
		pb = min(k - p, kc);
		for (i = 0; i<m; i += mc) {
			ib = min(m - i, mc);
			InnerKernel(ib, n, pb, &A(i, p), lda, &B(p, 0), ldb, &C(i, 0), ldc, i == 0);
		}
	}
}

void InnerKernel(int m, int n, int k, double *a, int lda,
	double *b, int ldb,
	double *c, int ldc, int first_time)
{
	int i, j;
	double
		packedA[m * k];
	static double
		packedB[kc*nb];    /* Note: using a static buffer is not thread safe... */

	for (j = 0; j<n; j += 4) {        /* Loop over the columns of C, unrolled by 4 */
		if (first_time)
			PackMatrixB(k, &B(0, j), ldb, &packedB[j*k]);
		for (i = 0; i<m; i += 4) {        /* Loop over the rows of C */
										  /* Update C( i,j ), C( i,j+1 ), C( i,j+2 ), and C( i,j+3 ) in
										  one routine (four inner products) */
			if (j == 0)
				PackMatrixA(k, &A(i, 0), lda, &packedA[i*k]);
			AddDot4x4(k, &packedA[i*k], 4, &packedB[j*k], k, &C(i, j), ldc);
		}
	}
}
void PackMatrixA(int k, double *a, int lda, double * a_to)
{
	int j;
	for (j = 0; j < k; j++)
	{
		double *a_ij_pntr = &A(0, j);
		*a_to = *a
			
			
			_ij_pntr;
		*(a_to + 1) = *£¨a_ij_pntr + 1);
		*(a_to + 2) = *(a_ij_pntr + 2);
		*(a_to + 3) = *(a_ij_pntr + 3);
		a_to += 4;
 	}
}