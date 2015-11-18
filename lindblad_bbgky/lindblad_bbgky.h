#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <cblas.h>

/* Functions */
int eps (int i, int j, int k);
double kdel (int i, int j);
int
dsdgdt (double *wspace, double *s, double *deltamat, double *gammamat,
	double *dtkr, double drv_amp, int latsize, double *dsdt);
