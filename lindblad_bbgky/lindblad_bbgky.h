#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <limits.h>
#include <string.h>
#include <cblas.h>

/* Functions */
int eps (int i, int j, int k);
double kdel (int i, int j);

int
dsdgdt_rwa (double *wspace, double *s, double *deltamat, double *gammamat,
	    double *dtkr, double drv_amp, double drv_freq, int latsize,
	    double *dsdt);

int
dsdgdt (double *wspace, double *s, double *deltamat, double *gammamat,
	double *dtkr, double drv_amp, int latsize, double *dsdt);
