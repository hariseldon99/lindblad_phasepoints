/* Lorenzo's implementation of the BBGKY equations in the homogeneous case */

#include "lindblad_bbgky.h"

//Levi civita symbol
int
eps (int i, int j, int k)
{
  int result = 0;
  if ((i == 0) && (j == 1) && (k == 2))
    {
      result = 1;
    }
  if ((i == 1) && (j == 2) && (k == 0))
    {
      result = 1;
    }
  if ((i == 2) && (j == 0) && (k == 1))
    {
      result = 1;
    }

  if ((i == 2) && (j == 1) && (k == 0))
    {
      result = -1;
    }
  if ((i == 1) && (j == 0) && (k == 2))
    {
      result = -1;
    }
  if ((i == 0) && (j == 2) && (k == 1))
    {
      result = -1;
    }

  return result;
}


double kdel(int i, int j){
  double result;
  if(i==j)
    result = 1.0;
  else{
    result = 0.0;
  }
  return result ;
}

int
dsdgdt (double *wspace, double *s, double *deltamat, double *gammamat, double *dtkr ,
	double drv_amp, int latsize, double *dsdt)
{

  //Pointer to cmat
  double *cmat, *dcdt_mat;
  int m, n, b, g;		//xyz indices
  int i, j;			//lattice indices
  double rhs;

  cmat = &s[3 * latsize];
  dcdt_mat = &dsdt[3 * latsize];

  //Set the diagonals of cmats to 0
  for (m = 0; m < 3; m++)
    for (n = 0; n < 3; n++)
      for (i = 0; i < latsize; i++)
	cmat[((n + 3 * m) * latsize * latsize) + (i + latsize * i)] = 0.0;

  //Calculate the mean field contributions:
  double *mf_s, *mf_cmat;

  mf_s = &wspace[0];
  mf_cmat = &wspace[3 * latsize];

  //cblas_dsymm (CblasRowMajor, CblasRight, CblasUpper, 3, latsize, 1.0, hopmat,
	       //latsize, s, latsize, 0.0, mf_s, latsize);

  for (b = 0; b < 3; b++)
    for (g = 0; g < 3; g++)
      {

	//cblas_dsymm (CblasRowMajor, CblasLeft, CblasUpper, latsize, latsize,
	//	     1.0, hopmat, latsize,
	//	     &cmat[(g + 3 * b) * latsize * latsize], latsize, 0.0,
	//	     &mf_cmat[(g + 3 * b) * latsize * latsize], latsize);
      }

  //Update the spins in dsdt
  for (i = 0; i < latsize; i++)
    for (m = 0; m < 3; m++)
      {
	rhs = 0.0;
	for (g = 0; g < 3; g++)
	  {
	    for (b = 0; b < 3; b++)
	      {
	        rhs =  0.0;
	      }
	  }
	rhs = -s[(i+3*m)];  
	dsdt[i + latsize * m] = rhs;
      }

  //Update the correlations in dgdt 
  for (m = 0; m < 3; m++)
    for (n = m; n < 3; n++)
      for (i = 0; i < latsize; i++)
	for (j = 0; j < latsize; j++)
	  {
	    rhs = 0.0;
	    {
	      for (b = 0; b < 3; b++)
		{
		 rhs = 0.0;
		 }
		}
	    rhs = - cmat[((n + 3 * m) * latsize * latsize) +
		     (j + latsize * i)];
	    
	    dcdt_mat[((n + 3 * m) * latsize * latsize) +
		     (j + latsize * i)] = 2.0 * rhs;
	    dcdt_mat[((m + 3 * n) * latsize * latsize) +
		     (i + latsize * j)] = 2.0 * rhs;
	  }



  return 0;
}
