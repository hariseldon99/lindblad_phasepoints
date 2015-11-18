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


double
kdel (int i, int j)
{
  double result;
  if (i == j)
    result = 1.0;
  else
    {
      result = 0.0;
    }
  return result;
}

int
dsdgdt (double *wspace, double *s, double *deltamat, double *gammamat,
	double *dtkr, double drv_amp, int latsize, double *dsdt)
{

  //Pointer to cmat
  double *cmat, *dcdt_mat;
  int m, n, b, g;		//xyz indices
  int i, j;			//lattice indices
  double rhs, rhs_iter;

  cmat = &s[3 * latsize];
  dcdt_mat = &dsdt[3 * latsize];

  //Time-dependent amplitudes
  double cos_dtkr_i, sin_dtkr_i;
  double cos_dtkr_j, sin_dtkr_j;

  //Set the diagonals of cmats to 0
  for (m = 0; m < 3; m++)
    for (n = 0; n < 3; n++)
      for (i = 0; i < latsize; i++)
	cmat[((n + 3 * m) * latsize * latsize) + (i + latsize * i)] = 0.0;

  //Calculate the mean field contributions:
  double *mf_sp, *mf_cmatp;	//Colvolution with deltamat
  double *mf_sm, *mf_cmatm;	//Colvolution with gammamat

  mf_sp = &wspace[0];
  mf_cmatp = &wspace[3 * latsize];

  mf_sm = &wspace[3 * latsize + 9 * latsize * latsize];
  mf_cmatm = &wspace[3 * latsize + 9 * latsize * latsize + 3 * latsize];

  cblas_dsymm (CblasRowMajor, CblasRight, CblasUpper, 3, latsize, 1.0,
	       deltamat, latsize, s, latsize, 0.0, mf_sp, latsize);
  cblas_dsymm (CblasRowMajor, CblasRight, CblasUpper, 3, latsize, 1.0,
	       gammamat, latsize, s, latsize, 0.0, mf_sm, latsize);

  for (b = 0; b < 3; b++)
    for (g = 0; g < 3; g++)
      {

	cblas_dsymm (CblasRowMajor, CblasLeft, CblasUpper, latsize, latsize,
		     1.0, deltamat, latsize,
		     &cmat[(g + 3 * b) * latsize * latsize], latsize, 0.0,
		     &mf_cmatp[(g + 3 * b) * latsize * latsize], latsize);
	cblas_dsymm (CblasRowMajor, CblasLeft, CblasUpper, latsize, latsize,
		     1.0, gammamat, latsize,
		     &cmat[(g + 3 * b) * latsize * latsize], latsize, 0.0,
		     &mf_cmatm[(g + 3 * b) * latsize * latsize], latsize);
      }

  //Update the spins in dsdt
  for (i = 0; i < latsize; i++)
    {
      cos_dtkr_i = cos (dtkr[i]);
      sin_dtkr_i = sin (dtkr[i]);
      for (m = 0; m < 3; m++)
	{
	  rhs_iter = 0.0;
	  for (n = 0; n < 3; n++)
	    {
	      rhs =
		drv_amp * cmat[i +
			       latsize * n] * (cos_dtkr_i * eps (0, n,
								 m) +
					       sin_dtkr_i * eps (1, n, m));
	      rhs +=
		eps (0, n,
		     m) * (s[i + latsize * n] * mf_sp[i + latsize * 0] +
			   mf_cmatp[((0 + 3 * n) * latsize * latsize) +
				    (i + latsize * i)]);
	      rhs -=
		0.5 * eps (0, n,
			   m) * (s[i + latsize * n] * mf_sm[i + latsize * 1] +
				 mf_cmatm[((1 + 3 * n) * latsize * latsize) +
					  (i + latsize * i)]);
	      rhs +=
		eps (1, n,
		     m) * (s[i + latsize * n] * mf_sp[i + latsize * 1] +
			   mf_cmatp[((1 + 3 * n) * latsize * latsize) +
				    (i + latsize * i)]);
	      rhs +=
		0.5 * eps (1, n,
			   m) * (s[i + latsize * n] * mf_sm[i + latsize * 0] +
				 mf_cmatm[((0 + 3 * n) * latsize * latsize) +
					  (i + latsize * i)]);

	      rhs_iter += rhs;
	    }
	  rhs_iter -=
	    0.5 * cmat[i + latsize * m] * ((1.0 + kdel (2, m)) + kdel (2, m));
	  dsdt[i + latsize * m] = rhs_iter;
	}
    }

  //Update the correlations in dgdt
  for (m = 0; m < 3; m++)
    for (n = m; n < 3; n++)
      for (i = 0; i < latsize; i++)
	{
	  cos_dtkr_i = cos (dtkr[i]);
	  sin_dtkr_i = sin (dtkr[i]);
	  for (j = 0; j < latsize; j++)
	    {
	      cos_dtkr_j = cos (dtkr[j]);
	      sin_dtkr_j = sin (dtkr[j]);
	      rhs_iter = 0.0;
	      {
		for (b = 0; b < 3; b++)
		  {
		    rhs = 0.0;
		  }
	      }

	      dcdt_mat[((n + 3 * m) * latsize * latsize) +
		       (j + latsize * i)] = rhs_iter;
	      dcdt_mat[((m + 3 * n) * latsize * latsize) +
		       (i + latsize * j)] = rhs_iter;
	    }
	}



  //Set the diagonals of dcdt to 0
  for (m = 0; m < 3; m++)
    for (n = 0; n < 3; n++)
      for (i = 0; i < latsize; i++)
	dcdt_mat[((n + 3 * m) * latsize * latsize) + (i + latsize * i)] = 0.0;


  return 0;
}
