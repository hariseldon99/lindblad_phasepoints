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

  double *cmat, *dcdt_mat;
  int m, n, b, k;		//xyz indices
  int i, j;			//lattice indices
  double rhs, rhs_iter, prod;

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
  double *mf_sp, *mf_cmatp;	//Convolution with deltamat
  double *mf_sm, *mf_cmatm;	//Convolution with gammamat

  mf_sp = &wspace[0];
  mf_cmatp = &wspace[3 * latsize];

  mf_sm = &wspace[3 * latsize + 9 * latsize * latsize];
  mf_cmatm = &wspace[3 * latsize + 9 * latsize * latsize + 3 * latsize];

  cblas_dsymm (CblasRowMajor, CblasRight, CblasUpper, 3, latsize, 1.0,
	       deltamat, latsize, s, latsize, 0.0, mf_sp, latsize);

  cblas_dsymm (CblasRowMajor, CblasRight, CblasUpper, 3, latsize, 1.0,
	       gammamat, latsize, s, latsize, 0.0, mf_sm, latsize);

  for (b = 0; b < 3; b++)
    for (k = 0; k < 3; k++)
      {

	cblas_dsymm (CblasRowMajor, CblasLeft, CblasUpper, latsize, latsize,
		     1.0, deltamat, latsize,
		     &cmat[(k + 3 * b) * latsize * latsize], latsize, 0.0,
		     &mf_cmatp[(k + 3 * b) * latsize * latsize], latsize);

	cblas_dsymm (CblasRowMajor, CblasLeft, CblasUpper, latsize, latsize,
		     1.0, gammamat, latsize,
		     &cmat[(k + 3 * b) * latsize * latsize], latsize, 0.0,
		     &mf_cmatm[(k + 3 * b) * latsize * latsize], latsize);
      }

  //Update the spins in dsdt
  for (i = 0; i < latsize; i++)
    {
      cos_dtkr_i = cos (dtkr[i]);
      sin_dtkr_i = sin (dtkr[i]);
      for (m = 0; m < 3; m++)
	{
	  rhs_iter = 0.0;
	  rhs = 0.0;
	  for (n = 0; n < 3; n++)
	    {
	      rhs +=
		s[i + latsize * n] * (cos_dtkr_i * eps (0, n, m) +
				      sin_dtkr_i * eps (1, n, m));
	    }
	  rhs_iter += drv_amp * rhs;
	  rhs = 0.0;
	  for (n = 0; n < 3; n++)
	    {
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

	    }
	  rhs_iter += rhs;
	  rhs_iter -=
	    0.5 * s[i + latsize * m] * (1.0 + kdel (2, m)) + kdel (2, m);
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
	    if (i != j)
	      {
		cos_dtkr_j = cos (dtkr[j]);
		sin_dtkr_j = sin (dtkr[j]);
		//Line 1 in BBGKY Dynamics. See PDF in source tree
		rhs_iter =
		  -cmat[(n + 3 * m) * latsize * latsize +
			(j + latsize * i)] * (1.0 + 0.5 * (kdel (m, 2) +
							   kdel (n, 2)));

		//Inside these brackets lie codes for all the other lines       
		{
		  //RHS iterates over lines 2 and 3, the drive part 
		  rhs = 0.0;
		  for (b = 0; b < 3; b++)
		    {
		      rhs +=
			cmat[(n + 3 * b) * latsize * latsize +
			     (j + latsize * i)] * (cos_dtkr_i * eps (0, b,
								     m) +
						   sin_dtkr_i * eps (1, b,
								     m));
		      rhs +=
			cmat[(b + 3 * m) * latsize * latsize +
			     (j + latsize * i)] * (cos_dtkr_j * eps (0, b,
								     n) +
						   sin_dtkr_j * eps (1, b,
								     n));

		    }
		  rhs_iter += drv_amp * rhs;

		  //RHS iterates over lines 4 and 5
		  rhs = 0.0;
		  for (b = 0; b < 3; b++)
		    {
		      prod = cmat[((n + 3 * b) * latsize * latsize) +
				  (j + latsize * i)];

		      rhs +=
			((mf_sp[i + latsize * 0] -
			  0.5 * mf_sm[i + latsize * 1]) * eps (0, b,
							       m) + (mf_sp[i +
									   latsize
									   *
									   1]
								     +
								     0.5 *
								     mf_sm[i +
									   latsize
									   *
									   0])
			 * eps (1, b, m)) * prod;
		      rhs -=
			((s[j + latsize * 0] * deltamat[j + latsize * i] -
			  0.5 * s[j + latsize * 1] * gammamat[j +
							      latsize * i]) *
			 eps (0, b,
			      m) + (s[j + latsize * 1] * deltamat[j +
								  latsize *
								  i] +
				    0.5 * s[j + latsize * 0] * gammamat[j +
									latsize
									*
									i]) *
			 eps (1, b, m)) * prod;
		    }
		  rhs_iter += rhs;
		  //RHS iterates over lines 6 and 7
		  rhs = 0.0;
		  for (b = 0; b < 3; b++)
		    {
		      prod = cmat[((b + 3 * m) * latsize * latsize) +
				  (j + latsize * i)];
		      rhs +=
			((mf_sp[j + latsize * 0] -
			  0.5 * mf_sm[j + latsize * 1]) * eps (0, b,
							       n) + (mf_sp[j +
									   latsize
									   *
									   1]
								     +
								     0.5 *
								     mf_sm[j +
									   latsize
									   *
									   0])
			 * eps (1, b, n)) * prod;
		      rhs -=
			((s[i + latsize * 0] * deltamat[j + latsize * i] -
			  0.5 * s[i + latsize * 1] * gammamat[j +
							      latsize * i]) *
			 eps (0, b,
			      n) + (s[i + latsize * 1] * deltamat[j +
								  latsize *
								  i] +
				    0.5 * s[i + latsize * 0] * gammamat[j +
									latsize
									*
									i]) *
			 eps (1, b, n)) * prod;
		    }
		  rhs_iter += rhs;
		  //RHS iterates over lines 8 and 9
		  rhs = 0.0;
		  for (b = 0; b < 3; b++)
		    {
		      prod = s[i + latsize * b];
		      rhs +=
			((mf_cmatp
			  [((n + 3 * 0) * latsize * latsize) +
			   (j + latsize * i)] -
			  0.5 * mf_cmatm[((n + 3 * 1) * latsize * latsize) +
					 (j + latsize * i)]) * eps (0, b,
								    m) +
			 (mf_cmatp
			  [((n + 3 * 1) * latsize * latsize) +
			   (j + latsize * i)] +
			  0.5 * mf_cmatm[((n + 3 * 0) * latsize * latsize) +
					 (j + latsize * i)]) * eps (1, b,
								    m)) *
			prod;
		    }
		  rhs_iter += rhs;
		  //RHS iterates over lines 10 and 11
		  rhs = 0.0;
		  for (b = 0; b < 3; b++)
		    {
		      prod = s[j + latsize * b];

		      rhs +=
			((mf_cmatp
			  [((m + 3 * 0) * latsize * latsize) +
			   (i + latsize * j)] -
			  0.5 * mf_cmatm[((m + 3 * 1) * latsize * latsize) +
					 (i + latsize * j)]) * eps (0, b,
								    n) +
			 (mf_cmatp
			  [((m + 3 * 1) * latsize * latsize) +
			   (i + latsize * j)] +
			  0.5 * mf_cmatm[((m + 3 * 0) * latsize * latsize) +
					 (i + latsize * j)]) * eps (1, b,
								    n)) *
			prod;

		    }
		  rhs_iter += rhs;
		  //RHS iterates over line 12 
		  rhs = 0.0;
		  for (b = 0; b < 3; b++)
		    {
		      prod = s[j + latsize * b];
		      rhs +=
			(kdel (m, 0) * deltamat[j + latsize * i] -
			 0.5 * kdel (m,
				     1) * gammamat[j + latsize * i]) * eps (0,
									    b,
									    n)
			* prod;
		      rhs +=
			(kdel (m, 1) * deltamat[j + latsize * i] +
			 0.5 * kdel (m,
				     0) * gammamat[j + latsize * i]) * eps (1,
									    b,
									    n)
			* prod;
		    }
		  rhs_iter += rhs;
		  //RHS iterates over line 13 
		  rhs = 0.0;
		  for (b = 0; b < 3; b++)
		    {
		      prod = s[i + latsize * b];
		      rhs +=
			(kdel (n, 0) * deltamat[j + latsize * i] -
			 0.5 * kdel (n,
				     1) * gammamat[j + latsize * i]) * eps (0,
									    b,
									    m)
			* prod;
		      rhs +=
			(kdel (n, 1) * deltamat[j + latsize * i] +
			 0.5 * kdel (n,
				     0) * gammamat[j + latsize * i]) * eps (1,
									    b,
									    m)
			* prod;
		    }
		  rhs_iter += rhs;
		  //RHS iterates over line 14
		  rhs = 0.0;
		  for (b = 0; b < 3; b++)
		    {
		      rhs +=
			s[j +
			  latsize * n] *
			((cmat
			  [(0 + 3 * b) * latsize * latsize +
			   (j + latsize * i)] + s[i + latsize * b] * s[j +
								       latsize
								       * 0]) *
			 (deltamat[j + latsize * i] * eps (0, b, m) +
			  0.5 * gammamat[j + latsize * i] * eps (1, b,
								 m)) +
			 (cmat
			  [(1 + 3 * b) * latsize * latsize +
			   (j + latsize * i)] + s[i + latsize * b] * s[j +
								       latsize
								       * 1]) *
			 (deltamat[j + latsize * i] * eps (1, b, m) -
			  0.5 * gammamat[j + latsize * i] * eps (0, b, m)));
		    }
		  rhs_iter -= rhs;
		  //RHS iterates over line 15
		  rhs = 0.0;
		  for (b = 0; b < 3; b++)
		    {
		      rhs +=
			s[i +
			  latsize * m] *
			((cmat
			  [(b + 3 * 0) * latsize * latsize +
			   (j + latsize * i)] + s[i + latsize * 0] * s[j +
								       latsize
								       * b]) *
			 (deltamat[j + latsize * i] * eps (0, b, n) +
			  0.5 * gammamat[j + latsize * i] * eps (1, b,
								 n)) +
			 (cmat
			  [(b + 3 * 1) * latsize * latsize +
			   (j + latsize * i)] + s[i + latsize * 1] * s[j +
								       latsize
								       * b]) *
			 (deltamat[j + latsize * i] * eps (1, b, n) -
			  0.5 * gammamat[j + latsize * i] * eps (0, b, n)));
		    }
		  rhs_iter -= rhs;
		  //RHS iterates over line 16
		  rhs = 0.0;
		  for (b = 0; b < 3; b++)
		    for (k = 0; k < 3; k++)
		      {
			rhs += (cmat[((k + 3 * b) * latsize * latsize) +
				     (j + latsize * i)] + s[i +
							    latsize * b] *
				s[j + latsize * k]) * (eps (0, b, m) * eps (k,
									    0,
									    n)
						       + eps (1, b,
							      m) * eps (k, 1,
									n));
		      }
		  rhs_iter -= gammamat[j + latsize * i] * rhs;
		}

		dcdt_mat[((n + 3 * m) * latsize * latsize) +
			 (j + latsize * i)] = rhs_iter;
		dcdt_mat[((m + 3 * n) * latsize * latsize) +
			 (i + latsize * j)] = rhs_iter;
	      }
	}

  //This writes to the python stdout. Use for debugging
  //PySys_WriteStdout(" %lf", -rhs);   

  //Set the diagonals of dcdt to 0
  for (m = 0; m < 3; m++)
    for (n = 0; n < 3; n++)
      for (i = 0; i < latsize; i++)
	dcdt_mat[((n + 3 * m) * latsize * latsize) + (i + latsize * i)] = 0.0;

  return 0;
}
