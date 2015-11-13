#include <Python.h>
#include <numpy/arrayobject.h>
#include "lindblad_bbgky.h"


static PyObject *
wrap_bbgky (PyObject * self, PyObject * args)
{
  PyObject *arg0 = NULL;
  PyObject *arg1 = NULL, *arg2 = NULL, *arg3 = NULL, *arg4 = NULL;
  PyObject *out = NULL;

  PyObject *workspace = NULL;
  PyObject *s = NULL, *deltamat = NULL, *gammamat = NULL, *dtkr = NULL;
  PyObject *dsdt = NULL;
  double drv_amp;
  int latsize, ret;

  if (!PyArg_ParseTuple
      (args, "OOOOOdiO!", &arg0, &arg1, &arg2, &arg3, &arg4, &drv_amp, &latsize,
       &PyArray_Type, &out))
    return NULL;

  workspace = PyArray_FROM_OTF (arg0, NPY_DOUBLE, NPY_IN_ARRAY);
  if (workspace == NULL)
    return NULL;
  s = PyArray_FROM_OTF (arg1, NPY_DOUBLE, NPY_IN_ARRAY);
  if (s == NULL)
    return NULL;
  if (PyArray_NDIM (s) != 1)
    goto fail;
  deltamat = PyArray_FROM_OTF (arg2, NPY_DOUBLE, NPY_IN_ARRAY);
  if ((deltamat == NULL) || (PyArray_NDIM (deltamat) != 1))
    goto fail;
  gammamat = PyArray_FROM_OTF (arg3, NPY_DOUBLE, NPY_IN_ARRAY);
  if ((gammamat == NULL) || (PyArray_NDIM (gammamat) != 1))
    goto fail;
  dtkr = PyArray_FROM_OTF (arg4, NPY_DOUBLE, NPY_IN_ARRAY);
  if ((dtkr == NULL) || (PyArray_NDIM (dtkr) != 1))
    goto fail;

  dsdt = PyArray_FROM_OTF (out, NPY_DOUBLE, NPY_INOUT_ARRAY);
  if ((dsdt == NULL) || (PyArray_NDIM (dsdt) != 1))
    goto fail;

  /* code that makes use of arguments */
  /* You will probably need at least
     nd = PyArray_NDIM(<..>)    -- number of dimensions
     d  ims = PyArray_DIMS(<..>)  -- npy_intp array of length nd
     showing length in each dim. */

  double *wspace_ptr;
  double *s_ptr, *deltamat_ptr, *gammamat_ptr, *dtkr_ptr, *dsdt_ptr;

  wspace_ptr = (double *) PyArray_DATA (workspace);

  s_ptr = (double *) PyArray_DATA (s);
  deltamat_ptr = (double *) PyArray_DATA (deltamat);
  gammamat_ptr = (double *) PyArray_DATA (gammamat);
  dtkr_ptr = (double *) PyArray_DATA (dtkr);
  dsdt_ptr = (double *) PyArray_DATA (dsdt);

  ret =
    dsdgdt ((double *) wspace_ptr, (double *) s_ptr, (double *) deltamat_ptr,
	    (double *) gammamat_ptr, (double *) dtkr_ptr, drv_amp, latsize,
	    (double *) dsdt_ptr);
  if (ret != 0)
    goto fail;

  Py_DECREF (workspace);
  Py_DECREF (s);
  Py_DECREF (deltamat);
  Py_DECREF (gammamat);
  Py_DECREF (dtkr);
  Py_DECREF (dsdt);
  Py_INCREF (Py_None);
  return Py_None;

fail:
  Py_XDECREF (workspace);
  Py_XDECREF (s);
  Py_XDECREF (deltamat);
  Py_XDECREF (gammamat);
  Py_XDECREF (dtkr);
  Py_XDECREF (dsdt);
  return NULL;
}

static PyMethodDef ModuleMethods[] = {
  {"bbgky", wrap_bbgky, METH_VARARGS | METH_KEYWORDS,
   "bbgky(s, dmat, gmat, dtkr, drv_amp, N, dsdt)\n\\n\
C code with cblas dependency that optimally computes the RHS of the bbgky dynamics. \n Call this function from python as lindblad_bbgky.bbgky(args)\n Arguments in the following order. All are either ints, doubles or 1d numpy arrays:n w\t-\t Workspace that is a Numpy array of minimum size 3*N+9*N*N\n s\t-\tNumpy array of all spins sx, sy, sz (vecs of size N) and correlations matrices (size N X N)\n \t\tflattened as [sx, sy, sz, gxx, gxy, gxz, gyx, gyy, gyz, gzx, gzy, gzz]),\n dmat\t-\t delta (cosine) matrix (NXN), flattened to 1d array,\n gmat\t-\t gamma (sine) matrix (NXN), flattened to 1d array,\n dtkr\t-\t1d array dtkr_i = Delta t + (k.r)_i,\n N\t-\tLattice size,\n\t-\t dsdt\t-\tOutput numpy array (same structure as s) "},
  {NULL, NULL, 0, NULL},
};

PyMODINIT_FUNC
initlindblad_bbgky (void)
{
  (void) Py_InitModule ("lindblad_bbgky", ModuleMethods);
  import_array ();
}
