# lnr
Linear regression tools in python

---

This code incorporates various prescriptions for linear regression of measurements with uncertainties
on both the dependent and independent variables which may be correlated. Each function has a detailed
help page but, briefly, the available functions are:

    lnr.bces(x1, x2, **kwargs)

  Bivariate Correlated Errors and intrinsic Scatter (BCES,
  [Akritas & Bershady, 1996](http://adsabs.harvard.edu/abs/1996ApJ...470..706A)). Python code
  translated from [FORTRAN code](http://www.astro.wisc.edu/~mab/archive/stats/stats.html)
  by Christina Bird and Matthew Bershady.

    lnr.kelly(x1, x2, **kwargs)

  Python wrapper around the [IDL Bayesian linear regression
  code](http://idlastro.gsfc.nasa.gov/ftp/pro/math/linmix_err.pro)
  by [Brandon Kelly (2007)](http://adsabs.harvard.edu/abs/2007ApJ...665.1489K), that accounts for
  correlated uncertainties in both variables and intrinsic scatter. Requires
  [`pidly`](https://github.com/anthonyjsmith/pIDLy) and an IDL license.

    lnr.mle(x1, x2, **kwargs)

  Maximum likelihood estimator including intrinsic scatter.

--

Additionally, the function `lnr.plot(t, a, b, **kwargs)` can be used to plot the best-fit linear
relation along with a shaded region representing the best-fit uncertainties. Since this function
can take a matplotlib axis as a keyword argument, it can be easily embedded within a larger
plotting function that includes, for instance, the data points from which the relation is derived.

----

(c) Cristóbal Sifón
Last modified 2016-02-13
