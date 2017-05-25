# lnr
Linear regression tools in python

---

This code incorporates various prescriptions for linear regression of measurements with uncertainties on both the dependent and independent variables which may be correlated. It also incorporates automatic conversions to logarithmic space in order to fit a power law, if so requested. Each function has a detailed help page but, briefly, the available functions are:

    lnr.bces(x1, x2, **kwargs)

Bivariate Correlated Errors and intrinsic Scatter (BCES, [Akritas & Bershady, 1996](http://adsabs.harvard.edu/abs/1996ApJ...470..706A)). Python code translated from [FORTRAN code](http://www.astro.wisc.edu/~mab/archive/stats/stats.html) by Christina Bird and Matthew Bershady.

    lnr.kelly(x1, x2, **kwargs)

Python wrapper around the [IDL Bayesian linear regression code](http://idlastro.gsfc.nasa.gov/ftp/pro/math/linmix_err.pro) by [Brandon Kelly (2007)](http://adsabs.harvard.edu/abs/2007ApJ...665.1489K), that accounts for correlated uncertainties in both variables and intrinsic scatter. Requires [`pidly`](https://github.com/anthonyjsmith/pIDLy) and an IDL license.

    lnr.mle(x1, x2, **kwargs)

Maximum likelihood estimator including intrinsic scatter.

--

There are additional, auxiliary functions:

    lnr.to_log(x, xerr)
    lnr.to_linear(logx, logxerr)

convert a given set of measurements and uncertainties between linear and log space.

    lnr.plot(t, a, b, **kwargs)

Used to plot the best-fit linear relation along with a shaded region representing the best-fit uncertainties. Since this function can take a matplotlib axis as a keyword argument, it can be easily embedded within a larger plotting function that includes, for instance, the data points from which the relation is derived.

----

## **Installation**

Clone this package with

    git clone https://github.com/cristobal-sifon/lnr.git

(you may also fork it to your own github account and then clone your fork). Then, simply type

    python setup.py install [--user]

where the `--user` flag is necessary if you do not have root privileges, unless your Python distribution was installed through Anaconda.

After this, just open a python terminal and type

    >>> from lnr import lnr

and all the modules described above will be available. Then type, e.g.,

    >>> help(lnr.bces)

to get more details.

----

(c) Cristóbal Sifón
Last modified 2017-05-25
