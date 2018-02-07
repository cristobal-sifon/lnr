# lnr
Linear regression tools in python

---

This code incorporates various prescriptions for linear regression of measurements with uncertainties on both the dependent and independent variables which may be correlated. It also incorporates automatic conversions to logarithmic space in order to fit a power law, if so requested. See [Hogg, Bovy & Lang (2010)](https://arxiv.org/abs/1008.4686) for considerations on fitting a line to data.

Each function has a detailed help page but, briefly, the available functions are:

    lnr.bces(x1, x2, **kwargs)

Bivariate Correlated Errors and intrinsic Scatter (BCES, [Akritas & Bershady, 1996](http://adsabs.harvard.edu/abs/1996ApJ...470..706A)). Python code translated from [FORTRAN code](http://www.astro.wisc.edu/~mab/archive/stats/stats.html) by Christina Bird and Matthew Bershady.

    lnr.kelly(x1, x2, **kwargs)

Python wrapper around the [IDL Bayesian linear regression code](http://idlastro.gsfc.nasa.gov/ftp/pro/math/linmix_err.pro) by [Brandon Kelly (2007)](http://adsabs.harvard.edu/abs/2007ApJ...665.1489K), that accounts for correlated uncertainties in both variables and intrinsic scatter. Requires [`pIDLy`](https://github.com/anthonyjsmith/pIDLy) and an IDL license.

    lnr.mle(x1, x2, **kwargs)

Maximum likelihood estimator including intrinsic scatter.

    lnr.mcmc(x1, x2, **kwargs)

Perform an MCMC analysis, accounting for intrinsic scatter and using appropriate priors for each parameter (see, e.g., [this post](http://dfm.io/posts/fitting-a-plane/), or [this one](http://jakevdp.github.io/blog/2014/06/14/frequentism-and-bayesianism-4-bayesian-in-python/#Prior-on-Slope-and-Intercept). The MCMC sampling is performed with [`emcee`](http://dfm.io/emcee/current/).

##

There are additional, auxiliary functions:

    lnr.to_log(x, **kwargs)
    lnr.to_linear(logx, **kwargs)

convert a given set of measurements and uncertainties between linear and log space, and

    lnr.plot(t, a, b, **kwargs)

can be used to plot the best-fit linear relation along with a shaded region representing the best-fit uncertainties. Since this function can take a matplotlib axis as a keyword argument, it can be easily embedded within a larger plotting function that includes, for instance, the data points from which the relation is derived.

----

## **Installation**

Clone this package with

    git clone https://github.com/cristobal-sifon/lnr.git

(you may also fork it to your own github account and then clone your fork). Then, simply type

    python setup.py install [--user]

where the `--user` flag is recommended so that the installation takes place in the home directory and does not require root privileges.

After this, just open a python terminal and type

    >>> import lnr

and all the modules described above will be available. Then type, e.g.,

    >>> help(lnr.bces)

to get more details.

----

(c) Cristóbal Sifón
Last modified 2018-01-07
