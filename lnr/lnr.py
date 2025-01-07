#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Various linear regression techniques

"""
from __future__ import absolute_import, division, print_function

import numpy as np
import sys
from matplotlib import pyplot as plt
from scipy import optimize, stats
import warnings

if sys.version_info[0] == 2:
    from itertools import izip
else:
    izip = zip
    xrange = range


def bces(
    x1,
    x2,
    x1err=[],
    x2err=[],
    cerr=[],
    logify=True,
    model="yx",
    bootstrap=5000,
    verbose="normal",
    seed=None,
    full_output=True,
):
    """
    Bivariate, Correlated Errors and intrinsic Scatter (BCES)
    translated from the FORTRAN code by Christina Bird and Matthew
    Bershady (Akritas & Bershady, 1996)

    Linear regression in the presence of heteroscedastic errors on both
    variables and intrinsic scatter

    Parameters
    ----------
      x1        : array of floats
                  Independent variable, or observable
      x2        : array of floats
                  Dependent variable
      x1err     : array of floats (optional)
                  Uncertainties on the independent variable
      x2err     : array of floats (optional)
                  Uncertainties on the dependent variable
      cerr      : array of floats (optional)
                  Covariances of the uncertainties in the dependent and
                  independent variables
      logify    : bool (default True)
                  Whether to take the log of the measurements in order
                  to estimate the best-fit power law instead of linear
                  relation
      model     : {'yx', 'xy', 'bi', 'orth'}
                  BCES model with which to calculate regression. See
                  Notes below for details.
      bootstrap : False or int (default 5000)
                  get the errors from bootstrap resampling instead of
                  the analytical prescription? if bootstrap is an int,
                  it is the number of bootstrap resamplings
      verbose   : {'quiet', 'normal', 'debug'}
                  Verbose level
      seed      : int (default None)
                  random seed
      full_output : bool (default True)
                  If True, return also the covariance between the
                  normalization and slope of the regression.

    Returns
    -------
      a         : tuple of length 2
                  Best-fit normalization and its uncertainty (a, da)
      b         : tuple of length 2
                  Best-fit slope and its uncertainty (b, db)

    Optional outputs
    ----------------
      cov_ab    : 2x2 array of floats
                  covariance between a and b. Returned if
                  `full_output==True`

    Notes
    -----
      If verbose is normal or debug, the results from all the BCES
      models will be printed (still, only the one selected in `model`
      will be returned).

      the `model` parameter:
        -'yx' stands for BCES(Y|X)
        -'xy' stands for BCES(X|Y)
        -'bi' stands for BCES Bisector
        -'orth' stands for BCES Orthogonal

    """
    rdm = np.random.default_rng(seed)

    def _bess_bootstrap(npts, x1, x2, x1err, x2err, cerr, nsim):
        ##added by Gerrit, July 2014
        # Unfortunately I needed a copy of the _bess function for
        # bootstrapping. Would be nicer if those two could be combined
        """
        Do the entire regression calculation for 4 slopes:
        OLS(Y|X), OLS(X|Y), bisector, orthogonal
        """
        # calculate sigma's for datapoints using length of confidence
        # intervals
        sig11var = np.sum(x1err**2, axis=1, keepdims=True) / npts
        sig22var = np.sum(x2err**2, axis=1, keepdims=True) / npts
        sig12var = np.sum(cerr, axis=1, keepdims=True) / npts

        # calculate means and variances
        x1av = np.mean(x1, axis=1, keepdims=True)
        x1var = x1.var(axis=1, keepdims=True)
        x2av = np.mean(x2, axis=1, keepdims=True)
        x2var = x2.var(axis=1, keepdims=True)
        covar_x1x2 = np.mean(
            (x1 - np.mean(x1, axis=1, keepdims=True))
            * (x2 - np.mean(x2, axis=1, keepdims=True)),
            axis=1,
            keepdims=True,
        )

        # compute the regression slopes for OLS(X2|X1), OLS(X1|X2),
        # bisector and orthogonal
        if model == "yx":
            modelint = 1
        else:
            modelint = 4
        b = np.zeros((modelint, nsim))
        b[0] = ((covar_x1x2 - sig12var) / (x1var - sig11var)).flatten()
        if model != "yx":
            b[1] = ((x2var - sig22var) / (covar_x1x2 - sig12var)).flatten()
            b[2] = (
                (b[0] * b[1] - 1 + np.sqrt((1 + b[0] ** 2) * (1 + b[1] ** 2)))
                / (b[0] + b[1])
            ).flatten()
            b[3] = 0.5 * (
                (b[1] - 1 / b[0])
                + np.sign(covar_x1x2).flatten() * (4 + (b[1] - 1 / b[0]) ** 2) ** 0.5
            )

        # compute intercepts for above 4 cases:
        a = x2av.flatten() - b * x1av.flatten()

        # set up variables to calculate standard deviations of slope and
        # intercept
        xi = []
        xi.append(
            (
                (x1 - x1av) * (x2 - b[0].reshape(nsim, 1) * x1 - a[0].reshape(nsim, 1))
                + b[0].reshape(nsim, 1) * x1err**2
            )
            / (x1var - sig11var)
        )
        if model != "yx":
            xi.append(
                (
                    (x2 - x2av)
                    * (x2 - b[1].reshape(nsim, 1) * x1 - a[1].reshape(nsim, 1))
                    + x2err**2
                )
                / covar_x1x2
            )
            xi.append(
                (
                    xi[0] * (1 + b[1].reshape(nsim, 1) ** 2)
                    + xi[1] * (1 + b[0].reshape(nsim, 1) ** 2)
                )
                / (
                    (b[0].reshape(nsim, 1) + b[1].reshape(nsim, 1))
                    * np.sqrt(
                        (1 + b[0].reshape(nsim, 1) ** 2)
                        * (1 + b[1].reshape(nsim, 1) ** 2)
                    )
                )
            )
            xi.append(
                (xi[0] / b[0].reshape(nsim, 1) ** 2 + xi[1])
                * b[3].reshape(nsim, 1)
                / np.sqrt(4 + (b[1].reshape(nsim, 1) - 1 / b[0].reshape(nsim, 1)) ** 2)
            )
        zeta = []
        for i in xrange(modelint):
            zeta.append(x2 - b[i].reshape(nsim, 1) * x1 - x1av * xi[i])

        # calculate  variance for all a and b
        bvar = np.zeros((4, nsim))
        avar = np.zeros((4, nsim))
        for i in xrange(modelint):
            bvar[i] = xi[i].var(axis=1, keepdims=False) / npts
            avar[i] = zeta[i].var(axis=1, keepdims=False) / npts
        return a, b, avar, bvar, xi, zeta

    def _bess(npts, x1, x2, x1err, x2err, cerr):
        """
        Do the entire regression calculation for 4 slopes:
          OLS(Y|X), OLS(X|Y), bisector, orthogonal
        """
        # calculate sigma's for datapoints using length of confidence
        # intervals
        sig11var = (x1err**2).sum() / npts
        sig22var = (x2err**2).sum() / npts
        sig12var = cerr.sum() / npts
        # calculate means and variances
        x1av = np.average(x1)
        x1var = np.var(x1)
        x2av = np.average(x2)
        x2var = np.var(x2)
        covar_x1x2 = ((x1 - x1av) * (x2 - x2av)).sum() / npts
        # compute the regression slopes for OLS(X2|X1), OLS(X1|X2),
        # bisector and orthogonal
        b = np.zeros(4)
        b[0] = (covar_x1x2 - sig12var) / (x1var - sig11var)
        b[1] = (x2var - sig22var) / (covar_x1x2 - sig12var)
        b[2] = (b[0] * b[1] - 1 + np.sqrt((1 + b[0] ** 2) * (1 + b[1] ** 2))) / (
            b[0] + b[1]
        )
        b[3] = 0.5 * (
            (b[1] - 1 / b[0])
            + np.sign(covar_x1x2) * np.sqrt(4 + (b[1] - 1 / b[0]) ** 2)
        )
        # compute intercepts for above 4 cases:
        a = x2av - b * x1av
        # set up variables to calculate standard deviations of slope
        # and intercept
        xi = [
            ((x1 - x1av) * (x2 - b[0] * x1 - a[0]) + b[0] * x1err**2)
            / (x1var - sig11var),
            ((x2 - x2av) * (x2 - b[1] * x1 - a[1]) + x2err**2) / covar_x1x2,
        ]
        xi.append(
            (xi[0] * (1 + b[1] ** 2) + xi[1] * (1 + b[0] ** 2))
            / ((b[0] + b[1]) * np.sqrt((1 + b[0] ** 2) * (1 + b[1] ** 2)))
        )
        xi.append(
            (xi[0] / b[0] ** 2 + xi[1]) * b[3] / np.sqrt(4 + (b[1] - 1 / b[0]) ** 2)
        )
        zeta = [x2 - bi * x1 - x1av * xii for bi, xii in zip(b, xi)]
        # calculate  variance for all a and b
        avar = np.var(zeta, axis=1) / npts
        bvar = np.var(xi, axis=1) / npts
        return a, b, avar, bvar, xi, zeta

    def _bootsamples(bootstrap, npts, x, y, xerr, yerr, cerr):
        b = rdm.integers(npts, size=(bootstrap, npts))
        out = np.transpose([x[b], y[b], xerr[b], yerr[b], cerr[b]], axes=(1, 0, 2))
        return out

    # ----  Main routine starts here  ---- #
    # convert to numpy arrays just in case
    x1, x2 = np.array([x1, x2])
    npts = len(x1)
    if len(x1err) == 0:
        x1err = np.zeros(npts)
    else:
        x1err = np.array(x1err)
    if len(x2err) == 0:
        x2err = np.zeros(npts)
    else:
        x2err = np.array(x2err)
    if len(cerr) == 0:
        cerr = np.zeros(npts)
    else:
        cerr = np.array(cerr)
    if logify:
        x1, x1errr = to_log(x1, x1err)
        x2, x1errr = to_log(x2, x2err)
    models = [
        ["yx", "xy", "bi", "orth"],
        ["BCES(Y|X)", "BCES(X|Y)", "BCES Bisector", "BCES Orthogonal"],
    ]
    # which to return?
    j = models[0].index(model)
    # are the errors defined?
    if verbose == "debug":
        print("x1 =", x1)
        print("x1err =", x1err)
        print("x2 =", x2)
        print("x2err =", x2err)
        print("cerr =", cerr)
        print("\n ** Returning values for", models[1][j], "**")
        if bootstrap is not False:
            print("    with errors from {} bootstrap resamplings".format(bootstrap))
        print()

    # calculate nominal fits
    bessresults = _bess(npts, x1, x2, x1err, x2err, cerr)
    a, b, avar, bvar, xi, zeta = bessresults
    # covariance between normalization and slope
    if full_output:
        cov_ab = np.cov(zeta[j], xi[j])

    if bootstrap is not False:
        # make bootstrap simulated datasets, and compute averages and
        # standard deviations of regression coefficients
        asim = np.zeros((bootstrap, 4))
        bsim = np.zeros((bootstrap, 4))
        samples = _bootsamples(bootstrap, npts, x1, x2, x1err, x2err, cerr)
        for i in xrange(bootstrap):
            asim[i], bsim[i] = _bess(npts, *samples[i])[:2]
        # this may happen when there are too few points and the chance of
        # all values being the same is not negligible (e.g., for 5 data
        # points this happens in ~1% of the samples)
        bad = (np.isnan(asim)) | (np.isinf(asim))
        nbad = bad[bad].size
        asim = asim[~bad].reshape((bootstrap - nbad // 4, 4))
        bsim = bsim[~bad].reshape((bootstrap - nbad // 4, 4))
        assum = np.sum(asim**2, axis=0)
        bssum = np.sum(bsim**2, axis=0)
        aavg = np.sum(asim, axis=0) / bootstrap
        bavg = np.sum(bsim, axis=0) / bootstrap

        sda = np.sqrt((assum - bootstrap * aavg**2) / (bootstrap - 1))
        sdb = np.sqrt((bssum - bootstrap * bavg**2) / (bootstrap - 1))
        sda[np.isnan(sda)] = 0
        sdb[np.isnan(sdb)] = 0

    if verbose in ("normal", "debug"):
        print("Fit                   B          err(B)")
        print("         A          err(A)")
        for i in xrange(4):
            print(
                "{0:<16s}  {1:9.2e} +/- {2:8.2e}"
                "    {3:10.3e} +/- {4:9.3e}".format(
                    models[1][i], b[i], bvar[i] ** 0.5, a[i], avar[i] ** 0.5
                )
            )
            if bootstrap is not False:
                print(
                    "{0:<16s}  {1:9.2e} +/- {2:8.2e}"
                    "    %10.3e +/- %9.3e".format(
                        "bootstrap", bavg[i], sdb[i], aavg[i], sda[i]
                    )
                )
            print()
        if verbose == "debug":
            print("cov[{0}] =".format(models[model]))
            print(cov_ab)

    if bootstrap is not False:
        if full_output:
            return (a[j], sda[j]), (b[j], sdb[j]), cov_ab
        else:
            return (a[j], sda[j]), (b[j], sdb[j])
    if full_output:
        out = ((a[j], np.sqrt(avar[j])), (b[j], np.sqrt(bvar[j])), cov_ab)
    else:
        out = ((a[j], np.sqrt(avar[j])), (b[j], np.sqrt(bvar[j])))
    return out


def kelly(
    x1,
    x2,
    x1err=[],
    x2err=[],
    cerr=[],
    logify=True,
    miniter=5000,
    maxiter=1e5,
    metro=True,
    silent=True,
    output="percentiles",
    full_output=None,
):
    """
    Python wrapper for the linear regression MCMC of Kelly (2007).
    Requires pidly (http://astronomy.sussex.ac.uk/~anthonys/pidly/) and
    an IDL license.

    Parameters
    ----------
      x1        : array of floats
                  Independent variable, or observable
      x2        : array of floats
                  Dependent variable
      x1err     : array of floats (optional)
                  Uncertainties on the independent variable
      x2err     : array of floats (optional)
                  Uncertainties on the dependent variable
      cerr      : array of floats (optional)
                  Covariances of the uncertainties in the dependent and
                  independent variables
      output    : {'full', 'percentiles', 'std'}
                  whether to return the full posterior distributions,
                  the median and lower and upper uncertainties, or the
                  median and standard deviation. Default 'percentiles'.
      full_output : bool (optional)
                  For backward compatibility. `full_output=True` sets
                  `output='full'` and `full_output=False` sets
                  `output='std'`. If not specified, this parameter is
                  ignored. DEPRECATED.

    Returns
    -------
      a, b, s   : float arrays
                  normalization, slope and intrinsic scatter, depending
                  on the `output` parameter
    """
    assert output in (
        "full",
        "percentiles",
        "std",
    ), "Invalid value of argument output. See function help for details."

    # import here so it's not required if any other function is called
    import pidly

    n = len(x1)
    assert len(x2) == n, "x1 and x2 must have same length"
    if len(x1err) == 0:
        x1err = np.zeros(n)
    if len(x2err) == 0:
        x2err = np.zeros(n)
    if len(cerr) == 0:
        cerr = np.zeros(n)
    if logify:
        x1, x1errr = to_log(x1, x1err)
        x2, x1errr = to_log(x2, x2err)
    idl = pidly.IDL()
    idl("x1", x1)
    idl("x2", x2)
    cmd = "linmix_err, x1, x2, fit"
    if len(x1err) == n:
        idl("x1err", x1err)
        cmd += ", xsig=x1err"
    if len(x2err) == n:
        idl("x2err", x2err)
        cmd += ", ysig=x2err"
    if len(cerr) == n:
        idl("cerr", cerr)
        cmd += ", xycov=cerr"
    cmd += ", miniter={0}, maxiter={1}".format(miniter, maxiter)
    if metro:
        cmd += ", /metro"
    if silent:
        cmd += ", /silent"
    idl(cmd)
    alpha = idl.ev("fit.alpha")
    beta = idl.ev("fit.beta")
    sigma = np.sqrt(idl.ev("fit.sigsqr"))

    if full_output is not None:
        msg = (
            "argument full_output is deprecated. Please use the argument"
            " output instead."
        )
        warnings.warn(msg, DeprecationWarning)
        output = "full" if full_output else "std"

    if output == "full":
        output = alpha, beta, sigma
    elif output == "percentiles":
        out = np.array(
            [
                [np.median(i), np.percentile(i, 16), np.percentile(i, 84)]
                for i in (alpha, beta, sigma)
            ]
        )
        out[:, 1:] = np.abs(out[:, 1:] - out[:, 0, np.newaxis])
    elif output == "std":
        out = np.array([[np.median(i), np.std(i)] for i in (alpha, beta, sigma)])
    return out


def mcmc(
    x1,
    x2,
    x1err=None,
    x2err=None,
    start=(1.0, 1.0, 0.5),
    starting_width=0.01,
    logify=True,
    nsteps=5000,
    nwalkers=100,
    nburn=0,
    seed=None,
    output="full",
):
    """
    Use emcee to find the best-fit linear relation or power law
    accounting for measurement uncertainties and intrinsic scatter.

    Assumes the following priors:
        intercept ~ uniform in the range (-inf,inf)
        slope ~ Student's t with 1 degree of freedom
        intrinsic scatter ~ 1/scatter

    Parameters
    ----------
      x1        : array of floats
                  Independent variable, or observable
      x2        : array of floats
                  Dependent variable
      x1err     : array of floats (optional)
                  Uncertainties on the independent variable
      x2err     : array of floats (optional)
                  Uncertainties on the dependent variable
      start     : tuple of 3 floats (optional)
                  Initial guesses for zero point, slope, and intrinsic
                  scatter. Results are not very sensitive to these
                  values so they shouldn't matter a lot.
      starting_width : float
                  Starting points for each walker will be drawn
                  from a normal distribution with mean `start` and
                  standard deviation `starting_width*start`
      logify    : bool (default True)
                  Whether to take the log of the measurements in order
                  to estimate the best-fit power law instead of linear
                  relation
      nsteps    : int (default 5000)
                  Number of steps each walker should take in the MCMC
      nwalkers  : int (default 100)
                  Number of MCMC walkers
      nburn     : int (default 500)
                  Number of samples to discard to give the MCMC enough
                  time to converge.
      seed      : int (default None)
                  random seed for initial guesses
      output    : list of ints or 'full' (default 'full')
                  If 'full', then return the full samples (except for
                  burn-in section) for each parameter. Otherwise, each
                  float corresponds to a percentile that will be
                  returned for each parameter.

    Returns
    -------
      The returned value is a numpy array whose shape depends on the
      choice of `output`, but in all cases it either corresponds to the
      posterior samples or the chosen percentiles of three parameters:
      the normalization (or intercept), the slope and the intrinsic
      scatter of a (log-)linear fit to the data.

    """
    import emcee

    rdm = np.random.default_rng()
    # just in case
    x1, x2 = np.array([x1, x2])
    if x1err is None:
        x1err = np.zeros(x1.size)
    else:
        x1err = np.array(x1err)
    if x2err is None:
        x2err = np.zeros(x1.size)
    else:
        x2err = np.array(x2err)

    def lnlike(theta, x, y, xerr, yerr):
        """Likelihood"""
        a, b, s = theta
        model = a + b * x
        sigma = ((b * xerr) ** 2 + yerr * 2 + s**2) ** 0.5
        lglk = (
            2 * np.log(sigma).sum()
            + (((y - model) / sigma) ** 2).sum()
            + np.log(x.size) * (2 * np.pi) ** 0.5 / 2
        )
        return -lglk

    def lnprior(theta):
        """
        Log-priors. Scatter must be positive; using a Student's t
        distribution with 1 dof for the slope.
        """
        a, b, s = theta
        # positive scatter
        if s < 0:
            return -np.inf
        # flat prior on intercept
        lnp_a = 0
        # Student's t for slope
        lnp_b = np.log(stats.t.pdf(b, 1))
        # Jeffrey's prior for scatter (not normalized)
        lnp_s = -np.log(s)
        # total
        return lnp_a + lnp_b + lnp_s

    def lnprob(theta, x, y, xerr, yerr):
        """Posterior"""
        return lnprior(theta) + lnlike(theta, x, y, xerr, yerr)

    if logify:
        x1, x1err = to_log(x1, x1err)
        x2, x2err = to_log(x2, x2err)
    start = np.array(start)
    ndim = start.size
    pos = rdm.normal(start, starting_width * start, (nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x1, x2, x1err, x2err))
    sampler.run_mcmc(pos, nsteps)
    samples = np.array([sampler.chain[:, nburn:, i].reshape(-1) for i in xrange(ndim)])
    # do I need this? I don't think so because I always take log10's
    if logify:
        samples[2] *= np.log(10)
    if output == "full":
        return samples
    else:
        try:
            values = [[np.percentile(s, o) for o in output] for s in samples]
            return values
        except TypeError:
            msg = (
                "ERROR: wrong value for argument output in mcmc()."
                ' Must be "full" or list of ints.'
            )
            print(msg)
            exit()
    return


def mle(
    x1,
    x2,
    x1err=None,
    x2err=None,
    s_int=True,
    slope=None,
    start=(1.0, 1.0, 0.1),
    bootstrap=1000,
    logify=True,
    seed=None,
    **kwargs
):
    """
    Maximum Likelihood Estimation of best-fit parameters

    Parameters
    ----------
      x1, x2    : float arrays
                  the independent and dependent variables.
      x1err, x2err : float arrays (optional)
                  measurement uncertainties on independent and
                  dependent variables. Any of the two, or both, can be
                  supplied.
      s_int     : boolean (default True)
                  whether to include intrinsic scatter in the MLE. Set to
                  `False` if slope is provided
      slope     : float (optional)
                  value of the slope if only the zero point is being fit.
                  If provided, only the first of the initial guess values
                  will be considered.
      start     : tuple of floats
                  initial guess for free parameters. If `s_int` is
                  True, then po must have 3 elements; otherwise it can
                  have one (the zero point) or two (for the zero point
                  and the slope). Additional parameters will be ignored
      bootstrap : int or False
                  if not `False`, it is the number of samples with which
                  to estimate uncertainties on the best-fit parameters
      logify    : boolean (default True)
                  whether to convert the values to log10's. This is to
                  calculate the best-fit power law. Note that the
                  result is given for the equation log(y)=a+b*log(x) --
                  i.e., the zero point must be converted to 10**a if
                  `logify=True`
      seed      : int (default None)
                  random seed for bootstrap
      kwargs    : dictionary (optional)
                  arguments passed to `scipy.optimize.fmin`

    Returns
    -------
      a         : float
                  Maximum Likelihood Estimate of the zero point. Note
                  that if logify=True, the power-law intercept is 10**a
      b         : float
                  Maximum Likelihood Estimate of the slope
      s         : float (optional, if s_int=True)
                  Maximum Likelihood Estimate of the intrinsic scatter

    """
    rdm = np.random.default_rng()
    x1, x2 = np.array([x1, x2])
    n = x1.size
    if x2.size != n:
        raise ValueError("x1 and x2 must have same length")
    if x1err is None:
        x1err = 1e-8 * np.absolute(x1.min()) * np.ones(n)
    else:
        x1err = np.array(x1err)
    if x2err is None:
        x2err = 1e-8 * np.absolute(x2.min()) * np.ones(n)
    else:
        x2err = np.array(x2err)
    if x1err.size != n:
        raise ValueError("x1err must have the same length as x1")
    if x2err.size != n:
        raise ValueError("x2err must have the same length as x2")
    if logify:
        x1, x1err = to_log(x1, x1err)
        x2, x2err = to_log(x2, x2err)

    # try to allow s_int without slope later
    if slope is not None:
        s_int = False

    norm = np.log(n * (2 * np.pi) ** 0.5) / 2
    f = lambda x, a, b: a + b * x
    if slope is not None:
        print("slope =", slope, start)
        w = lambda b, dx, dy: ((b * dx) ** 2 + dy**2) ** 0.5

        def _loglike(p, x, y, *args, slope=slope):
            wi = w(slope, *args)
            return norm + (2 * np.log(wi) + ((y - f(x, p[0], slope)) / wi) ** 2).sum()

        start = start[:1]

    elif s_int:
        w = lambda b, s, dx, dy: ((b * dx) ** 2 + dy**2 + s**2) ** 0.5

        def _loglike(p, x, y, *args):
            wi = w(p[1], p[2], *args)
            return norm + (2 * np.log(wi) + ((y - f(x, *p[:2])) / wi) ** 2).sum()

    else:
        w = lambda b, dx, dy: ((b * dx) ** 2 + dy**2) ** 0.5

        def _loglike(p, x, y, *args):
            wi = w(p[1], *args)
            return norm + (2 * np.log(wi) + ((y - f(x, *p[:2])) / wi) ** 2).sum()

        start = start[:2]

    fit = optimize.minimize(_loglike, start, args=(x1, x2, x1err, x2err), **kwargs).x
    # bootstrap errors?
    if bootstrap is False:
        return fit
    jboot = rdm.integers(0, n, (bootstrap, n))
    boot = [
        optimize.minimize(
            _loglike,
            start,
            args=(x1[j], x2[j], x1err[j], x2err[j]),
        ).x
        for j in jboot
    ]
    out_err = np.std(boot, axis=0)
    return fit, out_err


def plot(
    t,
    a,
    b,
    a_err=0,
    b_err=0,
    s=None,
    pivot=0,
    ax=None,
    log=False,
    color="b",
    lw=2,
    alpha=0.5,
    **kwargs
):
    """
    alpha is used to shade the uncertainties from a_err and b_err
    **kwargs is passed to plt.plot() for the central line only
    the error band has zorder=-10

    """
    if log:
        if pivot == 0:
            pivot = 1
        y = lambda A, B: 10**A * (t / pivot) ** B
    else:
        y = lambda A, B: A + B * (t - pivot)
    if ax is None:
        ax = plt
    # the length may vary depending on whether it's a default color
    # (e.g., 'r' or 'orange') or an rgb(a) color, etc, but as far as
    # I can think of none of these would have length 2.
    if len(color) != 2:
        color = (color, color)
    print("in lnr.plot: color =", color)
    ax.plot(t, y(a, b), ls="-", color=color[0], lw=lw, **kwargs)
    if a_err != 0 or b_err != 0:
        # to make it compatible with either one or two values
        a_err = np.array([a_err]).flatten()
        b_err = np.array([b_err]).flatten()
        if a_err.size == 1:
            a_err = [a_err, a_err]
        if b_err.size == 1:
            b_err = [b_err, b_err]
        err = [
            y(a - a_err[0], b - b_err[0]),
            y(a - a_err[0], b + b_err[1]),
            y(a + a_err[1], b - b_err[0]),
            y(a + a_err[1], b + b_err[1]),
        ]
        ylo = np.min(err, axis=0)
        yhi = np.max(err, axis=0)
        ax.fill_between(
            t, ylo, yhi, color=color[1], alpha=alpha, lw=0, edgecolor="none", zorder=-10
        )
    if s:
        if log:
            ax.plot(t, (1 + s) * y(a, b), ls="--", color=color[0], lw=lw)
            ax.plot(t, y(a, b) / (1 + s), ls="--", color=color[0], lw=lw)
        else:
            ax.plot(t, y(a, b) + s, ls="--", color=color[0], lw=lw)
            ax.plot(t, y(a, b) - s, ls="--", color=color[0], lw=lw)
    return


def scatter(slope, zero, x1, x2, x1err=[], x2err=[]):
    """
    Used mainly to measure scatter for the BCES best-fit

    """
    x1, x2 = np.array([x1, x2])
    n = len(x1)
    x2pred = zero + slope * x1
    s = sum((x2 - x2pred) ** 2) / (n - 1)
    if len(x2err) == n:
        s_obs = sum((x2err / x2) ** 2) / n
        s0 = s - s_obs
    # print(s**0.5, s_obs**0.5, s0**0.5)
    return s0**0.5


def to_linear(logx, logxerr=[], base=10, which="average"):
    """
    Take log measurements and uncertainties and convert to linear
    values.


    Parameters
    ----------
    logx : array of floats
        logarithm of measurements to be linearized

    Optional Parameters
    -------------------
    logxerr : array of floats
        uncertainties on logx
    base : float
        base with which the logarithms have been calculated
    which : {'lower', 'upper', 'both', 'average'}
        Which uncertainty to report; note that when converting to/from
        linear and logarithmic spaces, errorbar symmetry is not
        preserved. The following are the available options:

            if which=='lower': xerr = logx - base**(logx-logxerr)
            if which=='upper': xerr = base**(logx+logxerr) - logx

        If `which=='both'` then both values are returned, and if
        `which=='average'`, then the average of the two is returned.
        Default is 'average'.

    Returns
    -------
    x : array of floats
        values in linear space, i.e., base**logx
    xerr : array of floats
        uncertainties, as discussed above
    """
    if np.iterable(logx):
        return_scalar = False
    else:
        return_scalar = True
        logx = [logx]
    logx = np.array(logx)
    if not np.iterable(logxerr):
        logxerr = [logxerr]
    if len(logxerr) == 0:
        logxerr = np.zeros(logx.shape)
    else:
        logxerr = np.array(logxerr)
    assert logx.shape == logxerr.shape, "The shape of logx and logxerr must be the same"
    assert which in ("lower", "upper", "both", "average"), (
        "Valid values for optional argument `which` are 'lower', 'upper',"
        " 'average' or 'both'."
    )
    x = base**logx
    lo = x - base ** (logx - logxerr)
    hi = base ** (logx + logxerr) - x
    if return_scalar:
        x = x[0]
        lo = lo[0]
        hi = hi[0]
    if which == "both":
        return x, lo, hi
    if which == "lower":
        xerr = lo
    elif which == "upper":
        xerr = hi
    else:
        xerr = 0.5 * (lo + hi)
    return x, xerr


def to_log(x, xerr=[], base=10, which="average"):
    """
    Take linear measurements and uncertainties and transform to log
    values.


    Parameters
    ----------
    x : array of floats
        measurements of which to take logarithms

    Optional Parameters
    -------------------
    xerr : array of floats
        uncertainties on x
    base : float
        base with which the logarithms should be calculated. FOR NOW USE
        ONLY 10.
    which : {'lower', 'upper', 'both', 'average'}
        Which uncertainty to report; note that when converting to/from
        linear and logarithmic spaces, errorbar symmetry is not
        preserved. The following are the available options:

            if which=='lower': logxerr = logx - log(x-xerr)
            if which=='upper': logxerr = log(x+xerr) - logx

        If `which=='both'` then both values are returned, and if
        `which=='average'`, then the average of the two is returned.
        Default is 'average'.

    Returns
    -------
    logx : array of floats
        values in log space, i.e., base**logx
    logxerr : array of floats
        log-uncertainties, as discussed above
    """
    assert (
        np.issubdtype(type(base), np.floating)
        or np.issubdtype(type(base), np.integer)
        or base == "e"
    )
    if np.iterable(x):
        return_scalar = False
    else:
        return_scalar = True
        x = [x]
    x = np.array(x)
    if not np.iterable(xerr):
        xerr = [xerr]
    if len(xerr) == 0:
        xerr = np.zeros(x.shape)
    else:
        xerr = np.array(xerr)
    assert xerr.shape == x.shape, "The shape of x and xerr must be the same"
    assert which in ("lower", "upper", "both", "average"), (
        "Valid values for optional argument `which` are 'lower', 'upper',"
        " 'average' or 'both'."
    )

    if base == 10:
        f = lambda y: np.log10(y)
    elif base in (np.e, "e"):
        f = lambda y: np.log(y)
    else:
        f = lambda y: np.log(y) / np.log(base)
    logx = f(x)
    logxlo = logx - f(x - xerr)
    logxhi = f(x + xerr) - logx
    if return_scalar:
        logx = logx[0]
        logxlo = logxlo[0]
        logxhi = logxhi[0]
    if which == "both":
        return logx, logxlo, logxhi
    if which == "lower":
        logxerr = logxlo
    elif which == "upper":
        logxerr = logxhi
    else:
        logxerr = 0.5 * (logxlo + logxhi)
    return logx, logxerr
