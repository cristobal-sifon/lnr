#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Various linear regression techniques

"""
import numpy
import pylab
import stattools
from itertools import izip
from scipy import optimize

def bces(x1, x2, x1err=[], x2err=[], cerr=[], logify=True, model='yx', \
         bootstrap=5000, verbose='normal', full_output=True):
    """
    Bivariate, Correlated Errors and intrinsic Scatter (BCES)
    translated from the FORTRAN code by Christina Bird and Matthew Bershady
    (Akritas & Bershady, 1996)

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
                  Whether to take the log of the measurements in order to
                  estimate the best-fit power law instead of linear relation
      model     : {'yx', 'xy', 'bi', 'orth'}
                  BCES model with which to calculate regression. See Notes
                  below for details.
      bootstrap : False or int (default 5000)
                  get the errors from bootstrap resampling instead of the
                  analytical prescription? if bootstrap is an int, it is the
                  number of bootstrap resamplings
      verbose   : str (default 'normal')
                  Verbose level. Options are {'quiet', 'normal', 'debug'}
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
                  covariance between a and b. Returned if full_output is set to
                  True.

    Notes
    -----
      If verbose is normal or debug, the results from all the BCES models will
      be printed (still, only the one selected in *model* will be returned).

      the *model* parameter:
        -'yx' stands for BCES(Y|X)
        -'xy' stands for BCES(X|Y)
        -'bi' stands for BCES Bisector
        -'orth' stands for BCES Orthogonal

    """

    def _bess_bootstrap(npts, x1, x2, x1err, x2err, cerr, nsim):
        ##added by Gerrit, July 2014
        ##Unfortunately I needed a copy of the _bess function for bootstrapping.
        #Would be nicer if those two could be combined
        """
        Do the entire regression calculation for 4 slopes:
        OLS(Y|X), OLS(X|Y), bisector, orthogonal
        """
        #calculate sigma's for datapoints using length of confidence intervals
        sig11var = np.sum(x1err ** 2,axis=1,keepdims=True) / npts
        sig22var = np.sum(x2err ** 2,axis=1,keepdims=True) / npts
        sig12var = np.sum(cerr,axis=1,keepdims=True) / npts

        # calculate means and variances
        x1av = np.mean(x1,axis=1,keepdims=True)
        x1var = x1.var(axis=1,keepdims=True)
        x2av = np.mean(x2,axis=1,keepdims=True)
        x2var = x2.var(axis=1,keepdims=True)
        covar_x1x2 = np.mean((x1-np.mean(x1,axis=1,keepdims=True)) * \
                             (x2-np.mean(x2,axis=1,keepdims=True)),
                             axis=1,keepdims=True)

        # compute the regression slopes for OLS(X2|X1), OLS(X1|X2),
        # bisector and orthogonal
        if model == 'yx':
            modelint = 1
        else:
            modelint = 4
        b = np.zeros((modelint,nsim))
        b[0] = ((covar_x1x2 - sig12var) / (x1var - sig11var)).flatten()
        if model != 'yx':
            b[1] = ((x2var - sig22var) / (covar_x1x2 - sig12var)).flatten()
            b[2] = ((b[0] * b[1] - 1 + np.sqrt((1 + b[0] ** 2) * \
                   (1 + b[1] ** 2))) / (b[0] + b[1])).flatten()
            b[3] = 0.5 * ((b[1] - 1 / b[0]) + np.sign(covar_x1x2).flatten()* \
                   np.sqrt(4 + (b[1] - 1 / b[0]) ** 2))

        # compute intercepts for above 4 cases:
        a = x2av.flatten() - b * x1av.flatten()

        # set up variables to calculate standard deviations of slope and
        # intercept
        xi = []
        xi.append(((x1 - x1av) * (x2 - b[0].reshape(nsim,1) * x1 - \
                                  a[0].reshape(nsim,1)) + \
                   b[0].reshape(nsim,1) * x1err ** 2) / \
                  (x1var - sig11var))
        if model != 'yx':
            xi.append(((x2 - x2av) * (x2 - b[1].reshape(nsim,1) * x1 - \
                                      a[1].reshape(nsim,1)) + x2err ** 2) / \
                      covar_x1x2)
            xi.append((xi[0] * (1 + b[1].reshape(nsim,1) ** 2) + \
                       xi[1] * (1 + b[0].reshape(nsim,1) ** 2)) / \
                      ((b[0].reshape(nsim,1) + \
                       b[1].reshape(nsim,1)) * \
                       np.sqrt((1 + b[0].reshape(nsim,1) ** 2) * \
                               (1 + b[1].reshape(nsim,1) ** 2))))
            xi.append((xi[0] / b[0].reshape(nsim,1) ** 2 + xi[1]) * \
                      b[3].reshape(nsim,1) / \
                      np.sqrt(4 + (b[1].reshape(nsim,1) - \
                              1 / b[0].reshape(nsim,1)) ** 2))
        zeta = []
        for i in xrange(modelint):
            zeta.append(x2 - b[i].reshape(nsim,1) * x1 - x1av * xi[i])

        # calculate  variance for all a and b
        bvar = np.zeros((4,nsim))
        avar = np.zeros((4,nsim))
        for i in xrange(modelint):
            bvar[i] = xi[i].var(axis=1,keepdims=False)/ npts
            avar[i] = zeta[i].var(axis=1,keepdims=False) / npts
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
        x1av = numpy.average(x1)
        x1var = numpy.var(x1)
        x2av = numpy.average(x2)
        x2var = numpy.var(x2)
        covar_x1x2 = ((x1 - x1av) * (x2 - x2av)).sum() / npts
        # compute the regression slopes for OLS(X2|X1), OLS(X1|X2),
        # bisector and orthogonal
        b = numpy.zeros(4)
        b[0] = (covar_x1x2 - sig12var) / (x1var - sig11var)
        b[1] = (x2var - sig22var) / (covar_x1x2 - sig12var)
        b[2] = (b[0] * b[1] - 1 + numpy.sqrt((1 + b[0]**2) * \
               (1 + b[1] ** 2))) / (b[0] + b[1])
        b[3] = 0.5 * ((b[1] - 1 / b[0]) + numpy.sign(covar_x1x2) * \
               numpy.sqrt(4 + (b[1] - 1 / b[0])**2))
        # compute intercepts for above 4 cases:
        a = x2av - b * x1av
        # set up variables to calculate standard deviations of slope
        # and intercept
        xi = [((x1 - x1av) * (x2 - b[0]*x1 - a[0]) + b[0]*x1err**2) / \
               (x1var - sig11var),
              ((x2 - x2av) * (x2 - b[1]*x1 - a[1]) + x2err**2) / \
               covar_x1x2]
        xi.append((xi[0] * (1 + b[1]**2) + xi[1] * (1 + b[0]**2)) / \
                  ((b[0] + b[1]) * numpy.sqrt((1 + b[0]**2) * (1 + b[1]**2))))
        xi.append((xi[0] / b[0]**2 + xi[1]) * b[3] / \
                  numpy.sqrt(4 + (b[1] - 1 / b[0])**2))
        zeta = [x2 - bi*x1 - x1av*xii for bi, xii in zip(b, xi)]
        # calculate  variance for all a and b
        avar = numpy.var(zeta, axis=1) / npts
        bvar = numpy.var(xi, axis=1) / npts
        return a, b, avar, bvar, xi, zeta

    def _bootspbec(npts, x, y, xerr, yerr, cerr):
        """
        Bootstrap samples
        """
        b = numpy.random.randint(npts, size=npts)
        xboot = x[b]
        xerrboot = xerr[b]
        yboot = y[b]
        yerrboot = yerr[b]
        cerrboot = cerr[b]
        return xboot, yboot, xerrboot, yerrboot, cerrboot

    def _bootsamples(bootstrap, npts, x, y, xerr, yerr, cerr):
        b = numpy.random.randint(npts, size=(bootstrap,npts))
        out = numpy.transpose([x[b], y[b], xerr[b], yerr[b], cerr[b]],
                              axes=(1,0,2))
        return out

    # ----  Main routine starts here  ---- #
    # convert to numpy arrays just in case
    x1 = numpy.array(x1)
    x2 = numpy.array(x2)
    x1err = numpy.array(x1err)
    x2err = numpy.array(x2err)
    cerr = numpy.array(cerr)
    npts = len(x1)
    if len(x1err) == 0:
        x1err = numpy.zeros(npts)
    if len(x2err) == 0:
        x2err = numpy.zeros(npts)
    if len(cerr) == 0:
        cerr = numpy.zeros(npts)
    if logify:
        x1, x2, x1err, x2err = to_log(x1, x2, x1err, x2err)
    models = [['yx', 'xy', 'bi', 'orth'],
              ['BCES(Y|X)', 'BCES(X|Y)', 'BCES Bisector', 'BCES Orthogonal']]
    # which to return?
    j = models[0].index(model)
    # are the errors defined?
    if verbose == 'debug':
        print 'x1 =', x1
        print 'x1err =', x1err
        print 'x2 =', x2
        print 'x2err =', x2err
        print 'cerr =', cerr
        print '\n ** Returning values for', models[1][j], '**'
        if bootstrap is not False:
            print '    with errors from %d bootstrap resamplings' %bootstrap
        print ''

    # calculate nominal fits
    bessresults = _bess(npts, x1, x2, x1err, x2err, cerr)
    a, b, avar, bvar, xi, zeta = bessresults
    # covariance between normalization and slope
    if full_output:
        cov_ab = numpy.cov(zeta[j], xi[j])

    if bootstrap is not False:
        # make bootstrap simulated datasets, and compute averages and
        # standard deviations of regression coefficients
        #asum, assum, bsum, bssum, sda, sdb = numpy.zeros((6,4))
        asim = numpy.zeros((bootstrap,4))
        bsim = numpy.zeros((bootstrap,4))
        samples = _bootsamples(bootstrap, npts, x1, x2, x1err, x2err, cerr)
        for i in xrange(bootstrap):
            asim[i], bsim[i] = _bess(npts, *samples[i])[:2]
        # this may happen when there are too few points and the chance of
        # all values being the same is not negligible (e.g., for 5 data
        # points this happens in ~1% of the samples)
        bad = (numpy.isnan(asim)) | (numpy.isinf(asim))
        nbad = bad[bad].size
        asim = asim[~bad].reshape((bootstrap-nbad/4,4))
        bsim = bsim[~bad].reshape((bootstrap-nbad/4,4))
        assum = numpy.sum(asim**2, axis=0)
        bssum = numpy.sum(bsim**2, axis=0)
        aavg = numpy.sum(asim, axis=0) / bootstrap
        bavg = numpy.sum(bsim, axis=0) / bootstrap

        sda = numpy.sqrt((assum - bootstrap * aavg**2) / (bootstrap-1))
        sdb = numpy.sqrt((bssum - bootstrap * bavg**2) / (bootstrap-1))
        sda[numpy.isnan(sda)] = 0
        sdb[numpy.isnan(sdb)] = 0

    if verbose in ('normal', 'debug'):
        print 'Fit                   B          err(B)'
        print '         A          err(A)'
        for i in xrange(4):
            print '%s  %9.2e +/- %8.2e    %10.3e +/- %9.3e' \
                  %(models[1][i].ljust(16), b[i],
                    numpy.sqrt(bvar[i]), a[i], numpy.sqrt(avar[i]))
            if bootstrap is not False:
                print '%s  %9.2e +/- %8.2e    %10.3e +/- %9.3e' \
                      %('bootstrap'.ljust(16), bavg[i],
                        sdb[i], aavg[i], sda[i])
            print ''
        if verbose == 'debug':
            print 'cov[%s] =' %models[model]
            print covar_ab

    if bootstrap is not False:
        if full_output:
          return (a[j], sda[j]), (b[j], sdb[j]), cov_ab
        else:
          return (a[j], sda[j]), (b[j], sdb[j])
    if full_output:
        out = ((a[j], numpy.sqrt(avar[j])),
               (b[j], numpy.sqrt(bvar[j])),
               cov_ab)
    else:
        out = ((a[j], numpy.sqrt(avar[j])),
               (b[j], numpy.sqrt(bvar[j])))
    return out

def scatter(slope, zero, x1, x2, x1err=[], x2err=[]):
    """
    Used mainly to measure scatter for the BCES best-fit

    """
    n = len(x1)
    x2pred = zero + slope * x1
    s = sum((x2 - x2pred) ** 2) / (n - 1)
    if len(x2err) == n:
        s_obs = sum((x2err / x2) ** 2) / n
        s0 = s - s_obs
    print numpy.sqrt(s), numpy.sqrt(s_obs), numpy.sqrt(s0)
    return numpy.sqrt(s0)

def kelly(x1, x2, x1err=[], x2err=[], cerr=[], logify=True,
          miniter=5000, maxiter=1e5, metro=True, silent=True,
          full_output=False):
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
      full_output : bool
                  whether to return best-fit and 1sigma uncertainties or the
                  full MCMC chain

    Returns
    -------

    """
    import pidly
    n = len(x1)
    if len(x2) != n:
        raise ValueError('x1 and x2 must have same length')
    if len(x1err) == 0:
        x1err = numpy.zeros(n)
    if len(x2err) == 0:
        x2err = numpy.zeros(n)
    if len(cerr) == 0:
        cerr = numpy.zeros(n)
    if logify:
        x1, x2, x1err, x2err = to_log(x1, x2, x1err, x2err)
    idl = pidly.IDL()
    idl('x1', x1)
    idl('x2', x2)
    cmd = 'linmix_err, x1, x2, fit'
    if len(x1err) == n:
        idl('x1err', x1err)
        cmd += ', xsig=x1err'
    if len(x2err) == n:
        idl('x2err', x2err)
        cmd += ', ysig=x2err'
    if len(cerr) == n:
        idl('cerr', cerr)
        cmd += ', xycov=cerr'
    cmd += ', miniter=%d, maxiter=%d' %(miniter, maxiter)
    if metro:
        cmd += ', /metro'
    if silent:
        cmd += ', /silent'
    idl(cmd)
    alpha = idl.ev('fit.alpha')
    beta = idl.ev('fit.beta')
    sigma = numpy.sqrt(idl.ev('fit.sigsqr'))
    if full_output:
        return alpha, beta, sigma
    out = [(numpy.median(i), numpy.std(i)) for i in (alpha, beta, sigma)]
    return out

def mcmc(x1, x2, x1err=[], x2err=[], po=(1,1,0.5), logify=True,
         nsteps=5000, nwalkers=100, nburn=500, output='full'):
    """
    Use emcee to find the best-fit linear relation or power law
    accounting for measurement uncertainties and intrinsic scatter

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
      po        : tuple of 3 floats (optional)
                  Initial guesses for zero point, slope, and intrinsic
                  scatter. Results are not very sensitive to these values
                  so they shouldn't matter a lot.
      logify    : bool (default True)
                  Whether to take the log of the measurements in order to
                  estimate the best-fit power law instead of linear relation
      nsteps    : int (default 5000)
                  Number of steps each walker should take in the MCMC
      nwalkers  : int (default 100)
                  Number of MCMC walkers
      nburn     : int (default 500)
                  Number of samples to discard to give the MCMC enough time
                  to converge.
      output    : list of ints or 'full' (default 'full')
                  If 'full', then return the full samples (except for burn-in
                  section) for each parameter. Otherwise, each float
                  corresponds to a percentile that will be returned for
                  each parameter.

    Returns
    -------
      See *output* argument above for return options.

    """
    import emcee
    if len(x1err) == 0:
        x1err = numpy.ones(len(x1))
    if len(x2err) == 0:
        x2err = numpy.ones(len(x1))
    def lnlike(theta, x, y, xerr, yerr):
        a, b, s = theta
        model = a + b*x
        sigma = numpy.sqrt((b*xerr)**2 + yerr*2 + s**2)
        lglk = 2 * sum(numpy.log(sigma)) + \
               sum(((y-model) / sigma) ** 2) + \
               numpy.log(len(x)) * numpy.sqrt(2*numpy.pi) / 2
        return -lglk
    def lnprior(theta):
        a, b, s = theta
        if s >= 0:
            return 0
        return -numpy.inf
    def lnprob(theta, x, y, xerr, yerr):
        lp = lnprior(theta)
        return lp + lnlike(theta, x, y, xerr, yerr)
    if logify:
        x1, x2, x1err, x2err = to_log(x1, x2, x1err, x2err)
    start = numpy.array(po)
    ndim = len(start)
    pos = [start + 1e-4*numpy.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(x1,x2,x1err,x2err))
    sampler.run_mcmc(pos, nsteps)
    samples = numpy.array([sampler.chain[:,nburn:,i].reshape(-1) \
                           for i in xrange(ndim)])
    if logify:
        samples[2] *= numpy.log(10)
    if output == 'full':
        return samples
    else:
        try:
            values = [[numpy.percentile(s, o) for o in output]
                      for s in samples]
            return values
        except TypeError:
            msg = 'ERROR: wrong value for argument output in mcmc().'
            msg += ' Must be "full" or list of ints.'
            print msg
            exit()
    return

def mle(x1, x2, x1err=[], x2err=[], cerr=[], s_int=True, po=(1,0,0.1),
        bootstrap=1000, verbose=False, logify=True, full_output=False):
    """
    Maximum Likelihood Estimation of best-fit parameters

    Parameters
    ----------
      x1, x2    : float arrays
                  the independent and dependent variables.
      x1err, x2err : float arrays (optional)
                  measurement uncertainties on independent and dependent
                  variables. Any of the two, or both, can be supplied.
      cerr      : float array (same size as x1)
                  covariance on the measurement errors (NOT YET IMPLEMENTED)
      s_int     : boolean (default True)
                  whether to include intrinsic scatter in the MLE.
      po        : tuple of floats
                  initial guess for free parameters. If s_int is True, then
                  po must have 3 elements; otherwise it can have two (for the
                  zero point and the slope)
      bootstrap : int or False
                  if not False, it is the number of samples with which
                  to estimate uncertainties on the best-fit parameters
      verbose   : boolean (default False)
                  verbose?
      logify    : boolean (default True)
                  whether to convert the values to log10's. This is to
                  calculate the best-fit power law. Note that the result is
                  given for the equation log(y)=a+b*log(x) -- i.e., the
                  zero point must be converted to 10**a if logify=True
      full_output : boolean (default False)
                  numpy.optimize.fmin's full_output argument

    Returns
    -------
      a         : float
                  Maximum Likelihood Estimate of the zero point. Note that
                  if logify=True, the power-law intercept is 10**a
      b         : float
                  Maximum Likelihood Estimate of the slope
      s         : float (optional, if s_int=True)
                  Maximum Likelihood Estimate of the intrinsic scatter

    """
    n = x1.size
    if x2.size != n:
        raise ValueError('x1 and x2 must have same length')
    if len(x1err) == 0:
        x1err = numpy.zeros(n)
    if len(x2err) == 0:
        x2err = numpy.zeros(n)
    if logify:
        x1, x2, x1err, x2err = to_log(x1, x2, x1err, x2err)

    log = numpy.log
    fmin = optimize.fmin

    norm = log(n * (2*numpy.pi)**0.5) / 2
    f = lambda x, a, b: a + b * x
    if s_int:
        w = lambda b, s, dx, dy: ((b*dx)**2 + dy**2 + s**2)**0.5
        #w = lambda b, s: ((b*x1err)**2 + x2err**2 + s**2)**0.5
        def _loglike(p, x, y, *args):
            wi = w(p[1], p[2], *args)
            return norm + (2*log(wi) + ((y - f(x, *p[:2])) / wi)**2).sum()
    else:
        w = lambda b, dx, dy: ((b*dx)**2 + dy**2)**0.5
        def _loglike(p, x, y, *args):
            wi = w(p[1], *args)
            return norm + (2*log(wi) + ((y - f(x, *p[:2])) / wi)**2).sum()
        po = po[:2]

    fit = fmin(_loglike, po, args=(x1,x2,x1err,x2err),
               disp=verbose, full_output=full_output)
    # bootstrap errors?
    if bootstrap is False:
        return fit
    #def _loglike(p, x, y):
        #return norm + (2*log(p[2]) + ((y - f(x, *p[:2])) / p[2])**2).sum()
    jboot = numpy.random.randint(0, n, (bootstrap,n))
    boot = [fmin(_loglike, po, args=(x1[j],x2[j],x1err[j],x2err[j]),
                 disp=False, full_output=False)
            for j in jboot]
    out_err = numpy.std(boot, axis=0)
    # uncertainties by looking at the chi2
    #chi2 = (((x2 - f(*out[:2])) / x2err)**2).sum()
    #dof = len(x1) - 3 - 1
    #print 'chi2/dof = %.2f/%d = %.2f' %(chi2, dof, chi2/dof)
    out = numpy.transpose([fit, out_err])
    return out

def to_log(x1, x2, x1err=[], x2err=[]):
    """
    Take linear measurements and uncertainties and transform to log values.

    """
    logx1 = numpy.log10(numpy.array(x1))
    logx2 = numpy.log10(numpy.array(x2))
    if numpy.any(x1err):
        x1err = numpy.log10(numpy.array(x1)+numpy.array(x1err)) - logx1
    else:
        x1err = numpy.zeros(x1.size)
    if numpy.any(x2err):
        x2err = numpy.log10(numpy.array(x2)+numpy.array(x2err)) - logx2
    else:
        x2err = numpy.zeros(x1.size)
    return logx1, logx2, x1err, x2err

def plot(t, a, b, a_err=0, b_err=0, s=None, pivot=0, ax=None,
         log=False, color='b', lw=2, alpha=0.5, **kwargs):
    """
    alpha is used to shade the uncertainties from a_err and b_err

    **kwargs is passed to pylab.plot() for the central line only

    """
    if log:
        if pivot == 0:
            pivot = 1
        y = lambda A, B: 10**A * (t/pivot)**B
    else:
        y = lambda A, B: A + B * (t - pivot)
    if ax is None:
        ax = pylab
    ax.plot(t, y(a,b), ls='-', color=color, lw=lw, **kwargs)
    if a_err != 0 or b_err != 0:
        # to make it compatible with either one or two values
        a_err = numpy.array([a_err]).flatten()
        b_err = numpy.array([b_err]).flatten()
        if a_err.size == 1:
            a_err = [a_err, a_err]
        if b_err.size == 1:
            b_err = [b_err, b_err]
        err = [y(a-a_err[0], b-b_err[0]), y(a-a_err[0], b+b_err[1]),
               y(a+a_err[1], b-b_err[0]), y(a+a_err[1], b+b_err[1])]
        ylo = numpy.min(err, axis=0)
        yhi = numpy.max(err, axis=0)
        ax.fill_between(t, ylo, yhi, color=color, alpha=alpha, lw=0,
                        edgecolor='none')
    if s:
        if log:
            ax.plot(t, (1+s)*y(a,b), ls='--', color=color, lw=lw)
            ax.plot(t, y(a,b)/(1+s), ls='--', color=color, lw=lw)
        else:
            ax.plot(t, y(a,b) + s, ls='--', color=color, lw=lw)
            ax.plot(t, y(a,b) - s, ls='--', color=color, lw=lw)
    return
