import numpy as np
from numpy import testing

from lnr import bces, kelly, mcmc, mle, plot, scatter, to_linear, to_log


x = 3847
logx = np.log10(x)
xerr = 193


def test_mle():
    testing.assert_raises(AssertionError, mle, 1, [1])
    testing.assert_raises(AssertionError, mle, [1], 1)
    testing.assert_allclose(
        mle([0,1,2,3], [1.3,1.2,2.1,2.4], logify=False, bootstrap=False),
        [1.1200563114, 0.4199846599, 0.2049369291])


def test_to_linear():
    # no errors
    testing.assert_allclose(to_linear(logx)[0], x)
    # errors with default kwargs
    testing.assert_allclose(
        to_linear(3.5851221863068155, 0.0218064110470430), x, xerr)


def test_to_log():
    # no errors
    testing.assert_allclose(to_log(x)[0], logx)
    # errors with default kwargs
    testing.assert_allclose(
        to_log(x, xerr), [3.5851221863068155, 0.02180641104704306])
