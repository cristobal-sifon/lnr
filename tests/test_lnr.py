import numpy as np
from numpy import testing

from lnr import bces, kelly, mcmc, mle, plot, scatter, to_linear, to_log


x = 3847
logx = np.log10(x)
xerr = 193


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
