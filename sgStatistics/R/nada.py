import numpy as np
import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects
#-> import R's "NADA" package
nada = importr("NADA")

__all__ = ["cenken"]

def cenken(y, yc, x, xc):
    """
    Computes Kendall's tau for doubly (x and y) censored data. Computes the
    Akritas-Theil-Sen nonparametric line, with the Turnbull estimate of
    intercept.

    The function calls NADA.cenken of R using rpy2.

    Parameters
    ----------
    y : 1D array
        The dependent variable.
    yc :  1D bool array
        The flag of the censored data of y, True for censored.
    x : 1D array
        The independent variable.
    xc :  1D bool array
        The flag of the censored data of x, True for censored.

    Returns
    -------
    rdict: dict
        slope : float
            The slope of the linear fit.
        intercept : float
            The intercept of the linear fit.
        tau : float
            The Kendall's tau correlation coefficient.
        p : float
            The p-value of the null hypothesis.
    """
    x  = robjects.FloatVector(x)
    xc = robjects.BoolVector(xc)
    y  = robjects.FloatVector(y)
    yc = robjects.BoolVector(yc)
    results = nada.cenken(y, yc, x, xc)
    rdict = {}
    for n, idx in zip(results.names, range(len(results))):
        rdict[n] = results[idx][0]
    return rdict
