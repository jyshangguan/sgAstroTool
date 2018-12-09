###
## This package provides some convenient wrappers of the functions from ASURV
## from IRAF (stsdas.analysis.statistics).
###
import sys
import re
import numpy as np
import os

__all__ = ["kmestimate", "twosampt", "bhkmethod"]

def Stdout2(new_out):
    """
    Redirect the stdout.
    """
    old_out, sys.stdout = sys.stdout, new_out # replace sys.stdout
    return sys.stdout, old_out

tmp_file = "sgstatistics.tmp"
nout = open(tmp_file, "w")
nout, oout = Stdout2(nout)

from stsci.tools import capable
capable.OF_GRAPHICS = False
from pyraf import iraf
iraf.stsdas()
iraf.analysis()
iraf.statistics()
iraf_kmestimate = iraf.kmestimate
iraf_twosampt = iraf.twosampt
iraf_bhkmethod = iraf.bhkmethod

nout.close()
Stdout2(oout)
os.remove(tmp_file)

def kmestimate(censors, values, kme_data="tmp_kme_data.dat", kme_output="tmp_kme_output.dat",
               verbose=True, diff=False):
    """
    A convenient wrapper of iraf.kmestimate

    Parameters
    ----------
    censors : array like
        The list of censors, 0 for detection and -1 for upper limit.
    values : array like
        The list of values to calculate the distribution.
    kme_data : string, default: "tmp_kme_data.dat"
        The file to save the data file for the iraf.kmestimate task.
    kme_output : string, default: "tmp_kme_output.dat"
        The file to save the output file of the iraf.kmestimate task.
    verbose : bool, default: True
        Print warnings and provide extensive output if True.
    diff : bool, default: False
       Print differential form of Kaplan-Meier estimator if True.

    Returns
    -------
    results : dict
        Detections : float
            The number of detection data.
        Non-detection : float
            The number of non-detection data.
        Percentile : list or None
            The 75th, 50th, 25th percentile of the distribution. None, if the
            function goes wrong.
        Mean : list or None
            The mean and its uncertainty of the distribution. None, if the function
            goes wrong.

    Notes
    -----
    None.
    """
    censors = np.atleast_1d(censors)
    values = np.atleast_1d(values)
    fltr_nan = np.isnan(values)
    #-> Filter out the nan.
    if np.sum(fltr_nan) > 0:
        if verbose:
            print("There are nan in the values ignored!")
        fltr_nonnan = np.logical_not(fltr_nan)
        censors = censors[fltr_nonnan]
        values  = values[fltr_nonnan]
    flag_censor = censors == -1
    nNond = np.sum(flag_censor)
    nDetc = len(flag_censor) - nNond
    data = np.transpose( np.array([censors, values]) )
    np.savetxt(kme_data, data, delimiter=" ", fmt=["%d", "%.4f"])
    if os.path.isfile(kme_output):
        os.remove(kme_output)
    nout = open(kme_output, "w")
    nout, oout = Stdout2(nout)
    iraf_kmestimate(PYin="{0}[1,2]".format(kme_data), verbose=verbose, diff=diff)
    nout.close()
    Stdout2(oout)
    kmr = open(kme_output, "r")
    content = kmr.readlines()
    kmr.close()
    count = 0
    n_percentile = None
    n_mean = None
    for line in content:
        if line.find("Percentile") > -1:
            n_percentile = count+2
        if line.find("Mean=") > -1:
            n_mean = count
        count += 1
    #-> Filter the floats
    prog = re.compile("(?<=\s)*[\+-]?[0-9]*\.[0-9]*")
    if n_percentile is None:
        result_perc = [np.nan, np.nan, np.nan]
        flag_rm = False
    else:
        r_perc = re.findall(prog, content[n_percentile])
        result_perc = [float(r_perc[0]), float(r_perc[1]), float(r_perc[2])]
        flag_rm = True
    if n_mean is None:
        result_mean = np.nan
    else:
        r_mean = re.findall(prog, content[n_mean])
        result_mean = [float(r_mean[0]), float(r_mean[1])]
    result = {
        "Detections": nDetc,
        "Non-detection": nNond,
        "Percentile": result_perc,
        "Mean": result_mean
    }
    if kme_output != "tmp_kme_output.dat":
        flag_rm = False
    if flag_rm:
        os.remove(kme_data)
        os.remove(kme_output)
    return result


def twosampt(censor1, value1, censor2, value2,
             tst_data="tmp_tst_data.dat", tst_output="tmp_tst_output.dat",
             verbose=True):
    """
    A convenient wrapper of iraf.twosampt.
    The results are the probabilities for the two input samples drawn from
    the same parent sample.  The detailed explanations can be found in:
        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?twosampt
    
    If the results of the tests differ significantly, then the Peto-Prentice 
    test is probably the most reliable.  Peto-Prentice test reduces to Gehan 
    test when there are no censored 
    observations.

    Parameters
    ----------
    censor1 : array like
        The list of censors, 0 for detections, -1 for upper limits, and 1 for
        lower limits.
    value1 : array like
        The list of values to calculate the distribution.
    censor2 : array like
        The list of censors, 0 for detections, -1 for upper limits, and 1 for
        lower limits.
    value2 : array like
        The list of values to calculate the distribution.
    tst_data : string, default: "tmp_tst_data.dat"
        The file to save the data file for the iraf.twosampt task.
    tst_output : string, default: "tmp_tst_output.dat"
        The file to save the output file of the iraf.twosampt task.
    verbose : bool, default: True
        Print warnings and provide extensive output if True.

    Returns
    -------
    results : dict
        The possibilities for the two samples to be drawn from the
        sampe parent sample.  The methods adopted here are:
            Gehan Permutation
            Gehan Hypergeometric
            Logrank
            Peto&Peto
            Peto&Prentice

    Notes
    -----
    The two versions of the Gehan test assume that the censoring patterns of 
    the two samples are the same, but the version with hypergeometric variance 
    is more reliable in case of different censoring patterns. The logrank test 
    results appear to be correct as long as the censoring patterns are not very 
    different. Peto-Prentice seems to be the test least affected by differences 
    in the censoring patterns. There is little known about the limitations of 
    the Peto-Peto test.
    """
    value1  = np.atleast_1d(value1)
    censor1 = np.atleast_1d(censor1)
    value2  = np.atleast_1d(value2)
    censor2 = np.atleast_1d(censor2)
    fltr1_nan = np.isnan(value1)
    fltr2_nan = np.isnan(value2)
    if np.sum(fltr1_nan) > 0:
        if verbose:
            print("There are nan in the value1 ignored!")
        fltr1_nonnan = np.logical_not(fltr1_nan)
        value1  = value1[fltr1_nonnan]
        censor1 = censor1[fltr1_nonnan]
    if np.sum(fltr2_nan) > 0:
        if verbose:
            print("There are nan in the value2 ignored!")
        fltr2_nonnan = np.logical_not(fltr2_nan)
        value2  = value1[fltr2_nonnan]
        censor2 = censor1[fltr2_nonnan]
    v = np.concatenate([value1, value2])
    c = np.concatenate([censor1, censor2])
    g = np.concatenate([np.zeros_like(value1), np.ones_like(value2)])
    data = np.transpose(np.array([c, v, g]))
    np.savetxt(tst_data, data, fmt=["%d", "%.2f", "%d"])
    if os.path.isfile(tst_output):
        os.remove(tst_output)
    nout = open(tst_output, "w")
    nout, oout = Stdout2(nout)
    iraf_twosampt(PYin="{0}[1,2,3]".format(tst_data), first="0", second="1",
                  verbose=verbose)
    Stdout2(oout)
    tstr = open(tst_output, "r")
    content = tstr.readlines()
    tstr.close()
    #-> Number of censored data
    ncnsr = np.sum(np.abs(censor1)) + np.sum(np.abs(censor2))
    if ncnsr > 0:
        methodList = ["Gehan Permutation", "Gehan Hypergeometric", "Logrank",
                      "Peto&Peto", "Peto&Prentice"]
    else:
        methodList = ["Gehan Permutation", "Gehan Hypergeometric", "Logrank",
                      "Peto&Peto"]
    #-> Grab the probability results
    prog = re.compile("[0-9]*\.[0-9]*")
    count = 0
    results = {}
    for line in content:
        if line.find("Probability") > -1:
            poss = re.findall(prog, line)[0]
            results[methodList[count]] = poss
            count += 1
        if count > len(methodList):
            raise RuntimeError("There are more probabilities than expected...")
    if count < len(methodList):
        raise RuntimeError("There are less probabilities than expected...")
    if tst_output != "tmp_tst_output.dat":
        flag_rm = False
    else:
        flag_rm = True
    if flag_rm:
        os.remove(tst_data)
        os.remove(tst_output)
    return results

def bhkmethod(censor, value1, value2, bhk_data="tmp_bhk_data.dat",
              bhk_output="tmp_bhk_output.dat", verbose=True):
    """
    A convenient wrapper of iraf.bhkmethod.
    The results are the probability for the null hypothesis of the two
    quantities to be correlated. The detailed explanations can be found in:
        http://stsdas.stsci.edu/cgi-bin/gethelp.cgi?bhkmethod

    Parameters
    ----------
    censor : array like
        The list of censors, 0 for detections; negative values for upper limits,
        and positive for lower limits; 1, 2, 3, and 4 for X, Y, Both, and Mix.
        (I am not sure what Mix is for...)
    value1 : array like
        The list of values to calculate the correlation.
    value2 : array like
        The list of values to calculate the correlation.
    bhk_data : string, default: "tmp_bhk_data.dat"
        The file to save the data file for the iraf.bhkmethod task.
    bhk_output : string, default: "tmp_bhk_output.dat"
        The file to save the output file of the iraf.bhkmethod task.
    verbose : bool, default: True
        Print warnings and provide extensive output if True.

    Returns
    -------
    results : dict
        The Kendall's tau, Z-value, and the probability of the null hypothesis
        that the two quantities are correlated.
        Keywords: "tau", "z-value", "Probability"

    Notes
    -----
    None.
    """
    censor = np.atleast_1d(censor)
    value1 = np.atleast_1d(value1)
    value2 = np.atleast_1d(value2)
    fltr_nan = np.isnan(value1) | np.isnan(value2)
    #-> Filter out the nan.
    if np.sum(fltr_nan) > 0:
        if verbose:
            print("There are nan in the values ignored!")
        fltr_nonnan = np.logical_not(fltr_nan)
        censor = censor[fltr_nonnan]
        value1 = value1[fltr_nonnan]
        value2 = value2[fltr_nonnan]
    #-> Analyze the data
    fltr_ux = censor == -1 # Upper limit X
    fltr_uy = censor == -2 # Upper limit Y
    fltr_ub = censor == -3 # Upper limit Both
    fltr_um = censor == -4 # Upper limit Mix
    fltr_lx = censor == 1  # Lower limit X
    fltr_ly = censor == 2  # Lower limit Y
    fltr_lb = censor == 3  # Lower limit Both
    fltr_lm = censor == 4  # Lower limit Mix
    nUx = np.sum(fltr_ux)
    nUy = np.sum(fltr_uy)
    nUb = np.sum(fltr_ub)
    nUm = np.sum(fltr_um)
    nLx = np.sum(fltr_lx)
    nLy = np.sum(fltr_ly)
    nLb = np.sum(fltr_lb)
    nLm = np.sum(fltr_lm)
    if verbose:
        print("Upper limits: X({0}) Y({1}) Both({2}) Mix({3})".format(nUx, nUy, nUb, nUm))
        print("Lower limits: X({0}) Y({1}) Both({2}) Mix({3})".format(nLx, nLy, nLb, nLm))
    #-> Dump the data file
    data = np.array( np.transpose([censor, value1, value2]) )
    np.savetxt(bhk_data, data, delimiter=" ", fmt=["%d", "%.4f", "%.4f"])
    if os.path.isfile(bhk_output):
        os.remove(bhk_output)
    nout = open(bhk_output, "w")
    nout, oout = Stdout2(nout)
    iraf_bhkmethod(PYin="{0}[1,2,3]".format(bhk_data), verbose=verbose)
    nout.close()
    Stdout2(oout)
    bhk = open(bhk_output, "r")
    content = bhk.readlines()
    bhk.close()
    count = 0
    n_percentile = None
    n_mean = None
    prog = re.compile("[0-9]*\.[0-9]*")
    for line in content:
        if line.find("Kendall's tau") > -1:
            n_tau = count
            break
        count += 1
    n_z = count + 1
    n_p = count + 2
    results = {
        "tau": re.findall(prog, content[n_tau])[0],
        "z-value": re.findall(prog, content[n_z])[0],
        "Probability": re.findall(prog, content[n_p])[0]
    }
    return results
