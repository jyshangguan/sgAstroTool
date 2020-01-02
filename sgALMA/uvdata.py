# The functions working on uv data.
#
#
import numpy as np

def average_uvdata(vis, datacolumn, uvbins, units="klambda", axis="real",
                   tmp_key="DATA_DESC_ID=0", verbose=False):
    """
    Get the averaged uv data as a function of uv distance.
    Use visstat() in casa.

    Parameters
    ----------
    vis : string
        The name of the ms.
    datacolumn : string
        The data column to be used for visstat.
    uvbins : list
        The boundaries of the bins of the uv distance.
    units : string (default: "klambda")
        The unit of the uv distance.
    axis : string (default: "real")
        The axis of the data.
    tmp_key : string
        The key of the output tempt dict. Not sure whether it would change.
    verbose : bool (default: False)
        Print auxiliary information if True.

    Returns
    -------
    uvpoints : array
        The uv distance, units following the input units.
    avg_amps : array
        The averaged amplitude of the visiblity, units: mJy.
    error_amps : array
        The uncertainty of the avg_amps, units: mJy.
    """
    uvpoints = uvbins[:-1] + np.diff(uvbins) / 2 # get array of uvmidpoints over which avg taken
    avg_amps = [] # define an empty list in which to put the averaged amplitudes
    stddev_amps = [] # define an empty list in which to put the amp stddev
    numpoints = []
    #-> Iterate over the uvrange & build up the binned data values (looking only
    # at the Real part -- a pretty good approach if the signal is approximately
    # symmetric and you have recentered the phase center on the peak of emission):
    for loop in range(len(uvpoints)):
        uvrange = "{0}~{1} {2}".format(uvbins[loop], uvbins[loop+1], units)
        tmp = visstat(vis=vis, axis=axis, uvrange=uvrange, datacolumn=datacolumn)
        if tmp is None:
            avg_amps.append(np.nan)
            stddev_amps.append(np.nan)
            numpoints.append(np.nan)
            if verbose:
                print("** No good sampling in {0}".format(uvrange))
        else:
            avg_amps.append(tmp[tmp_key]["mean"])
            stddev_amps.append(tmp[tmp_key]["stddev"])
            numpoints.append(tmp[tmp_key]["npts"])
        if verbose:
            print("{0} -- Done.".format(uvrange))
    avg_amps = np.array(avg_amps) *  1000. # units: mJy
    error_amps = np.array(stddev_amps)/(np.sqrt(numpoints) - 1) * 1000. # units: mJy
    return uvpoints, avg_amps, error_amps
