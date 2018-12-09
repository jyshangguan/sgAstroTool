import numpy as np
import matplotlib.pyplot as plt

__all__ = ["plotDA", "plotdata", "flagDA"]

def plotDA(dx, dy, dxe=None, dye=None, cdim=None, cList=None, FigAx=None, label_on=False,
           **kwargs):
    """
    Plot the data array of GRAVITY.

    Parameters
    ----------
    dx : numpy 3-dim array
        The data array plotted as x axis which should have the dimensions
        [time, baseline, channel].
    dy : numpy 3-dim array
        The data array plotted as y axis which should have the dimensions
        [time, baseline, channel].
    dxe : numpy 3-dim array
        The data array plotted as the error of x axis which should have
        the dimensions [time, baseline, channel].
    dye : numpy 3-dim array
        The data array plotted as the error of y axis which should have
        the dimensions [time, baseline, channel].
    cdim : string (optional)
        The dimension (time, baseline, or channel) to be used as color code.
    cList : list (optional)
        List the colors to be used.
        The default is ["red", "orange", "green", "blue", "cyan", "purple"].
    FigAx : tuple (optional)
        The tuple of (fig, ax) of the figure.
    label_on : bool, default: False
        The flag to label the symbols if True.
    **kwargs : dict
        The keywords of pyplot.plot().

    Returns
    -------
    (fig, ax) : tuple
        The fig and ax of the figure.
    """
    assert cdim in ["time", "baseline", "channel", None]
    if FigAx is None:
        fig = plt.figure(figsize=(7, 7))
        ax  = plt.gca()
    else:
        fig, ax = FigAx
    if cList is None:
        cList = ["red", "orange", "green", "blue", "cyan", "purple"]
    if cdim is None:
        x = dx.flatten()
        y = dy.flatten()
        if (dxe is None) and (dye is None):
            ax.plot(x, y, **kwargs)
        else:
            if not dxe is None:
                xe = dxe.flatten()
            else:
                xe = None
            if not dye is None:
                ye = dye.flatten()
            else:
                ye = None
            ax.errorbar(x, y, xerr=xe, yerr=ye, **kwargs)
    else:
        if cdim == "time":
            asrc = 0
        elif cdim == "baseline":
            asrc = 1
        elif cdim == "channel":
            asrc = 2
        else:
            raise RuntimeError("Error in caxis ({0})!".format(caxis))
        #-> Reformat the array to make the color axis in the first dimension.
        dxr = np.moveaxis(dx, asrc, 0)
        dyr = np.moveaxis(dy, asrc, 0)
        if not dxe is None:
            dxer = np.moveaxis(dxe, asrc, 0)
        if not dye is None:
            dyer = np.moveaxis(dye, asrc, 0)
        for loop in range(dxr.shape[0]):
            x = dxr[loop, :, :].flatten()
            y = dyr[loop, :, :].flatten()
            c = cList[loop%len(cList)]
            if label_on:
                label = "{0}: {1}".format(cdim, loop)
            else:
                label = None
            if (dxe is None) and (dye is None):
                ax.plot(x, y, color=c, label=label, **kwargs)
            else:
                if not dxe is None:
                    xe = dxer[loop, :, :].flatten()
                else:
                    xe = None
                if not dye is None:
                    ye = dyer[loop, :, :].flatten()
                else:
                    ye = None
                ax.errorbar(x, y, xerr=xe, yerr=ye, color=c, label=label, **kwargs)
    return (fig, ax)

def plotdata(data, xaxis, yaxis, xeaxis=None, yeaxis=None, time=None, baseline=None,
             channel=None, mask=None, cdim=None, cList=None, FigAx=None, label_on=False,
             **kwargs):
    """
    Plot the data array of GRAVITY.

    Parameters
    ----------
    data : Dict
        The dict of GRAVITY data.
    xaxis : string
        The keyword to plot as the x axis.
    yaxis : string
        The keyword to plot as the y axis.
    xeaxis : string
        The keyword to plot as the error of x axis.
    yeaxis : string
        The keyword to plot as the error of y axis.
    time : list (optional)
        List the column number of the time dimension to be plotted.
    baseline : list (optional)
        List the column number of the baseline dimension to be plotted.
    channel : list (optional)
        List the column number of the channel to be plotted.
    mask : numpy 3-dim bool array (optional)
        Additional mask to the data array with dimensions the same as da.
    cdim : string (optional)
        The dimension (time, baseline, or channel) to be used as color code.
    cList : list (optional)
        List the colors to be used.
        The default is ["red", "orange", "green", "blue", "cyan", "purple"].
    FigAx : tuple (optional)
        The tuple of (fig, ax) of the figure.
    **kwargs : dict
        The keywords of pyplot.plot().

    Returns
    -------
    FigAx : tuple, (fig, ax)
        The fig and ax of the figure.
    """
    kList = data.keys()
    assert xaxis in kList
    assert yaxis in kList
    dx = data[xaxis]
    dy = data[yaxis]
    nt, nb, nc = dx.shape
    if time is None:
        tFlag = []
    else:
        tFlag = list(set(np.arange(nt)) - set(time))
    if baseline is None:
        bFlag = []
    else:
        bFlag = list(set(np.arange(nb)) - set(baseline))
    if channel is None:
        cFlag = []
    else:
        cFlag = list(set(np.arange(nc)) - set(channel))
    dx = flagDA(dx, tFlag, bFlag, cFlag, mask)
    dy = flagDA(dy, tFlag, bFlag, cFlag, mask)
    if not xeaxis is None:
        assert xeaxis in kList
        dxe = flagDA(data[xeaxis], tFlag, bFlag, cFlag, mask)
    else:
        dxe = None
    if not yeaxis is None:
        assert yeaxis in kList
        dye = flagDA(data[yeaxis], tFlag, bFlag, cFlag, mask)
    else:
        dye = None
    FigAx = plotDA(dx, dy, dxe, dye, cdim, cList, FigAx, label_on, **kwargs)
    return FigAx

def flagDA(da, time=[], baseline=[], channel=[], mask=None):
    """
    Flag the data array with numpy array mask.

    Parameters
    ----------
    da : numpy 3-dim array
        The data array with dimensions [time, baseline, channel].
    time : list
        List the column numbers of the first dimension to be flagged.
    baseline : list
        List the column numbers of the second dimension to be flagged.
    channel : list
        List the column numbers of the third dimension to be flagged.
    mask : numpy 3-dim bool array (optional)
        Additional mask to the data array with dimensions the same as da.

    Returns
    -------
    da : numpy 3-dim masked array
        The masked data array.
    """
    nt, nb, nc = da.shape
    msk = np.zeros([nt, nb, nc], dtype=bool)
    for t in time:
        msk[t, :, :] = True
    for b in baseline:
        msk[:, b, :] = True
    for c in channel:
        msk[:, :, c] = True
    msk_nan = np.isnan(da)
    msk_fin = msk | msk_nan
    if not mask is None:
        assert mask.shape == msk.shape
        msk_fin = msk_fin | mask
    da = np.ma.array(da, mask=msk_fin)
    return da
