import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.stats import sigma_clip
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy.visualization import SqrtStretch, AsinhStretch, LinearStretch, LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
import astropy.units as u
from scipy.interpolate import interp1d
import sgSpec as bf
from .analysis import *

__all__ = ["Plot_Map", "Plot_Line_Diagnose", "Plot_Simple_Diagnose", "Plot_Mom_Maps",
           "Plot_Channel_Maps"]

stretchDict = {
    "SqrtStretch": SqrtStretch,
    "AsinhStretch": AsinhStretch,
    "LinearStretch": LinearStretch,
    "LogStretch": LogStretch,
}

def Plot_Simple_Diagnose(cube, vel_range=[-400, 400], mask=None, mask_kws={}, map_vrange=None,
                         map_interpolation="none", contour_levels=[-1, 1, 3, 9],
                         do_fit=True, plot_w20=True, plot_w50=True, verbose=False):
    """
    Plot the results of a simple diagnostics of the data cube.
    """
    slab = cube.spectral_slab(vel_range[0] * u.km/u.s, vel_range[1] * u.km/u.s)
    m0 = slab.moment(order=0)
    if mask is None:
        mask_m0 = Mask_Segmentation(m0.value, wcs=m0.wcs, **mask_kws)
    else:
        mask_m0 = mask
    skyrms = SkyRMS_pixel(m0.value, mask_m0)
    spc_velc, spc_flux = Spectrum_Mask(cube, mask_m0)
    #--> Visualize the results
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_axes([0.05, 0.05, 0.45, 0.9])
    ax2 = fig.add_axes([0.58, 0.05, 0.45, 0.9])
    #---> Plot the masked mom0 map to make sure the roi is correct.
    if map_vrange is None:
        vmin = None
        vmax = None
    else:
        vmin = map_vrange[0]*skyrms
        vmax = map_vrange[1]*skyrms
    norm = ImageNormalize(stretch=AsinhStretch(), vmin=vmin, vmax=vmax)
    levels = [i*skyrms for i in contour_levels]
    contour_dict={"map": m0.value, "levels": levels, "kws": {"colors": "k", "linewidths": 2.}}
    Plot_Map(m0, norm=norm, contour_dict=contour_dict, FigAx=(fig, ax1), imshow_interpolation=map_interpolation)
    #---> Plot the spectrum and the model fitting
    fltr_nan = np.logical_not(np.isnan(spc_flux))
    spc_velc = spc_velc[fltr_nan]
    spc_flux = spc_flux[fltr_nan]
    if len(spc_flux) > 3: # Analyze the spectrum only when there are enough data
        pldres = Plot_Line_Diagnose(spc_velc, spc_flux, vel_range, do_fit, plot_w20,
                                    plot_w50, FigAx=(fig, ax2), verbose=verbose)
        if do_fit ^ pldres[1]:
            ax2.text(0.05, 0.95, "BusyFit fails", fontsize=24, transform=ax2.transAxes,
                     horizontalalignment='left', verticalalignment='top', backgroundcolor="w")
        ax2.axvspan(xmin=vel_range[0], xmax=vel_range[1], color="red", ls="--", lw="2.", alpha=0.1, label="Included")
        ax2.axvline(x=vel_range[0], color="red", ls="--", lw="2.", alpha=0.3)
        ax2.axvline(x=vel_range[1], color="red", ls="--", lw="2.", alpha=0.3)
        ax2.legend(fontsize=18)
    else:
        ax2.text(0.33, 0.53, "No spectral data...", fontsize=24, transform=ax2.transAxes,
                 horizontalalignment='left', verticalalignment='top', backgroundcolor="w")
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
    return (fig, (ax1, ax2))

def Plot_Mom_Maps(m0, m1, m2, norm0=None, norm1=None, norm2=None, contour_dict={},
                  map_vrange=[None, None], vperc1=[10, 90], vperc2=[20, 99], map_interpolation="none",
                  xlim=None, ylim=None):
    """
    Plot of the moment maps.
    """
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_axes([0.04, 0.08, 0.28, 0.84])
    ax2 = fig.add_axes([0.325, 0.08, 0.28, 0.84])
    ca2 = fig.add_axes([0.608, 0.08, 0.017, 0.84])
    ax3 = fig.add_axes([0.665, 0.08, 0.28, 0.84])
    ca3 = fig.add_axes([0.948, 0.08, 0.017, 0.84])
    #-> Plot mom0
    if norm0 is None:
        norm0 = ImageNormalize(stretch=AsinhStretch(), vmin=map_vrange[0], vmax=map_vrange[1])
    Plot_Map(m0, norm=norm0, contour_dict=contour_dict, FigAx=(fig, ax1), imshow_interpolation=map_interpolation,
             xlim=xlim, ylim=ylim)
    ax1.text(0.05, 0.95, "Moment 0", fontsize=24, transform=ax1.transAxes,
             horizontalalignment='left', verticalalignment='top', backgroundcolor="w")
    #-> Plot mom1
    if norm1 is None:
        fltr = np.logical_not(np.isnan(m1.value))
        vmin, vmax = np.percentile(m1.value[fltr], vperc1)
        norm1 = ImageNormalize(stretch=LinearStretch(), vmin=vmin, vmax=vmax)
    fig, ax2, im2 = Plot_Map(m1, norm=norm1, contour_dict=contour_dict, beam_on=False,
                             FigAx=(fig, ax2), imshow_interpolation=map_interpolation,
                             xlim=xlim, ylim=ylim)
    ax2.set_ylabel("")
    ax2.set_yticklabels([])
    cb2 = fig.colorbar(im2, cax=ca2, orientation="vertical")
    cb2.ax.minorticks_on()
    cb2.ax.tick_params(axis='both', which='major', length=8, labelsize=18, width=1.)
    cb2.ax.tick_params(axis="both", which="minor", length=5, width=1.)
    ax2.text(0.05, 0.95, "Moment 1 (km/s)", fontsize=24, transform=ax2.transAxes,
             horizontalalignment='left', verticalalignment='top', backgroundcolor="w")
    #-> Plot mom2
    if norm2 is None:
        fltr = np.logical_not(np.isnan(m2.value))
        vmin, vmax = np.percentile(m2.value[fltr], vperc2)
        norm2 = ImageNormalize(stretch=LinearStretch(), vmin=vmin, vmax=vmax)
    fig, ax3, im3 = Plot_Map(m2, norm=norm2, contour_dict=contour_dict, beam_on=False,
                             FigAx=(fig, ax3), imshow_interpolation=map_interpolation,
                             xlim=xlim, ylim=ylim)
    ax3.set_ylabel("")
    ax3.set_yticklabels([])
    cb3 = fig.colorbar(im3, cax=ca3, orientation="vertical")
    cb3.ax.minorticks_on()
    cb3.ax.tick_params(axis='both', which='major', length=8, labelsize=18, width=1.)
    cb3.ax.tick_params(axis="both", which="minor", length=5, width=1.)
    ax3.text(0.05, 0.95, "Moment 2 (km/s)", fontsize=24, transform=ax3.transAxes,
             horizontalalignment='left', verticalalignment='top', backgroundcolor="w")
    return (fig, ax1, (ax2, cb2), (ax3, cb3))

def Plot_Channel_Maps(cube, nrows, ncols, mask=None, norm=None, contour_on=True,
                      contour_dict={}, contour_levels=[-1, 1, 3], figure_size=None,
                      panel_size=5, channel_label=True, map_interpolation="none",
                      xlim=None, ylim=None):
    """
    Plot the channel maps.
    """
    if mask is None:
        spc_x = cube.spectral_axis
        slab  = cube.spectral_slab(spc_x[0], spc_x[-1])
        m0    = slab.moment(order=0)
        mask  = Mask_Segmentation(m0.value, wcs=m0.wcs)
    if norm is None:
        chnrms = CubeRMS_pixel(cube.value, mask)
        norm = ImageNormalize(stretch=AsinhStretch(), vmin=-2*chnrms, vmax=5*chnrms)
    if figure_size is None:
        figure_size = (panel_size*ncols, panel_size*nrows)
    fig = plt.figure(figsize=figure_size)
    axo = fig.add_subplot(111) # The out axis
    axo.spines['top'].set_visible(False)
    axo.spines['right'].set_visible(False)
    axo.spines['bottom'].set_visible(False)
    axo.spines['left'].set_visible(False)
    axo.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
    axo.set_xlabel('R.A. Offset (")', fontsize=28, labelpad=25)
    axo.set_ylabel('Dec. Offset (")', fontsize=28, labelpad=40)
    axs = fig.subplots(nrows, ncols)
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    counter = 0
    for loop_r in range(nrows):
        for loop_c in range(ncols):
            ax = axs[loop_r, loop_c]
            try:
                m = cube[counter, :, :]
                v = cube.spectral_axis[counter]
            except:
                ax.axis('off')
                continue
            chnrms = SkyRMS_pixel(m.value, mask)
            contourDict={}
            if contour_on:
                contourDict["map"] = contour_dict.get("map", m.value)
                contourDict["levels"] = contour_dict.get("levels", [i*chnrms for i in contour_levels])
                contourDict["kws"] = contour_dict.get("kws", {"colors": "black", "linewidths": 2.})
            fig, ax, im = Plot_Map(m, FigAx=(fig, ax), norm=norm, contour_dict=contourDict,
                                   imshow_interpolation=map_interpolation, xlim=xlim,
                                   ylim=ylim)
            if channel_label:
                ax.text(0.05, 0.95, r"${0:.0f}$ km/s".format(v.value), fontsize=24,
                        transform=ax.transAxes, horizontalalignment='left',
                        verticalalignment='top', color="k", backgroundcolor="white") #
            if loop_r != (nrows-1):
                ax.set_xticklabels([])
            if loop_c != 0:
                ax.set_yticklabels([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            counter += 1
    return (fig, axo, axs)

def Plot_Line_Diagnose(spc_x, spc_f, rmsFlag=[-400, 400], do_fit=True, plot_w20=True,
                       plot_w50=True, verbose=False, FigAx=None):
    """
    Plot the spectrum fitted with a busyfit model.

    Parameters
    ----------
    spc_x : array_like
        The x axis of the spectrum, preferentially in units: km/s.
    spc_f : array_like
        The flux of the spectrum.
    rmsFlag : list
        The range of x axis, within which the spectrum is not used to calculate
        the rms.
    do_fit : bool, default: True
        Fit the spectrum with the busy function if True.
    plot_w20 : bool
        Plot the W20 region derived from busyfit.
    plot_w50 : bool
        Plot the W50 region derived from busyfit.
    verbose : bool, default: False
        Print more information if True.
    FigAx : tuple (optional)
        The fig and ax of a figure.

    Returns
    -------
    FigAx : tuple
        The fig and ax of the figure.
    flag_fit : bool
        The flag whether there is a successful fit.
    """
    spc_len = len(spc_x)
    if spc_len < 3:
        raise ValueError("The length of the spectrum ({0}) is not enough!".format(spc_len))
    if FigAx is None:
        fig = plt.figure()
        ax  = plt.gca()
        FigAx = (fig, ax)
    else:
        fig, ax = FigAx
    #-> Plot the spectrum
    ax.step(spc_x.value, spc_f.value, color="k")
    #-> Plot the 1 sigma noise level
    spc_rms = bf.rms_spectrum(spc_x.value, spc_f.value, flag=rmsFlag)
    spc_med = bf.median_spectrum(spc_x.value, spc_f.value, flag=rmsFlag)
    ax.axhline(y=spc_med, color="gray", alpha=0.5, ls="--")
    ax.axhspan(spc_med-spc_rms, spc_med+spc_rms, color="gray", alpha=0.3, label=r"1 $\sigma$")
    #-> Plot the busyfit results
    flag_fit = False
    if do_fit:
        try:
            x_fit = spc_x.value
            y_fit = spc_f.value - spc_med
            fit_res  = bf.busyfit(x_fit, y_fit, rms=spc_rms)
            bf.busyFitPlot(fit_res, plot_w20=plot_w20, plot_w50=plot_w50, FigAx=(fig, ax),
                           yoffset=spc_med, color="r")
            flag_fit = True
            if verbose:
                w20_res = bf.lineParameters(fit_res, perc=20)
                print("Line center is {0:.5f}.".format(w20_res["Xc"]))
                print("W20 is {0:.2f}.".format(w20_res["Wline"]))
        except:
            if verbose:
                print("Cannot plot the busyfit results...")
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', length=8, labelsize=18, width=1., direction="in")
    ax.tick_params(axis='both', which='minor', length=5, width=1., direction="in")
    ax.set_xlabel("Velocity ({0})".format(spc_x.unit), fontsize=24)
    ax.set_ylabel("flux ({0})".format(spc_f.unit), fontsize=24)
    return (FigAx, flag_fit)

def Plot_Map(mom, cmap="viridis", norm=None, FigAx=None, imshow_interpolation="none",
             contour_dict={}, colorbar_on=False, colorbar_kws={}, beam_on=True,
             beam_kws={}, plain=False, xlim=None, ylim=None):
    """
    Plot the moment maps.
    """
    if FigAx is None:
        fig = plt.figure()
        ax  = plt.gca()
    else:
        fig, ax = FigAx
    x = mom.value
    if norm is None:
        rms  = np.nanstd(sigma_clip(x.flatten()))
        fltr = np.logical_not(np.isnan(x))
        vperc = np.atleast_1d(colorbar_kws.get("vpercentile", 98))
        if len(vperc) == 1:
            prc1 = 100-vperc
            prc2 = vperc
        elif len(vperc) == 2:
            prc1, prc2 = vperc
        else:
            raise ValueError("The length of vpercentile ({0}) is at most 2!".format(vperc))
        vmin = np.percentile(x[fltr], prc1)
        vmax = np.percentile(x[fltr], prc2)
        norm = ImageNormalize(x, stretch=LinearStretch(), vmin=vmin, vmax=vmax)
    #-> Make relative coordinates
    header = mom.wcs.to_header()
    cr_ra = header["CRVAL1"]
    cd_ra = header["CDELT1"]
    cr_dec = header["CRVAL2"]
    cd_dec = header["CDELT2"]
    dec0, ra0 = mom.world[0, 0] # Lower left
    dec1, ra1 = mom.world[-1, -1] # Upper right
    ra0 = (ra0.value - cr_ra - 0.5 * cd_ra) * 3600
    ra1 = (ra1.value - cr_ra + 0.5 * cd_ra) * 3600
    dec0 = (dec0.value - cr_dec - 0.5 * cd_dec) * 3600
    dec1 = (dec1.value - cr_dec + 0.5 * cd_dec) * 3600
    extent = [ra0, ra1, dec0, dec1]
    #-> Plot the image
    im = ax.imshow(x, cmap=cmap, extent=extent, norm=norm, origin="lower", interpolation=imshow_interpolation)
    ax.set_aspect("equal")
    if not xlim is None:
        ax.set_xlim(xlim)
    if not ylim is None:
        ax.set_ylim(ylim)
    #-> Draw the contour
    contour_data = contour_dict.get("map", None)
    if not contour_data is None:
        levels = contour_dict.get("levels", 5)
        contour_kws = contour_dict.get("kws", {})
        ax.contour(contour_data, levels, extent=extent, **contour_kws)
    #-> Plot the beam
    if beam_on:
        bmaj = mom.beam.major.to(u.arcsec).value
        bmin = mom.beam.minor.to(u.arcsec).value
        bpa  = mom.beam.pa
        Plot_Beam(ax, bmaj, bmin, bpa, **beam_kws)
    #-> Tune the axis
    if not plain:
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', length=8, width=1., labelsize=18)
        ax.tick_params(axis='both', which='minor', length=5, width=1., labelsize=18)
        ax.set_xlabel('R.A. Offset (")', fontsize=24)
        ax.set_ylabel('Dec. Offset (")', fontsize=24)
    #-> Draw the colorbar
    if colorbar_on:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(colorbar_kws.get("position", "right"),
                                  size=colorbar_kws.get("size", "5%"),
                                  pad=colorbar_kws.get("pad", 0.0))
        cb_orientation = colorbar_kws.get("orientation", "vertical")
        cb  = fig.colorbar(im, cax=cax, orientation=cb_orientation)
        if cb_orientation == "horizontal":
            cb.ax.xaxis.set_ticks_position("top")
            cb.ax.xaxis.set_label_position("top")
        if not plain:
            cb.ax.minorticks_on()
            cb.ax.tick_params(axis='both', which='major', length=8, labelsize=18, width=1.)
            cb.ax.tick_params(axis="both", which="minor", length=5, width=1.)
        return (fig, (ax, cb))
    else:
        return (fig, ax, im)

def Plot_Beam(ax, bmaj, bmin, bpa, **ellipse_kws):
    """
    Plot the beam.

    Parameters
    ----------
    ax : axis object
        The figure axis.
    bmaj : float
        The beam major axis, units: arcsec.
    bmin : float
        The beam minor axis, units: arcsec.
    bpa : float
        The beam position angle, units: degree.
    **ellipse_kws : dict
        The keywords for Ellipse.  The default color is white and the default
        position is on the lower-left corner.
    """
    # However, to plot it we need to negate the BPA since the rotation is the opposite direction
    # due to flipping RA.
    if ellipse_kws.get("zorder", None) is None:
        ellipse_kws["zorder"] = 5
    if ellipse_kws.get("color", None) is None:
        ellipse_kws["color"] = "white"
    if ellipse_kws.get("xy", None) is None:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        bx = xlim[0] - 0.1 * (xlim[0] - xlim[1])
        by = ylim[0] + 0.1 * (ylim[1] - ylim[0])
        ellipse_kws["xy"] = (bx, by)
    ax.add_artist(Ellipse(width=bmin, height=bmaj, angle=-bpa, **ellipse_kws))
