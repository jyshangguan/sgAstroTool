import datetime
import numpy as np
from astropy.io import fits
from astropy import units as u
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy

__all__ = ["GravitySet", "GravityData", "GravityVis", "GravityP2VMRED", "gravity_DimLen"]

class GravitySet(object):
    """
    A set of gravity data observed at different times.
    """
    def __init__(self, gd_list=None, file_list=None, verbose=True):
        """
        Parameters
        ----------
        gd_list : list (optional)
            The list of GravityData.  If it is provided, the parameters to read
            from fits files will be ignored.
        file_list : list (optional)
            The list of fits file names.
        verbose : bool, default: True
            Print notes if True.
        """
        if gd_list is None:
            assert not file_list is None
            gd_list = []
            for filename in file_list:
                gd_list.append(GravityData(filename=filename, verbose=verbose))
        else:
            if verbose & (not file_list is None):
                print("The gd_list is used so file_list is ignored!")
        self.gd_list = gd_list
        self.__length = len(gd_list)

    def plot_uv(self, insname="ft", FigAx=None, colorcode=None, scatter_kws=None,
                legend_kws=None, ignored_channels=None, label_fontsize=24, tick_labelsize=18,
                text_fontsize=16, show_colorbar=False, verbose=False):
        """
        Plot the UV coverage of the data.

        Parameters
        ----------
        insname : string
            The instrument name, "ft", "sc", or their polarization components.
        colorcode : string (optional)
            The keyword of the data used for the color code, vis2 or visamp, or
            other keywords.  The data should have the same shape as the uv coordinates.
        FigAx : tuple (optional)
            The Figure, Axes and (optional) Colorbar objects generated in prior.
        scatter_kws : dict (optional)
            The keywords for plt.scatter() function
        legend_kws : dict (optional)
            The keywords for plt.legend() function
        ignored_channels : list (optional)
            The list of channel indices (0~4 for FT and 0~209 for SC) ignored in
            the plot.
        label_fontsize : float, default: 24
            The fontsize of the labels of both axes.
        tick_labelsize : float, default: 18
            The fontsize of the ticklabel of both axes.
        text_fontsize : float, default: 16
            The fontsize of the text in the figure.
        show_colorbar : bool, default: False
            Show the colorbar when the color code is used, if True.
        verbose : bool, default: False
            Print auxiliary information if True.

        Returns
        -------
        fig : Figure object
        ax : Axes object
        cb : Colorbar object (optional)
        """
        if FigAx is None:
            fig = plt.figure(figsize=(7, 7))
            ax  = plt.gca()
            FigAx = (fig, ax)
        #-> Plot each data
        for loop in range(self.__length):
            if loop > 0:
                legend_kws = None
            FigAx = self[loop].plot_uv(insname=insname, FigAx=FigAx, colorcode=colorcode,
                                       scatter_kws=deepcopy(scatter_kws), legend_kws=legend_kws,
                                       ignored_channels=ignored_channels, label_fontsize=label_fontsize,
                                       tick_labelsize=tick_labelsize, text_fontsize=text_fontsize,
                                       show_colorbar=show_colorbar, verbose=verbose)
        return FigAx

    def plot_visibility(self, insname="ft", visdata="vis2", FigAx=None, flagged=True,
                        errorbar_kws=None, legend_kws=None, ignored_channels=None,
                        label_fontsize=24, tick_labelsize=18, text_fontsize=16,
                        verbose=False):
        """
        Plot the visibility data (vis2, visamp, or visphi, as well as their errors).

        Parameters
        ----------
        insname : string
            The instrument name, "ft", "sc", or their polarization components.
        visdata : string
            The visibility data, vis2, visamp, or visphi.
        FigAx : tuple (optional)
            The Figure and Axes objects generated in prior.
        flagged : bool, default: True
            Plot the flagged data, if True.
        errorbar_kws : dict (optional)
            The keywords for plt.errorbar() function
        legend_kws : dict (optional)
            The keywords for plt.legend() function
        ignored_channels : list (optional)
            The list of channel indices (0~4 for FT and 0~209 for SC) ignored in
            the plot.
        label_fontsize : float, default: 24
            The fontsize of the labels of both axes.
        tick_labelsize : float, default: 18
            The fontsize of the ticklabel of both axes.
        verbose : bool, default: False
            Print auxiliary information if True.

        Returns
        -------
        fig : Figure object
        ax : Axes object
        """
        if FigAx is None:
            fig = plt.figure(figsize=(7, 7))
            ax  = plt.gca()
        else:
            fig, ax = FigAx
        #-> Plot each data
        for loop in range(self.__length):
            if loop > 0:
                legend_kws = None
            try:
                self[loop].plot_visibility(insname=insname, visdata=visdata, FigAx=(fig, ax),
                                           flagged=flagged, errorbar_kws=deepcopy(errorbar_kws),
                                           legend_kws=legend_kws, ignored_channels=ignored_channels,
                                           label_fontsize=label_fontsize, tick_labelsize=tick_labelsize,
                                           text_fontsize=text_fontsize, verbose=verbose)
            except:
                print("Cannot plot: {0} data!".format(self[loop].obsdate))
        return (fig, ax)

    def plot_t3(self, insname="ft", t3data="phi", FigAx=None, flagged=True, errorbar_kws=None,
                legend_kws=None, ignored_channels=None, label_fontsize=24, tick_labelsize=18,
                text_fontsize=16, verbose=False):
        """
        Plot the T3 data (t3amp or t3phi as well as their errors).

        Parameters
        ----------
        insname : string
            The instrument name, "ft", "sc", or their polarization components.
        visdata : string
            The visibility data, vis2, visamp, or visphi.
        FigAx : tuple (optional)
            The Figure and Axes objects generated in prior.
        flagged : bool, default: True
            Plot the flagged data, if True.
        errorbar_kws : dict (optional)
            The keywords for plt.errorbar() function
        legend_kws : dict (optional)
            The keywords for plt.legend() function
        ignored_channels : list (optional)
            The list of channel indices (0~4 for FT and 0~209 for SC) ignored in
            the plot.
        label_fontsize : float, default: 24
            The fontsize of the labels of both axes.
        tick_labelsize : float, default: 18
            The fontsize of the ticklabel of both axes.
        text_fontsize : float, default: 16
            The fontsize of the text in the figure.
        verbose : bool, default: False
            Print auxiliary information if True.

        Returns
        -------
        fig : Figure object
        ax : Axes object
        """
        if FigAx is None:
            fig = plt.figure(figsize=(7, 7))
            ax  = plt.gca()
        else:
            fig, ax = FigAx
        #-> Plot each data
        for loop in range(self.__length):
            if loop > 0:
                legend_kws = None
            try:
                self[loop].plot_t3(insname=insname, t3data=t3data, FigAx=(fig, ax),
                                   flagged=flagged, errorbar_kws=deepcopy(errorbar_kws),
                                   legend_kws=legend_kws, ignored_channels=ignored_channels,
                                   label_fontsize=label_fontsize, tick_labelsize=tick_labelsize,
                                   text_fontsize=text_fontsize, verbose=verbose)
            except:
                print("Cannot plot: {0} data!".format(self[loop].obsdate))
        return (fig, ax)

    def plot_strehl(self, visdata="vis2", channel=3, FigAx=None, errorbar_kws=None,
                    legend_kws=None, label_fontsize=24, tick_labelsize=18, text_fontsize=16,
                    verbose=False):
        """
        Plot the Strehl versus visibility of the FT data.

        Parameters
        ----------
        visdata : string
            The visibility data, vis2, visamp, or visphi.
        channel : float
            The number of the channel to look at.
        FigAx : tuple (optional)
            The Figure and Axes objects generated in prior.
        errorbar_kws : dict (optional)
            The keywords for errorbar() function.
        legend_kws : dict (optional)
            The keywords for plt.legend() function
        label_fontsize : float, default: 24
            The fontsize of the labels of both axes.
        tick_labelsize : float, default: 18
            The fontsize of the ticklabel of both axes.
        text_fontsize : float, default: 16
            The fontsize of the text.
        verbose : bool, default: False
            Print auxiliary information if True.

        Returns
        -------
        fig : Figure object
        ax : Axes object
        """
        if FigAx is None:
            fig = plt.figure(figsize=(7, 7))
            ax  = plt.gca()
        else:
            fig, ax = FigAx
        #-> Get the vis keyword
        visdata = visdata.upper()
        if visdata == "VIS2":
            viskw = "OI_VIS2:VIS2DATA"
            visekw = "OI_VIS2:VIS2ERR"
            ylabel = r"Vis.$^2$"
        elif visdata == "VISAMP":
            viskw = "OI_VIS:VISAMP"
            visekw = "OI_VIS:VISAMPERR"
            ylabel = "Vis. Amp."
        else:
            raise KeyError("Cannot recognize the visdata ({0})!".format(visdata))
        #-> Plot the data
        #--> Setup the plot keywords
        if errorbar_kws is None:
            errorbar_kws = {}
        kList = errorbar_kws.keys()
        if not "marker" in kList:
            errorbar_kws["marker"] = "."
        if not (("ls" in kList) or ("linestyle" in kList)):
            errorbar_kws["ls"] = "none"
        flag = 0
        for gd in self.gd_list:
            nbaseline = gd.get_dimlen("BASELINE")
            for bsl in range(nbaseline):
                bsl_tn = gd.baseline_tn(bsl)
                bsl_idx = gd.index_tn(bsl_tn)
                strehlList = gd.get_qc("qc_acq_strehl")[bsl_idx]
                if not "color" in kList:
                    errorbar_kws["color"] = "C{0}".format(bsl)
                x  = np.average(strehlList)
                xe = np.abs(strehlList[0] - strehlList[1]) / 2.
                y  = gd.get_data_flagged(viskw)[bsl, channel]
                ye = gd.get_data_flagged(visekw)[bsl, channel]
                if (flag < nbaseline) & (not "label" in kList):
                    errorbar_kws["label"] = "{0[0]}-{0[1]}".format(bsl_tn)
                    flag += 1
                else:
                    errorbar_kws["label"] = None
                ax.errorbar(x, y, xerr=xe, yerr=ye, **errorbar_kws)
        ax.text(0.95, 0.95, "Channel: {0}".format(channel), fontsize=text_fontsize,
                transform=ax.transAxes, horizontalalignment='right', verticalalignment='top')
        ax.set_xlabel("Strehl Ratio", fontsize=label_fontsize)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
        if legend_kws is None:
            ax.legend(loc="lower right", fontsize=text_fontsize, ncol=2, handletextpad=0.1,
                      columnspacing=0.2)
        else:
            ax.legend(**legend_kws)
        return (fig, ax)

    def plot_perobs(self, function, show_obsdate=True, text_fontsize=16,
                    sharex=True, sharey=True, **function_kwargs):
        """
        Plot the data of the individual nights separately.

        Parameters
        ----------
        function : string
            The name of the function to plot for the individual observation.
        show_obsdate : bool, default: True
            Show the obsdate in the title of each panel.
        text_fontsize : float, default: 16
            The fontsize of the text in the figure.
        sharex : bool, default: True
            Share the X axis for each panel.
        sharey : bool, default: True
            Share the Y axis for each panel.
        **function_kwargs : kwargs of the plot function.

        Returns
        -------
        fig : Figure object
        ax : array of Axes objects
        """
        ndat = self.__length
        nrow = np.ceil(np.sqrt(ndat))
        ncol = np.ceil(ndat / nrow)
        nrow = int(nrow)
        ncol = int(ncol)
        fig, axs = plt.subplots(nrow, ncol, sharex=sharex, sharey=sharey)
        fig.set_size_inches(5*ncol, 5*nrow)
        ncount = 0
        for loop_r in range(nrow):
            for loop_c in range(ncol):
                kwargs = deepcopy(function_kwargs)
                ax = axs[loop_r, loop_c]
                if ncount < ndat:
                    kwargs["FigAx"] = (fig, ax)
                    exec("self[ncount].{0}(**kwargs)".format(function))
                    if loop_r != (nrow-1):
                        ax.set_xlabel("")
                    if loop_c != 0:
                        ax.set_ylabel("")
                    if show_obsdate:
                        ax.set_title(self[ncount].obsdate, fontsize=text_fontsize)
                    ncount += 1
                else:
                    ax.axis('off')
        if sharex:
            plt.subplots_adjust(hspace=0.1)
        if sharey:
            plt.subplots_adjust(wspace=0)
        return (fig, axs)

    def get_Data_obsdate(self, obsdate, **kwargs):
        """
        Get the data according to the obsdate.

        Parameters
        ----------
        obsdate : datetime object or string (list)
            The observation date information to select the data.
                datetime object -> call get_Data_obsdate_single();
                y-m-dTh:m:s -> call get_Data_obsdate_single();
                y-m-d -> call get_Data_obsdate_range() for [y-m-dT12:00:00, y-m-(d+1)T12:00:00].
                [y1-m1-d1Th1:m1:s1, y2-m2-d2Th2:m2:s2] -> call get_Data_obsdate_range().
        **kwargs : Additional parameters for get_Data_obsdate_single or get_Data_obsdate_range.

        Returns
        -------
        Single or a list of GravityData.
        """
        #-> Manage the input
        #--> If obsdate is a string, first try to convert it to a datetime object.
        if type(obsdate) == type(datetime.datetime.now()):
            return self.get_Data_obsdate_single(obsdate, **kwargs)
        elif type(obsdate) == type("string"):
            try:
                obsdate_inp = datetime.datetime.strptime(obsdate, "%Y-%m-%dT%H:%M:%S")
                get_Data_func = self.get_Data_obsdate_single
            except:
                try:
                    date_list = obsdate.split("T")[0].split("-")
                    #---> This is following the convention of the our team.
                    obsdate_s = "{0}-{1}-{2}T12:00:00".format(date_list[0], date_list[1], date_list[2])
                    obsdate_s = datetime.datetime.strptime(obsdate_s, "%Y-%m-%dT%H:%M:%S")
                    obsdate_e = obsdate_s + datetime.timedelta(days=1)
                    obsdate_inp = [obsdate_s, obsdate_e]
                    get_Data_func = self.get_Data_obsdate_range
                except:
                    error_content = "The input obsdate ({0}) is not managable! The format %Y-%m-%d or %Y-%m-%dT%H:%M:%S is preferred."
                    raise ValueError(error_content.format(obsdate))
            return get_Data_func(obsdate_inp, **kwargs)
        #--> Else, if it is a list, it should contain the strings of the start and end obsdates.
        elif type(obsdate) == type(["obsdate_start", "obsdate_end"]):
            assert len(obsdate) == 2
            assert (type(obsdate[0]) == type("string")) & (type(obsdate[1]) == type("string"))
            try:
                obsdate_s = datetime.datetime.strptime(obsdate[0], "%Y-%m-%dT%H:%M:%S")
                obsdate_e = datetime.datetime.strptime(obsdate[1], "%Y-%m-%dT%H:%M:%S")
            except:
                raise ValueError("The date should be in %Y-%m-%dT%H:%M:%S rather than the input one ({0})!".format(obsdate))
            obsdate_inp = [obsdate_s, obsdate_e]
            return self.get_Data_obsdate_range(obsdate_inp, **kwargs)
        else:
            raise ValueError("The input obsdate ({0}) is not recognized!".format(obsdate))

    def get_Data_obsdate_range(self, obsdate_range, verbose=True):
        """
        Get the data observed within the obsdate range.

        Parameters
        ----------
        obsdate_range : list of two datetime.datetime
            The start and end of observation date.  The end obsdate is not included.
        verbose : bool, default: True

        Returns
        -------
        gdList : list
            A list of GravityData within the obsdate range.
        """
        assert len(obsdate_range) == 2
        obsdateList = np.array(self.get_DataSeries("obsdate", verbose=verbose))
        fltr = (obsdateList >= obsdate_range[0]) & (obsdateList < obsdate_range[1])
        gdList = []
        for obsdate in obsdateList[fltr]:
            gdList.append(self.get_Data_obsdate(obsdate, nearest=False, verbose=verbose))
        return GravitySet(gd_list=gdList)

    def get_Data_obsdate_single(self, obsdate, nearest=True, verbose=True):
        """
        Get the data of a given obsdate.

        Parameters
        ----------
        obsdate : datetime.datetime
            The observation date.
        nearest : bool, default: True
            Allow to return the nearest obsdate, if True.
        verbose : bool, default: True

        Returns
        -------
        gd : GravityData
        """
        obsdateList = self.get_DataSeries("obsdate", verbose=verbose)
        if nearest:
            idx = np.argmin(np.abs(np.array(obsdateList) - obsdate))
        else:
            idx = obsdateList.index(obsdate)
        gd = self.gd_list[idx]
        return gd

    def get_gd_list(self):
        """
        Get the GravityData list.
        """
        return self.gd_list

    def get_DataSeries_flagged(self, datakey, mask=None, insname="ft", obsdate=None,
                               verbose=True):
        """
        Get the time series of data identified by data_key.  The function only
        calls GravityData.get_data_flagged().

        Parameters
        ----------
        datakey : string
            The keyword of the data to be extracted.
        """
        if obsdate is None:
            gd_list = self.gd_list
        else:
            gd_list = self.get_Data_obsdate(obsdate)
        dataList = []
        for gd in gd_list:
            dataList.append(gd.get_data_flagged(datakey, mask=mask, insname=insname,
                                                verbose=verbose))
        return dataList

    def get_DataSeries(self, datakey="obsdate", insname=None, obsdate=None, verbose=True):
        """
        Get the time series of data identified by data_key.

        Parameters
        ----------
        datakey : string
            The keyword of the data to be extracted.
        insname : string (optional)
            The instrument name, only necessary when get_data() is used.
        obsdate : datetime object or string (list)
            Following what is used in get_Data_obsdate().
        verbose : bool, default: True
            Print notes if True.

        Returns
        -------
        dataList : list
            The list of data.
        """
        if obsdate is None:
            gd_list = self.gd_list
        else:
            gd_list = self.get_Data_obsdate(obsdate)
        dataList = []
        for gd in gd_list:
            if datakey == "obsdate":
                dataList.append(gd.obsdate)
            elif datakey in gd.get_qc_keys():
                dataList.append(gd.get_qc(datakey))
            else:
                assert not insname is None
                dataList.append(gd.get_data(datakey, insname=insname, verbose=verbose))
        return dataList

    def __getitem__(self, key):
        """
        Get the individual GravityData.
        """
        return self.gd_list[key]

    def __len__(self):
        """
        Get the length of the GravityData list.
        """
        return len(self.gd_list)


class GravityData(object):
    """
    The object of gravity data of a single observation.
    """
    def __init__(self, hdulist=None, filename=None, verbose=True):
        """
        Parameters
        ----------
        hdulist : dict (optional)
            The hdulist of data.  If hdulist is provided, the parameters to read
            from a fits file will be ignored.
        filename : string (optional)
            The fits file name.
        verbose : bool, default: True
            Print notes if True.
        """
        #-> Prior properties
        self.__catglist_vis = ["SINGLE_SCI_VIS", "SINGLE_SCI_VIS_CALIBRATED", "SINGLE_CAL_VIS",
                               "DUAL_SCI_VIS", "DUAL_SCI_VIS_CALIBRATED"]
        self.__catglist_p2vmred = ["SINGLE_SCI_P2VMRED", "DUAL_SCI_P2VMRED", "SINGLE_CAL_P2VMRED"]
        self.__polalist = ["P1", "P2"]
        self.__inslist = ["FT", "FT_P1", "FT_P2", "SC", "SC_P1", "SC_P2"]
        self.dim2 = {
            "OI_VIS" : gravity_DimLen("BASELINE"),
            "OI_VIS2": gravity_DimLen("BASELINE"),
            "OI_FLUX": gravity_DimLen("TELESCOPE"),
            "OI_T3": gravity_DimLen("TRIANGLE"),
        }
        #-> Read in the fits file hdulist is None
        if hdulist is None:
            assert not filename is None
            hdulist = fits.open(filename, mode='readonly')
        else:
            if verbose & (not (filename is None)):
                print("The hdulist is used so the other parameters are ignored!")
        #-> Information from the header
        header = hdulist[0].header
        self.header = header
        #--> Basical information
        self.catg = header.get("HIERARCH ESO PRO CATG", None)
        self.polamode = header.get("HIERARCH ESO INS POLA MODE", None)
        if not self.polamode in ["COMBINED", "SPLIT"]:
            raise ValueError("The polarization mode ({0}) is not recognized!".format(self.polamode))
        self.specres = header.get("HIERARCH ESO INS SPEC RES", None)
        if not self.specres in ["LOW", "MEDIUM", "HIGH"]:
            raise ValueError("The spectral resolution ({0}) is not recognized!".format(self.specres))
        self.obsdate = datetime.datetime.strptime(header["DATE-OBS"], '%Y-%m-%dT%H:%M:%S')
        self.object = header["OBJECT"]
        self.ra=header['RA']
        self.dec=header['DEC']
        #-> Get the data
        if self.catg in self.__catglist_vis:
            self.data = GravityVis(hdulist, verbose=verbose)
        elif self.catg in self.__catglist_p2vmred:
            self.data = GravityP2VMRED(hdulist, verbose=verbose)
        else:
            raise ValueError("The catg ({0}) is not supported!".format(self.catg))

    def plot_p2vmred_variability(self, insname="ft", time_interval=100, baseline=0, channel=2):
        """
        Plot the time variability of the P2VMRED data.

        Parameters
        ----------
        insname : string
            The instrument name, "ft", "sc", or their polarization components.
        time_interval : int, default: 100
            The interval to skip when plottinng the time series.
        baseline : int, default: 0
            The index of baseline dimension, 0~5 for Gravity data.
        channel : int, default: 2
            The index of channel dimension, 0~4 for Gravity data.

        Returns
        -------
        The Figure object and the list of four Axes objects.
        """
        assert insname.upper() in self.__inslist
        if (self.polamode == "SPLIT") & (insname in ["FT", "SC"]):
            raise KeyError("Only plot one polarization mode at a time!")
        time = self.get_data("oi_vis:time", insname=insname)
        gdly = self.get_data("oi_vis:gdelay", insname=insname) * 1e6
        gdlyb = self.get_data("oi_vis:gdelay_boot", insname=insname) * 1e6
        f1f2 = self.get_data("oi_vis:f1f2", insname=insname)
        snr = self.get_data("oi_vis:snr", insname=insname)
        snrb = self.get_data("oi_vis:snr_boot", insname=insname)
        visdata = self.get_data("oi_vis:visdata", insname=insname)
        visf1f2 = self.get_data("oi_vis:f1f2", insname=insname)
        visp2vm = (np.absolute(visdata)/np.sqrt(visf1f2))
        #-> Plot the data
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True) #plt.figure(figsize=(10, 3))
        fig.set_size_inches(16, 16)
        #--> Visamp
        x = time[::time_interval, baseline]
        y = visp2vm[::time_interval, baseline, channel]
        ax1.plot(x, y, ls="none", marker=".", color="k")
        ax1.set_ylabel("Vis. Amp.", fontsize=24)
        ax1.minorticks_on()
        ax1.tick_params(axis='both', which='major', labelsize=18)
        #--> GDELAY
        y = gdly[::time_interval, baseline]
        ax2.plot(x, y, ls="-", color="r", label="GDELAY")
        y = gdlyb[::time_interval, baseline]
        ax2.plot(x, y, ls="-", color="b", label="GDELAY_BOOT")
        ax2.set_ylabel(r"GDELAY ($\mu$m)", fontsize=24)
        ax2.legend(loc="upper left", fontsize=14)
        ax2.minorticks_on()
        ax2.tick_params(axis='both', which='major', labelsize=18)
        #--> F1F2
        y = f1f2[::time_interval, baseline, channel]
        ax3.plot(x, y, ls="-", color="r")
        ax3.set_ylabel("F1F2", fontsize=24)
        ax3.minorticks_on()
        ax3.tick_params(axis='both', which='major', labelsize=18)
        #--> SNR
        y = snr[::time_interval, baseline]
        ax4.plot(x, y, ls="-", color="r", label="SNR")
        y = snrb[::time_interval, baseline]
        ax4.plot(x, y, ls="-", color="b", label="SNR_BOOT")
        ax4.set_ylabel("SNR", fontsize=24)
        ax4.legend(loc="upper left", fontsize=14)
        ax4.minorticks_on()
        ax4.tick_params(axis='both', which='major', labelsize=18)
        ax4.set_xlabel("Time", fontsize=24)
        plt.subplots_adjust(hspace=0)
        return (fig, [ax1, ax2, ax3, ax4])

    def plot_uv(self, insname="ft", colorcode=None, FigAx=None, scatter_kws=None,
                legend_kws=None, ignored_channels=None, label_fontsize=24, tick_labelsize=18,
                text_fontsize=16, show_colorbar=False, verbose=False):
        """
        Plot the UV coverage of the data.

        Parameters
        ----------
        insname : string
            The instrument name, "ft", "sc", or their polarization components.
        colorcode : string (optional)
            The keyword of the data used for the color code, vis2 or visamp, or
            other keywords.  The data should have the same shape as the uv coordinates.
        FigAx : tuple (optional)
            The Figure, Axes and (optional) Colorbar objects generated in prior.
        scatter_kws : dict (optional)
            The keywords for plt.scatter() function
        legend_kws : dict (optional)
            The keywords for the plt.legend() function.  The label of the plot will
            not be provided if legend_kws is None.
        ignored_channels : list (optional)
            The list of channel indices (0~4 for FT and 0~209 for SC) ignored in
            the plot.
        label_fontsize : float, default: 24
            The fontsize of the labels of both axes.
        tick_labelsize : float, default: 18
            The fontsize of the ticklabel of both axes.
        text_fontsize : float, default: 16
            The fontsize of the text in the figure.
        show_colorbar : bool, default: False
            Show the colorbar when the color code is used, if True.
        verbose : bool, default: False
            Print auxiliary information if True.

        Returns
        -------
        fig : Figure object
        ax : Axes object
        cb : Colorbar object (optional)
        """
        insname = insname.upper()
        assert insname in self.__inslist
        if (self.polamode == "SPLIT") & (len(insname) == 2):
            for pk in self.__polalist:
                FigAx = self.plot_uv(insname="{0}_{1}".format(insname, pk), colorcode=colorcode,
                                     FigAx=FigAx, scatter_kws=deepcopy(scatter_kws),
                                     legend_kws=deepcopy(legend_kws), ignored_channels=ignored_channels,
                                     label_fontsize=label_fontsize, tick_labelsize=tick_labelsize,
                                     text_fontsize=text_fontsize, show_colorbar=show_colorbar,
                                     verbose=verbose)
                legend_kws = None
            return FigAx
        #-> For a single polarization situation
        assert self.catg in self.__catglist_vis
        #-> Get the data
        u, v = self.uv_mas(insname=insname, verbose=verbose)
        if colorcode is None:
            c = None
        else:
            colorcode = colorcode.upper()
            if colorcode in ["VIS2", "OI_VIS2:VIS2DATA"]:
                c = self.get_data("OI_VIS2:VIS2DATA", insname=insname, verbose=verbose)
                clabel = r"Vis.$^2$"
            elif colorcode in ["VISAMP", "OI_VIS:VISAMP"]:
                c = self.get_data("OI_VIS:VISAMP", insname=insname, verbose=verbose)
                clabel = "Vis. Amp."
            else:
                c = self.get_data(colorcode, insname=insname, verbose=verbose)
                clabel = colorcode
            assert u.shape == c.shape
        #--> Select channels
        if not ignored_channels is None:
            chnList = range(self.get_dimlen("CHANNEL_{0}".format(insname)))
            for cidx in ignored_channels:
                chnList.remove(cidx)
            u = u[:, chnList]
            v = v[:, chnList]
            if not c is None:
                c = c[:, chnList]
        #-> Plot
        if FigAx is None:
            fig = plt.figure(figsize=(7, 7))
            ax  = plt.gca()
            cb  = None
        else:
            nfa = len(FigAx)
            if nfa == 2:
                fig, ax = FigAx
                cax = None
                cb  = None
            elif nfa == 3:
                fig, ax, cax = FigAx
                cb = None
            elif nfa == 4:
                fig, ax, cax, cb = FigAx
            else:
                raise ValueError("The length of FigAx ({0}) is not correct!".format(nfa))
        if scatter_kws is None:
            scatter_kws = {}
        kList = scatter_kws.keys()
        if not "s" in kList:
            scatter_kws["s"] = 60
        for loop_b in range(self.get_dimlen("BASELINE")):
            if not legend_kws is None:
                scatter_kws["label"] = "{0[0]}-{0[1]}".format(self.baseline_tn(loop_b))
            if (not "c" in kList) & (not c is None):
                scatter_kws["c"] = [c[loop_b, :], c[loop_b, :]]
                if not "marker" in kList:
                    mList = ["o", "P", "X", "^", "d", "p"]
                    scatter_kws["marker"] = mList[loop_b]
                if not "vmin" in kList:
                    scatter_kws["vmin"] = np.min(c)
                    if colorcode in ["VIS2", "VISAMP", "OI_VIS2:VIS2DATA", "OI_VIS:VISAMP"]:
                        scatter_kws["vmin"] = np.max([scatter_kws["vmin"], 0])
                if not "vmax" in kList:
                    scatter_kws["vmax"] = np.max(c)
                    if colorcode in ["VIS2", "VISAMP", "OI_VIS2:VIS2DATA", "OI_VIS:VISAMP"]:
                        scatter_kws["vmax"] = np.min([scatter_kws["vmax"], 1])
                if not "edgecolor" in kList:
                    scatter_kws["edgecolor"] = "k"
                if not "linewidth" in kList:
                    scatter_kws["linewidth"] = 0.5
            elif not "color" in kList:
                scatter_kws["color"] = "C{0}".format(loop_b)
            x = u[loop_b, :]
            y = v[loop_b, :]
            im = ax.scatter([-x, x], [-y, y], **scatter_kws)
        ax.set_aspect('equal')
        ax.set_xlabel(r"U (mas$^{-1}$)", fontsize=label_fontsize)
        ax.set_ylabel(r"V (mas$^{-1}$)", fontsize=label_fontsize)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
        if show_colorbar & ("c" in scatter_kws.keys()):
            if cax is None:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
            if cb is None:
                cb = plt.colorbar(im, cax=cax)
                cb.set_label(clabel, fontsize=label_fontsize)
                cb.ax.minorticks_on()
                cb.ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
        #-> Legend
        if not legend_kws is None:
            if len(legend_kws.keys()) == 0:
                ax.legend(loc="lower right", fontsize=text_fontsize, ncol=2, handletextpad=0.1,
                          columnspacing=0.2)
            else:
                ax.legend(**legend_kws)
        if show_colorbar:
            FigAx = (fig, ax, cax, cb)
        else:
            FigAx = (fig, ax, im)
        return FigAx

    def plot_visibility(self, insname="ft", visdata="vis2", FigAx=None, flagged=True,
                        errorbar_kws=None, legend_kws=None, ignored_channels=None,
                        label_fontsize=24, tick_labelsize=18, text_fontsize=16,
                        verbose=False):
        """
        Plot the visibility data (vis2, visamp, or visphi, as well as their errors).

        Parameters
        ----------
        insname : string
            The instrument name, "ft", "sc", or their polarization components.
        visdata : string
            The visibility data, vis2, visamp, or visphi.
        FigAx : tuple (optional)
            The Figure and Axes objects generated in prior.
        flagged : bool, default: True
            Plot the flagged data, if True.
        errorbar_kws : dict (optional)
            The keywords for plt.errorbar() function
        legend_kws : dict (optional)
            The keywords for the plt.legend() function.  The label of the plot will
            not be provided if legend_kws is None.
        ignored_channels : list (optional)
            The list of channel indices (0~4 for FT and 0~209 for SC) ignored in
            the plot.
        label_fontsize : float, default: 24
            The fontsize of the labels of both axes.
        tick_labelsize : float, default: 18
            The fontsize of the ticklabel of both axes.
        text_fontsize : float, default: 16
            The fontsize of the text in the figure.
        verbose : bool, default: False
            Print auxiliary information if True.

        Returns
        -------
        fig : Figure object
        ax : Axes object
        """
        insname = insname.upper()
        assert insname in self.__inslist
        #-> If the split polarizations are required to be plotted together.
        if (self.polamode == "SPLIT") & (len(insname) == 2):
            for pk in self.__polalist:
                FigAx = self.plot_visibility(insname="{0}_{1}".format(insname, pk),
                                             visdata=visdata, FigAx=FigAx, flagged=flagged,
                                             errorbar_kws=deepcopy(errorbar_kws),
                                             legend_kws=deepcopy(legend_kws), ignored_channels=ignored_channels,
                                             label_fontsize=label_fontsize, tick_labelsize=tick_labelsize,
                                             text_fontsize=text_fontsize, verbose=verbose)
                legend_kws = None
            return FigAx
        #-> For a single polarization situation
        assert self.catg in self.__catglist_vis
        #-> Get the data
        ruv = self.ruv_mas(insname=insname, verbose=verbose)
        visdata = visdata.upper()
        if visdata == "VIS2":
            viskw = "OI_VIS2:VIS2DATA"
            visekw = "OI_VIS2:VIS2ERR"
            ylabel = r"Vis.$^2$"
        elif visdata == "VISAMP":
            viskw = "OI_VIS:VISAMP"
            visekw = "OI_VIS:VISAMPERR"
            ylabel = "Vis. Amp."
        elif visdata == "VISPHI":
            viskw = "OI_VIS:VISPHI"
            visekw = "OI_VIS:VISPHIERR"
            ylabel = "Phase (Degree)"
        else:
            raise KeyError("Cannot recognize the visdata ({0})!".format(visdata))
        if flagged:
            vis  = self.get_data_flagged(viskw, insname=insname, verbose=verbose)
            vise = self.get_data_flagged(visekw, insname=insname, verbose=verbose)
        else:
            vis  = self.get_data(viskw, insname=insname, verbose=verbose)
            vise = self.get_data(visekw, insname=insname, verbose=verbose)
        #--> Select channels
        if not ignored_channels is None:
            chnList = range(self.get_dimlen("CHANNEL_{0}".format(insname)))
            for cidx in ignored_channels:
                chnList.remove(cidx)
            ruv  = ruv[:, chnList]
            vis  = vis[:, chnList]
            vise = vise[:, chnList]
        #-> Plot
        if FigAx is None:
            fig = plt.figure(figsize=(7, 7))
            ax  = plt.gca()
        else:
            fig, ax = FigAx
        if errorbar_kws is None:
            errorbar_kws = {}
        kList = errorbar_kws.keys()
        if not "marker" in kList:
            errorbar_kws["marker"] = "o"
        if not (("ls" in kList) or ("linestyle" in kList)):
            errorbar_kws["ls"] = "none"
        for loop_b in range(self.get_dimlen("BASELINE")):
            if not "color" in kList:
                errorbar_kws["color"] = "C{0}".format(loop_b)
            if not (("mec" in kList) or ("markeredgecolor" in kList)):
                errorbar_kws["mec"] = errorbar_kws["color"]
            if not legend_kws is None:
                errorbar_kws["label"] = "{0[0]}-{0[1]}".format(self.baseline_tn(loop_b))
            x = ruv[loop_b, :].flatten()
            y = vis[loop_b, :].flatten()
            e = vise[loop_b, :].flatten()
            ax.errorbar(x, y, yerr=e, **errorbar_kws)
        ax.set_xlabel(r"$r_\mathrm{UV}$ (mas$^{-1})$", fontsize=label_fontsize)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
        #-> Legend
        if not legend_kws is None:
            if len(legend_kws.keys()) == 0:
                ax.legend(loc="lower right", fontsize=text_fontsize, ncol=2, handletextpad=0.1,
                          columnspacing=0.2)
            else:
                ax.legend(**legend_kws)
        return (fig, ax)

    def plot_t3(self, insname="ft", t3data="phi", FigAx=None, flagged=True, errorbar_kws=None,
                legend_kws=None, ignored_channels=None, label_fontsize=24, tick_labelsize=18,
                text_fontsize=16, verbose=False):
        """
        Plot the T3 data (t3amp or t3phi as well as their errors).

        Parameters
        ----------
        insname : string
            The instrument name, "ft", "sc", or their polarization components.
        visdata : string
            The visibility data, vis2, visamp, or visphi.
        FigAx : tuple (optional)
            The Figure and Axes objects generated in prior.
        flagged : bool, default: True
            Plot the flagged data, if True.
        errorbar_kws : dict (optional)
            The keywords for plt.errorbar() function
        legend_kws : dict (optional)
            The keywords for the plt.legend() function.  The label of the plot will
            not be provided if legend_kws is None.
        ignored_channels : list (optional)
            The list of channel indices (0~4 for FT and 0~209 for SC) ignored in
            the plot.
        label_fontsize : float, default: 24
            The fontsize of the labels of both axes.
        tick_labelsize : float, default: 18
            The fontsize of the ticklabel of both axes.
        text_fontsize : float, default: 16
            The fontsize of the text in the figure.
        verbose : bool, default: False
            Print auxiliary information if True.

        Returns
        -------
        fig : Figure object
        ax : Axes object
        """
        insname = insname.upper()
        assert insname in self.__inslist
        #-> If the split polarizations are required to be plotted together.
        if (self.polamode == "SPLIT") & (len(insname) == 2):
            for pk in self.__polalist:
                FigAx = self.plot_t3(insname="{0}_{1}".format(insname, pk), t3data=t3data,
                                     FigAx=FigAx, flagged=flagged, errorbar_kws=deepcopy(errorbar_kws),
                                     legend_kws=deepcopy(legend_kws), ignored_channels=ignored_channels,
                                     label_fontsize=label_fontsize, tick_labelsize=tick_labelsize,
                                     text_fontsize=text_fontsize, verbose=verbose)
                legend_kws = None
            return FigAx
        #-> For a single polarization situation
        assert self.catg in self.__catglist_vis
        #-> Get the data
        ruv3 = self.ruv3_mas(insname=insname, verbose=verbose)
        t3data = t3data.upper()
        if t3data == "PHI":
            t3kw  = "OI_T3:T3PHI"
            t3ekw = "OI_T3:T3PHIERR"
            ylabel = "Closure Phase (Degree)"
        elif t3data == "AMP":
            t3kw = "OI_T3:T3AMP"
            t3ekw = "OI_T3:T3AMPERR"
            ylabel = "Closure Amplitude"
        else:
            raise KeyError("Cannot recognize the t3data ({0})!".format(t3data))
        if flagged:
            t3  = self.get_data_flagged(t3kw, insname=insname, verbose=verbose)
            t3e = self.get_data_flagged(t3ekw, insname=insname, verbose=verbose)
        else:
            t3  = self.get_data(t3kw, insname=insname, verbose=verbose)
            t3e = self.get_data(t3ekw, insname=insname, verbose=verbose)
        #--> Select channels
        if not ignored_channels is None:
            chnList = range(self.get_dimlen("CHANNEL_{0}".format(insname)))
            for cidx in ignored_channels:
                chnList.remove(cidx)
            ruv3 = ruv3[:, chnList]
            t3   = t3[:, chnList]
            t3e  = t3e[:, chnList]
        #-> Plot
        if FigAx is None:
            fig = plt.figure(figsize=(7, 7))
            ax  = plt.gca()
        else:
            fig, ax = FigAx
        if errorbar_kws is None:
            errorbar_kws = {}
        kList = errorbar_kws.keys()
        if not "marker" in kList:
            errorbar_kws["marker"] = "o"
        if not (("ls" in kList) or ("linestyle" in kList)):
            errorbar_kws["ls"] = "none"
        for loop_t in range(self.get_dimlen("TRIANGLE")):
            if not "color" in kList:
                errorbar_kws["color"] = "C{0}".format(loop_t)
            if not (("mec" in kList) or ("markeredgecolor" in kList)):
                errorbar_kws["mec"] = errorbar_kws["color"]
            if not legend_kws is None:
                errorbar_kws["label"] = "{0[0]}-{0[1]}-{0[2]}".format(self.triangle_tn(loop_t))
            x = ruv3[loop_t, :].flatten()
            y = t3[loop_t, :].flatten()
            e = t3e[loop_t, :].flatten()
            ax.errorbar(x, y, yerr=e, **errorbar_kws)
        ax.set_xlabel(r"$r_\mathrm{UV, max}$ (mas$^{-1})$", fontsize=label_fontsize)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
        #-> Legend
        if not legend_kws is None:
            if len(legend_kws.keys()) == 0:
                ax.legend(loc="lower right", fontsize=text_fontsize, ncol=1, handletextpad=0.1)
            else:
                ax.legend(**legend_kws)
        return (fig, ax)

    def plot_p2vmred(self, insname="ft", xdata="oi_vis:gdelay", ydata="oi_vis:f1f2",
                     FigAx=None, flagged=True, mask=None, baseline=None, ignored_channels=None,
                     xperc=None, yperc=None, cperc=None, plot_kws=None, legend_kws=None,
                     label_fontsize=24, tick_labelsize=18, text_fontsize=16, point_limit=None,
                     verbose=False):
        """
        Plot the P2VMRED data.  The most relevant as far as I see is GDELAY--F1F2.

        Parameters
        ----------
        insname : string
            The instrument name, "ft", "sc", or their polarization components.
        xdata : string, default: "oi_vis:gdelay"
            The data on the x axis.
        ydata : string, default: "oi_vis:f1f2"
            The data on the y axis.
        FigAx : tuple (optional)
            The Figure and Axes objects generated in prior.
        flagged : bool, default: True
            Plot the flagged data, if True.
        mask : array_like (optional)
            The mask to flag the data.
        baseline : list (optional)
            The list of names of the telescope pair, e.g., ["UT4", "UT3"].
        ignored_channels : list (optional)
            The list of channel indices (0~4 for FT and 0~209 for SC) ignored in
            the plot.
        xperc : list (optional)
            The list to calculate the demarcation lines of percentiles for the xdata.
        yperc : list (optional)
            The list to calculate the demarcation lines of percentiles for the ydata.
        cperc : list (optional)
            The list of color names for the demarcation lines.
        plot_kws : dict (optional)
            The keywords for the plt.plot() function.
        legend_kws : dict (optional)
            The keywords for the plt.legend() function.  The label of the plot will
            not be provided if legend_kws is None.
        label_fontsize : float, default: 24
            The fontsize of the labels of both axes.
        tick_labelsize : float, default: 18
            The fontsize of the ticklabel of both axes.
        text_fontsize : float, default: 16
            The fontsize of the text in the figure.
        point_limit : int (optional)
            Control the number of plotted points.
        verbose : bool, default: False
            Print auxiliary information if True.

        Returns
        -------
        fig : Figure object
        ax : Axes object
        """
        insname = insname.upper()
        assert insname in self.__inslist
        #-> If the split polarizations are required to be plotted together.
        if (self.polamode == "SPLIT") & (len(insname) == 2):
            for pk in self.__polalist:
                FigAx = self.plot_p2vmred(insname="{0}_{1}".format(insname, pk),
                                          xdata=xdata, ydata=ydata, FigAx=FigAx,
                                          flagged=flagged, mask=mask, baseline=baseline,
                                          ignored_channels=ignored_channels, xperc=xperc,
                                          yperc=yperc, cperc=cperc, plot_kws=deepcopy(plot_kws),
                                          legend_kws=deepcopy(legend_kws), label_fontsize=label_fontsize,
                                          tick_labelsize=tick_labelsize, text_fontsize=text_fontsize,
                                          point_limit=point_limit, verbose=verbose)
                legend_kws = None
            return FigAx
        #-> For a single polarization situation
        assert self.catg in self.__catglist_p2vmred
        #-> Get data
        if flagged:
            x = self.get_data_flagged(xdata, mask=mask, insname=insname, verbose=verbose)
            y = self.get_data_flagged(ydata, mask=mask, insname=insname, verbose=verbose)
        else:
            x = self.get_data(xdata, insname=insname, verbose=verbose)
            y = self.get_data(ydata, insname=insname, verbose=verbose)
        #--> Select channels
        ndim = len(x.shape)
        if not ignored_channels is None:
            chnList = range(self.get_dimlen("CHANNEL_{0}".format(insname)))
            for cidx in ignored_channels:
                chnList.remove(cidx)
            if ndim == 2:
                x = x[:, chnList]
                y = y[:, chnList]
            if ndim == 3:
                x = x[:, :, chnList]
                y = y[:, :, chnList]
        #--> Select baseline
        if not baseline is None:
            bsl_idx = self.index_baseline(baseline, verbose=verbose)
            if bsl_idx is None:
                raise ValueError("The baseline ({0}) is not recognized!".format(baseline))
            if ndim == 2:
                x = x[:, bsl_idx]
                y = y[:, bsl_idx]
            elif ndim == 3:
                x = x[:, bsl_idx, :]
                y = y[:, bsl_idx, :]
            else:
                raise RuntimeError("The dimension of the data ({0}) is not compatible!".format(ndim))
        #--> Flatten the data
        if flagged:
            x = x[~x.mask]
            y = y[~y.mask]
        else:
            x = x.flatten()
            y = y.flatten()
        #-> Prepare plot
        if xdata == "oi_vis:gdelay":
            xlabel = r"|GDELAY| ($\mu$m)"
            x = np.abs(x) * 1e6 #units: micron
        else:
            xlabel = xdata
        if ydata == "oi_vis:f1f2":
            ylabel = "F1F2"
        else:
            ylabel = ydata
        #--> Simple statistics
        if xperc is None:
            xperc = [20, 50, 70, 90]
            try:
                xPercs = np.percentile(x, xperc)
            except:
                xPercs = []
        if yperc is None:
            yperc = [20, 50, 70, 90]
            try:
                yPercs = np.percentile(y, yperc)
            except:
                yPercs = []
        if cperc is None:
            ncolors = np.max([len(xPercs), len(yPercs)])
            if ncolors > 10:
                raise ValueError("There are too many lines to draw.  cperc should be specified!")
            cPercs = ["C{0}".format(c) for c in range(ncolors)]
        #-> Limit the number of data points
        if not point_limit is None:
            nd = len(x)
            assert nd == len(y)
            if nd > point_limit:
                idx = np.random.choice(range(nd), int(point_limit))
                x = np.array([x[i] for i in idx])
                y = np.array([y[i] for i in idx])
            else:
                if verbose:
                    print("The number of data ({0}) is below the point_limit ({1})!".format(nd, point_limit))
        #-> Plot
        if FigAx is None:
            fig = plt.figure(figsize=(7, 7))
            ax  = plt.gca()
        else:
            fig, ax = FigAx
        if plot_kws is None:
            plot_kws = {}
        kList = plot_kws.keys()
        if not "marker" in kList:
            plot_kws["marker"] = "."
        if not "color" in kList:
            plot_kws["color"] = "k"
        if not (("ls" in kList) or ("linestyle" in kList)):
            plot_kws["ls"] = "none"
        if not "alpha" in kList:
            plot_kws["alpha"] = 0.05
        ax.plot(x, y, **plot_kws)
        for loop_x in range(len(xPercs)):
            if legend_kws is None:
                label = None
            else:
                label = "{0}%".format(xperc[loop_x])
            ax.axvline(x=xPercs[loop_x], ymin=0.9, ymax=1, ls="--", lw=2, color=cPercs[loop_x],
                       label=label)
        for loop_y in range(len(yPercs)):
            if legend_kws is None:
                label = None
            else:
                label = "{0}%".format(yperc[loop_y])
            ax.axhline(y=yPercs[loop_y], xmin=0.9, xmax=1, ls=":", lw=2, color=cPercs[loop_y],
                       label=label)
        ax.set_xlabel(xlabel, fontsize=label_fontsize)
        ax.set_ylabel(ylabel, fontsize=label_fontsize)
        ax.set_yscale("log")
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=tick_labelsize)
        if not baseline is None:
            ax.set_title("-".join(baseline), fontsize=text_fontsize)
        if not legend_kws is None:
            if len(legend_kws.keys()) == 0:
                ax.legend(loc="lower right", fontsize=text_fontsize, ncol=2, handletextpad=0.1,
                        columnspacing=0.2)
            else:
                ax.legend(**legend_kws)
        return (fig, ax)

    def get_time_dit(self, insname="ft", unit_read="s"):
        """
        Get the integration time of one DIT of the data.

        Parameters
        ----------
        insname : string
            The instrument name, "ft" for fringe tracking or "sc" for science.
        unit_read : string, default: us
            The unit of the read-in data.  The default value of GRAVITY data is
            microsecond.

        Returns
        -------
        dit : Astropy Quantity
            The integration time of one DIT.
        """
        insname = "GRAVITY_{0}".format(insname.upper())
        if insname == "GRAVITY_SC":
            det_num = "DET2"
        elif insname == "GRAVITY_FT":
            det_num = "DET3"
        else:
            raise ValueError("Cannot recognize the insname ({0})!".format(insname))
        keyword = "HIERARCH ESO {0} SEQ1 DIT".format(det_num)
        dit = self.header.get(keyword) * u.Unit(unit_read)
        return dit

    def get_time_start(self, insname="ft", unit_read="us", verbose=False):
        """
        Get the start time of the data (first time of VISTIME).

        Parameters
        ----------
        insname : string
            The instrument name, "ft", "sc", or their polarization components.
        unit_read : string, default: us
            The unit of the read-in data.  The default value of GRAVITY data is
            microsecond.
        verbose : bool
            Print more information if True.

        Returns
        -------
        time_start : Astropy Quantity
            The start time of the data.
        """
        vis_time = self.get_data("OI_VIS:TIME", insname=insname, verbose=verbose)
        if self.catg in self.__catglist_p2vmred:
            time_start = vis_time[0, 0] * u.Unit(unit_read)
        else:
            raise ValueError("The start time for {0} is meaningless!".format(self.catg))
        return time_start

    def get_time_end(self, insname="ft", unit_read="us", verbose=False):
        """
        Get the end time of the data (last time of VISTIME).

        Parameters
        ----------
        insname : string
            The instrument name, "ft", "sc", or their polarization components.
        unit_read : string, default: us
            The unit of the read-in data.  The default value of GRAVITY data is
            microsecond.
        verbose : bool
            Print more information if True.

        Returns
        -------
        time_end : Astropy Quantity
            The end time of the data.
        """
        vis_time = self.get_data("OI_VIS:TIME", insname=insname, verbose=verbose)
        if self.catg in self.__catglist_p2vmred:
            time_end = vis_time[-1, 0] * u.Unit(unit_read)
        else:
            raise ValueError("The end time for {0} is meaningless!".format(self.catg))
        return time_end

    def ruv3_mas(self, insname="ft", verbose=False):
        """
        Calculate the uv distance for the triangle with units: milli-arcsec^-1.

        Parameters
        ----------
        insname : string
            The instrument name, "ft", "sc", or their polarization components.
        verbose : bool
            Print more information if True.

        Returns
        -------
        ruv3 : array
            The maximum UV distance of each triangle and channel, units: milli-arcsec^-1.
        """
        ruv_mas = self.ruv_mas(insname=insname, verbose=verbose)
        nt = self.get_dimlen("TRIANGLE")
        nc = self.get_dimlen("CHANNEL_{0}".format(insname))
        ruv3 = np.zeros([nt, nc])
        idxList = [(0, 1), (0, 2), (1, 2)] # The index of the telescope pairs of
                                           # the three baselines of the triangle.
        for loop_t in range(nt):
            trg_list = self.triangle_si(loop_t)
            ruv_t = []
            for (idx1, idx2) in idxList:
                si_list = [trg_list[idx1], trg_list[idx2]]
                tn_list = self.list_tn(self.index_si(si_list)) # Get the name of the telescope pair
                ruv_t.append(ruv_mas[self.index_baseline(tn_list), :])
            ruv_t = np.array(ruv_t)
            ruv3[loop_t, :] = np.max(ruv_t, axis=0)
        return ruv3

    def ruv_mas(self, insname="ft", verbose=False):
        """
        Calculate the uv distance with units: milli-arcsec^1.

        Parameters
        ----------
        insname : string
            The instrument name, "ft", "sc", or their polarization components.
        verbose : bool
            Print more information if True.

        Returns
        -------
        ruv : array
            The UV distance of each baseline and channel, units: milli-arcsec^-1.
        """
        u, v = self.uv_mas(insname=insname, verbose=verbose)
        ruv = np.sqrt(u**2 + v**2)
        return ruv

    def uv_mas(self, insname="ft", verbose=False):
        """
        Calculate the uv coordinates with units: milli-arcsec^1.

        Parameters
        ----------
        insname : string
            The instrument name, "ft", "sc", or their polarization components.
        verbose : bool
            Print more information if True.

        Returns
        -------
        (u_mas, v_mas) : tuple of arrays
            The UV coordinates of each baseline and channel, units: milli-arcsec^-1.
        """
        u = self.get_data_fulldim("OI_VIS:UCOORD", insname=insname, verbose=verbose)
        v = self.get_data_fulldim("OI_VIS:VCOORD", insname=insname, verbose=verbose)
        wavelength = self.get_data("OI_WAVELENGTH:EFF_WAVE", insname=insname, verbose=verbose)
        u_mas = u / wavelength / (180. / np.pi * 3.6e6)
        v_mas = v / wavelength / (180. / np.pi * 3.6e6)
        return (u_mas, v_mas)

    def list_tn(self, ti=None):
        """
        List the name of the telescopes according to "OI_ARRAY:TEL_NAME".

        Parameters
        ----------
        ti : (list of) int (optional)
            The index of the telescope, 0~3 for GRAVITY.

        Returns
        -------
        tel_list : list
            The list of telescope names.
        """
        tel_name = list(self.get_data("OI_ARRAY:TEL_NAME", insname="aux"))
        if ti is None:
            tel_list = tel_name
        else:
            tiList = np.atleast_1d(ti)
            tel_list = []
            for ti in tiList:
                tel_list.append(tel_name[ti])
        return tel_list

    def list_si(self, ti=None):
        """
        List the index of the stations according to "OI_ARRAY:STA_INDEX".

        Parameters
        ----------
        ti : (list of) int (optional)
            The index of the telescope, 0~3 for GRAVITY.

        Returns
        -------
        sta_list : list
            The list of station indices.
        """
        sta_index = list(self.get_data("OI_ARRAY:STA_INDEX", insname="aux"))
        if ti is None:
            sta_list = sta_index
        else:
            tiList = np.atleast_1d(ti)
            sta_list = []
            for ti in tiList:
                sta_list.append(sta_index[ti])
        return sta_list

    def index_tn(self, tn):
        """
        Get the telescope index from TEL_NAME.

        Parameters
        ----------
        tn : (list of) string
            The telescope name from "OI_ARRAY:TEL_NAME".

        Returns
        -------
        idx : int or list
            The index of the telescope, 0~3 for GRAVITY. If tn is a string of TEL_NAME,
            return an int of the index.  If si is a list of TEL_NAME, return
            a list of indices.
        """
        tel_name = self.list_tn()
        tnList = np.atleast_1d(tn)
        if len(tnList) == 1:
            idx = tel_name.index(tnList[0])
        else:
            idx = []
            for tn in tnList:
                idx.append(tel_name.index(tn))
        return idx

    def index_si(self, si):
        """
        Get the telescope index from STA_INDEX.

        Parameters
        ----------
        si : (list of) int
            The station index from "OI_ARRAY:STA_INDEX".

        Returns
        -------
        idx : int or list
            The index of the telescope, 0~3 for GRAVITY. If si is an int of STA_INDEX,
            return an int of the index.  If si is a list of STA_INDEX, return
            a list of indices.
        """
        sta_index = self.list_si()
        siList = np.atleast_1d(si)
        if len(siList) == 1:
            idx = sta_index.index(siList[0])
        else:
            idx = []
            for si in siList:
                idx.append(sta_index.index(si))
        return idx

    def baseline_si(self, bsl_index):
        """
        Get the station index of the baseline given the baseline index.

        Parameters
        ----------
        bsl_index : int
            The index of the baseline, 0-5.

        Returns
        -------
        bsl_si : list
            The list of station index of the baseline.
        """
        if self.polamode == "COMBINED":
            vis_station = self.get_data("OI_VIS:STA_INDEX", insname="sc") # SC and FT provides the same information.
        else:
            vis_station = self.get_data("OI_VIS:STA_INDEX", insname="sc_p1") # SC and FT provides the same information.
        if self.catg in self.__catglist_p2vmred:
            bsl_si = list(vis_station[0, bsl_index, :])
        elif self.catg in self.__catglist_vis:
            bsl_si = list(vis_station[bsl_index, :])
        else:
            raise ValueError("Self.catg ({0}) is not recognized!".format(self.catg))
        return bsl_si

    def baseline_tn(self, bsl_index):
        """
        Get the telescope name of the baseline given the baseline index.

        Parameters
        ----------
        bsl_index : int
            The index of the baseline, 0-5.

        Returns
        -------
        bsl_tn : list
            The list of telescope name of the baseline.
        """
        bsl_si = self.baseline_si(bsl_index)
        bsl_tn = self.list_tn(self.index_si(bsl_si))
        return bsl_tn

    def triangle_si(self, trg_index):
        """
        Get the station index of the triangle given the triangle index.

        Parameters
        ----------
        trg_index : int
            The index of the triangle, 0-3.

        Returns
        -------
        trg_tn : list
            The list of station index of the triangle.
        """
        assert self.catg in self.__catglist_vis
        if self.polamode == "COMBINED":
            t3_station = self.get_data("OI_T3:STA_INDEX", insname="sc") # SC and FT provides the same information.
        else:
            t3_station = self.get_data("OI_T3:STA_INDEX", insname="sc_p1") # SC and FT provides the same information.
        trg_si = list(t3_station[trg_index, :])
        return trg_si

    def triangle_tn(self, trg_index):
        """
        Get the telescope name of the triangle given the triangle index.

        Parameters
        ----------
        trg_index : int
            The index of the triangle, 0-3.

        Returns
        -------
        trg_tn : list
            The list of telescope name of the triangle.
        """
        trg_si = self.triangle_si(trg_index)
        trg_tn = self.list_tn(self.index_si(trg_si))
        return trg_tn

    def index_baseline(self, tn_list, verbose=False):
        """
        Get the index of the baseline dimension (0~5).

        Parameters
        ----------
        tn_list : list
            The list of two telescope names, e.g., ["UT4", "UT3"].
        verbose : bool, default: False
            Print auxiliary information if True.

        Returns
        -------
        bsl_index : int
            The index of the baseline dimension for the requested telescope pair.
        """
        assert len(tn_list) == 2
        si_list = np.sort(self.list_si(self.index_tn(tn_list)))
        bsl_index = None
        for loop_b in range(self.get_dimlen("BASELINE")):
            bsl_list = np.sort(self.baseline_si(loop_b))
            if (bsl_list == si_list).all():
                bsl_index = loop_b
                break
        if verbose & (bsl_index is None):
            print("The baseline ({0}) is not recognized!".format(tn_list))
        return bsl_index

    def index_triangle(self, tn_list, verbose=False):
        """
        Get the index of the triangle dimension (0~3).

        Parameters
        ----------
        tn_list : list
            The list of three telescope names, e.g., ["UT4", "UT3", "UT2"].
        verbose : bool, default: False
            Print auxiliary information if True.

        Returns
        -------
        trg_index : int
            The index of the triangle dimension for the requested telescope triangle.
        """
        assert len(tn_list) == 3
        si_list = np.sort(self.list_si(self.index_tn(tn_list)))
        trg_index = None
        for loop_t in range(self.get_dimlen("TRIANGLE")):
            trg_list = np.sort(self.triangle_si(loop_t))
            if (trg_list == si_list).all():
                trg_index = loop_t
                break
        if verbose & (trg_index is None):
            print("The triangle ({0}) is not recognized!".format(tn_list))
        return trg_index

    def get_info(self, verbose=True):
        """
        Get the information of the data structure.
        """
        insList = ["GRAVITY_SC", "GRAVITY_FT", "AUXILIARY"]
        extDict = {
            "ins_list": insList
        }
        for ins in insList:
            extDict[ins] = []
        hdulist = self.data.hdulist
        for hdu in hdulist:
            if hdu.name == "PRIMARY":
                continue
            ins = hdu.header.get("INSNAME", "AUXILIARY")
            colNameList = []
            if not hdu.is_image:
                for col in hdu.columns:
                    colNameList.append(col.name)
            extDict[ins].append((hdu.name, colNameList))
        if verbose:
            for ins in insList:
                print("[{0}]".format(ins))
                for hduTp in extDict[ins]:
                    print("**{0}:".format(hduTp[0]))
                    if len(hduTp[1]) != 0:
                        print("    {0}".format(", ".join(hduTp[1])))
                    else:
                        print("    [Image]")
                print("\n")
        return extDict

    def get_extension(self, keyword, insname="ft", multiple=False, verbose=False):
        """
        Get the hdu according to the keyword and the insname.

        Parameters
        ----------
        keyword : string
            The keyword of the extension, e.g., "OI_VIS", case free.
        insname : string
            The instrument name, "ft" for fringe tracking, "sc" for science (or
            their polarization mode), or "aux" for auxiliary data without INSNAME
            keyword.
        multiple : bool
            Allow the output list to have more than one extensions, if True.
        verbose : bool
            Print more information if True.

        Returns
        -------
        extList : list
            The list of hdu(s) matched with the keyword.
        """
        extList = self.data.get_extension(keyword, insname=insname, multiple=multiple,
                                          verbose=verbose)
        return extList

    def get_data(self, keyword, insname="ft", verbose=False):
        """
        Get the data identified by the keyword.

        Parameters
        ----------
        keyword : string
            The keyword of the data, "extName:datName", e.g., "OI_VIS:VISDATA".
            The keyword is case free.
        insname : string
            The instrument name, "ft" for fringe tracking, "sc" for science (or
            their polarization mode), or "aux" for auxiliary data without INSNAME
            keyword.
        verbose : bool
            Print more information if True.

        Returns
        -------
        data : array
            The data array or None if the keyword is not found.
        """
        data = self.data.get_data(keyword, insname=insname, verbose=verbose)
        return data

    def get_data_fulldim(self, keyword, insname="ft", verbose=False):
        """
        Get the data expanded in full dimention.

        Parameters
        ----------
        keyword : string
            The keyword of the data, "extName:datName", e.g., "OI_VIS:VISDATA".
            The keyword is case free.
        insname : string
            The instrument name, "ft", "sc", or their polarization components.
        verbose : bool
            Print more information if True.

        Returns
        -------
        data_array: array
            The data array expanded to its full dimension.
        """
        data_fulldim = self.data.get_data_fulldim(keyword, insname=insname, verbose=verbose)
        return data_fulldim

    def get_data_flagged(self, keyword, mask=None, insname="ft", verbose=False):
        """
        Get the masked data according to the flag.

        Parameters
        ----------
        keyword : string
            The keyword of the data, "extName:datName", e.g., "OI_VIS:VISDATA".
            The keyword is case free.
        mask : bool array (optional)
            The specified mask.  If None, the "REJECTION_FLAG" will be used to
            flag the data.
        insname : string
            The instrument name, "ft", "sc", or their polarization components.
        verbose : bool
            Print more information if True.

        Returns
        -------
        data_flagged: masked array
            The masked data array.
        """
        data_flagged = self.data.get_data_flagged(keyword, mask=mask, insname=insname,
                                                  verbose=verbose)
        return data_flagged

    def get_qc_keys(self):
        """
        Get the keywords of the qc data.
        """
        return self.data.qc_dict.keys()

    def get_qc(self, keyword):
        """
        Get the qc data identified by the keyword.

        Parameters
        ----------
        keyword : string
            The keyword of the qc data.

        Returns
        -------
        The qc data or None if the keyword is not found.
        """
        return self.data.get_qc(keyword)

    def get_hdulist(self):
        """
        Get the hdulist of the data.
        """
        return self.data.hdulist

    def update_data(self, keyword, newdata, insname="ft", verbose=False):
        """
        Update the data in the hdulist.

        Parameters
        ----------
        keyword : string
            The keyword of the data, "extName:datName", e.g., "OI_VIS:VISDATA".
            The keyword is case free.
        newdata : array
            The new data to be used.
        insname : string
            The instrument name, "ft" for fringe tracking, "sc" for science (or
            their polarization mode), or "aux" for auxiliary data without INSNAME
            keyword.
        verbose : bool
            Print more information if True.
        """
        if ":" in keyword:
            keyword = keyword.upper()
            extName, datName = keyword.split(":")
        else:
            raise ValueError("The keyword ({0}) is not recognized!".format(keyword))
        hduList = self.get_extension(extName, insname=insname, multiple=False, verbose=verbose)
        hdudata = hduList[0].data
        assert hdudata[datName].shape == newdata.shape
        hdudata[datName] = newdata

    def writeto(self, filename, **kwargs):
        """
        Write the hdulist of the data.
        """
        self.data.hdulist.writeto(filename, **kwargs)

    def filename(self):
       """
       Get the file name of the data hdulist.
       """
       return self.data.hdulist.filename()

    def get_dimlen(self, keyword):
        """
        Get the length of the given dimension.

        Parameters
        ----------
        keyword : string
            The keyword of the data dimension.
        """
        return gravity_DimLen(keyword, self.specres)

    def get_header(self, keyword, insname="aux", match=False, verbose=True):
        """
        Search the keyword in the header.
        """
        keyword = keyword.upper()
        if ":" in keyword:
            keyword = keyword.upper()
            extName, datName = keyword.split(":")
        else:
            extName, datName = ["PRIMARY", keyword]
        hduList = self.get_extension(extName, insname=insname, multiple=True, verbose=verbose)
        header_dict = {}
        #-> Go through all the hdu
        for hdu in hduList:
            header = hdu.header
            #-> Search for the keywords in the header
            for hk in header.keys():
                if match:
                    selc = datName == hk
                else:
                    selc = datName in hk
                if selc:
                    header_dict[hk] = header[hk]
        return header_dict


class GravityVis(object):
    """
    The object of gravity visibility data.
    """
    def __init__(self, hdulist=None, filename=None, verbose=True):
        """
        Parameters
        ----------
        hdulist : dict (optional)
            The hdulist of data.  If hdulist is provided, the parameters to read
            from a fits file will be ignored.
        filename : string (optional)
            The fits file name.
        verbose : bool, default: True
            Print notes if True.
        """
        #-> Prior properties
        self.__catglist = ["SINGLE_SCI_VIS", "SINGLE_SCI_VIS_CALIBRATED", "SINGLE_CAL_VIS",
                           "DUAL_SCI_VIS", "DUAL_SCI_VIS_CALIBRATED"]
        self.__inslist = ["FT", "FT_P1", "FT_P2", "SC", "SC_P1", "SC_P2", "AUX"]
        self.dim2 = {
            "OI_VIS" : gravity_DimLen("BASELINE"),
            "OI_VIS2": gravity_DimLen("BASELINE"),
            "OI_FLUX": gravity_DimLen("TELESCOPE"),
            "OI_T3": gravity_DimLen("TRIANGLE"),
        }
        self.datakey_list = ["OI_VIS", "OI_VIS2", "OI_T3", "OI_FLUX"]
        #-> Read in the fits file hdulist is None
        if hdulist is None:
            assert not filename is None
            hdulist = fits.open(filename, mode='readonly')
        else:
            if verbose & (not (filename is None)):
                print("The hdulist is used so the other parameters are ignored!")
        self.hdulist = hdulist
        #-> Information from the header
        header = hdulist[0].header
        self.header = header
        #--> Basical information
        self.catg = header.get("HIERARCH ESO PRO CATG", None)
        if verbose & (not self.catg in self.__catglist):
            print("The catg ({0}) has not been tested before!".format(self.catg))
        self.polamode = header.get("HIERARCH ESO INS POLA MODE", None)
        if not self.polamode in ["COMBINED", "SPLIT"]:
            raise ValueError("The polarization mode ({0}) is not recognized!".format(self.polamode))
        self.specres = header.get("HIERARCH ESO INS SPEC RES", None)
        if not self.specres in ["LOW", "MEDIUM", "HIGH"]:
            raise ValueError("The spectral resolution ({0}) is not recognized!".format(self.specres))
        self.obsdate = datetime.datetime.strptime(header["DATE-OBS"], "%Y-%m-%dT%H:%M:%S")
        self.object = header["OBJECT"]
        self.ra=header['RA']
        self.dec=header['DEC']
        #--> QC data from the header
        tel_code = ["1", "2", "3", "4"] # Telescopy code
        bsl_code = ["12", "13", "14", "23", "24", "34"] # Baseline code
        qc_acq_strehl = []
        qc_ft_rms = []
        qc_ft_tracking = []
        qc_ft_frames = []
        for kw_t in tel_code:
            qc_acq_strehl.append(header.get("HIERARCH ESO QC ACQ FIELD{0} STREHL".format(kw_t), np.nan))
        for kw_b in bsl_code:
            qc_ft_rms.append(header.get("HIERARCH ESO QC PHASE_FT{0} RMS".format(kw_b), np.nan))
            qc_ft_tracking.append(header.get("HIERARCH ESO QC TRACKING_RATIO_FT{0}".format(kw_b), np.nan))
            qc_ft_frames.append(header.get("HIERARCH ESO QC ACCEPTED_RATIO_SC{0}_P1".format(kw_b), np.nan))
        self.qc_dict = {
            "qc_acq_strehl": np.array(qc_acq_strehl),
            "qc_ft_rms": np.array(qc_ft_rms),
            "qc_ft_tracking": np.array(qc_ft_tracking),
            "qc_ft_frames": np.array(qc_ft_frames),
            "ambi_fwhm": [header.get("HIERARCH ESO ISS AMBI FWHM START", np.nan),
                          header.get("HIERARCH ESO ISS AMBI FWHM END", np.nan)],
            "ambi_tau0": [header.get("HIERARCH ESO ISS AMBI TAU0 START", np.nan),
                          header.get("HIERARCH ESO ISS AMBI TAU0 END", np.nan)],
         }

    def get_extension(self, keyword, insname="ft", multiple=False, verbose=False):
        """
        Get the hdu according to the keyword and the insname.

        Parameters
        ----------
        keyword : string
            The keyword of the extension, e.g., "OI_VIS", case free.
        insname : string
            The instrument name, "ft" for fringe tracking, "sc" for science (or
            their polarization mode), or "aux" for auxiliary data without INSNAME
            keyword.
        multiple : bool
            Allow the output list to have more than one extensions, if True.
        verbose : bool
            Print more information if True.

        Returns
        -------
        extList : list
            The list of hdu(s) matched with the keyword.
        """
        hdulist = self.hdulist
        keyword = keyword.upper()
        insname = insname.upper()
        if not insname in self.__inslist:
            errortext = "The insname ({0}) is incorrect!".format(insname)
            raise KeyError(errortext)
        insname = "GRAVITY_{0}".format(insname)
        extList = []
        for hdu in hdulist:
            #--> Use INSNAME to determine the data to include
            if (hdu.name == keyword) & (insname in str(hdu.header.get('INSNAME', "GRAVITY_AUX"))):
                extList.append(hdu)
        extCount = len(extList)
        if extCount == 0:
            raise KeyError("The extension ({0}) is not found!".format(keyword))
        elif (extCount > 1) & (not multiple):
            raise KeyError("There are {0} same extensions {1} for {2}!".format(extCount, keyword, insname))
        return extList

    def get_data(self, keyword, insname="ft", verbose=False):
        """
        Get a copy of the data identified by the keyword.

        Parameters
        ----------
        keyword : string
            The keyword of the data, "extName:datName", e.g., "OI_VIS:VISDATA".
            The keyword is case free.
        insname : string
            The instrument name, "ft" for fringe tracking, "sc" for science (or
            their polarization mode), or "aux" for auxiliary data without INSNAME
            keyword.
        verbose : bool
            Print more information if True.

        Returns
        -------
        data : array
            The data array or None if the keyword is not found.
        """
        if ":" in keyword:
            keyword = keyword.upper()
            extName, datName = keyword.split(":")
        else:
            raise ValueError("The keyword ({0}) is not recognized!".format(keyword))
        hduList = self.get_extension(extName, insname=insname, multiple=False, verbose=verbose)
        try:
            data = hduList[0].data[datName].copy()
        except:
            if verbose:
                print("The data ({0}) is not found in the extension {0}.".format(datName, extName))
            return None
        return data

    def get_data_fulldim(self, keyword, insname="ft", verbose=False):
        """
        Get the data expanded in full dimention (baseline/telescope/triangle, channel).

        keyword : string
            The keyword of the data, "extName:datName", e.g., "OI_VIS:VISDATA".
            The keyword is case free.
        insname : string
            The instrument name, "ft", "sc", or their polarization components.
        verbose : bool
            Print more information if True.

        Returns
        -------
        data_array: array
            The data array expanded to its full dimension.
        """
        if ":" in keyword:
            keyword = keyword.upper()
            extName, datName = keyword.split(":")
        else:
            raise ValueError("The keyword ({0}) is not recognized!".format(keyword))
        if not extName in self.datakey_list:
            raise ValueError("The extension ({0}) is not supported to be expanded!".format(extName))
        else:
            ndim2 = self.dim2[extName]
        insname = insname.upper()
        if insname in self.__inslist[:-1]:
            nchn = self.get_dimlen("CHANNEL_{0}".format(insname))
        else:
            raise ValueError("Cannot recognize insname ({0})!".format(insname))
        data_array = self.get_data(keyword, insname=insname, verbose=verbose)
        if data_array is None:
            raise ValueError("The keyword ({0}) is not recognized!".format(keyword))
        dshape = data_array.shape
        if len(dshape) == 2:
            if dshape[1] == nchn:
                data_array_full = data_array
            else:
                raise ValueError("The second dimension is not channel!")
        else:
            if (len(dshape) == 1) & (dshape[0] == ndim2):
                data_array_full = np.zeros((dshape[0], nchn), dtype=np.float)
                for loop in range(nchn):
                    data_array_full[:, loop] = data_array
            else:
                raise ValueError("The shape of {0} ({1}) is not correct ({2},)!".format(keyword, dshape, ndim2))
        return data_array_full

    def get_data_flagged(self, keyword, mask=None, insname="ft", verbose=False):
        """
        Get the masked data according to the flag in the same extension of the data.

        Parameters
        ----------
        keyword : string
            The keyword of the data, "extName:datName", e.g., "OI_VIS:VISDATA".
            The keyword is case free.
        mask : bool array (optional)
            The specified mask.  If None, the "REJECTION_FLAG" will be used to
            flag the data.
        insname : string
            The instrument name, "ft", "sc", or their polarization components.
        verbose : bool
            Print more information if True.

        Returns
        -------
        data_flagged: masked array
            The masked data array.
        """
        data_array = self.get_data_fulldim(keyword, insname=insname, verbose=verbose)
        extName, datName = keyword.split(":")
        if mask is None:
            mask = self.get_data("{0}:flag".format(extName), insname=insname, verbose=verbose)
        else:
            assert data_array.shape == mask.shape
        data_flagged = np.ma.array(data_array, mask=mask)
        return data_flagged

    def get_qc(self, keyword):
        """
        Get the qc data identified by the keyword.

        Parameters
        ----------
        keyword : string
            The keyword of the qc data.

        Returns
        -------
        The qc data or None if the keyword is not found.
        """
        return self.qc_dict.get(keyword, None)

    def get_dimlen(self, keyword):
        """
        Get the length of the given dimension.

        Parameters
        ----------
        keyword : string
            The keyword of the data dimension.
        """
        return gravity_DimLen(keyword, self.specres)


class GravityP2VMRED(object):
    """
    The object of gravity p2vmred data.
    """
    def __init__(self, hdulist=None, filename=None, verbose=True):
        """
        Parameters
        ----------
        hdulist : dict (optional)
            The hdulist of data.  If hdulist is provided, the parameters to read
            from a fits file will be ignored.
        filename : string (optional)
            The fits file name.
        verbose : bool, default: True
            Print notes if True.
        """
        #-> Prior properties
        self.__catglist = ["SINGLE_SCI_P2VMRED", "DUAL_SCI_P2VMRED", "SINGLE_CAL_P2VMRED"]
        self.__inslist = ["FT", "FT_P1", "FT_P2", "SC", "SC_P1", "SC_P2", "AUX"]
        self.dim2 = {
            "OI_VIS" : gravity_DimLen("BASELINE"),
            "OI_VIS2": gravity_DimLen("BASELINE"),
            "OI_FLUX": gravity_DimLen("TELESCOPE"),
            "OI_T3": gravity_DimLen("TRIANGLE"),
        }
        self.datakey_list = ["OI_VIS", "OI_VIS2", "OI_T3", "OI_FLUX"]
        #-> Read in the fits file hdulist is None
        if hdulist is None:
            assert not filename is None
            hdulist = fits.open(filename, mode='readonly')
        else:
            if verbose & (not (filename is None)):
                print("The hdulist is used so the other parameters are ignored!")
        self.hdulist = hdulist
        #-> Information from the header
        header = hdulist[0].header
        self.header = header
        #--> Basical information
        self.catg = header.get("HIERARCH ESO PRO CATG", None)
        if verbose & (not self.catg in self.__catglist):
            print("The catg ({0}) has not been tested before!".format(self.catg))
        self.polamode = header.get("HIERARCH ESO INS POLA MODE", None)
        if not self.polamode in ["COMBINED", "SPLIT"]:
            raise ValueError("The polarization mode ({0}) is not recognized!".format(self.polamode))
        self.specres = header.get("HIERARCH ESO INS SPEC RES", None)
        if not self.specres in ["LOW", "MEDIUM", "HIGH"]:
            raise ValueError("The spectral resolution ({0}) is not recognized!".format(self.specres))
        self.obsdate = datetime.datetime.strptime(header["DATE-OBS"], "%Y-%m-%dT%H:%M:%S")
        self.object = header["OBJECT"]
        self.ra=header['RA']
        self.dec=header['DEC']
        #--> QC data from the header
        tel_code = ["1", "2", "3", "4"] # Telescopy code
        bsl_code = ["12", "13", "14", "23", "24", "34"] # Baseline code
        qc_acq_strehl = []
        qc_tau0 = []
        for kw_t in tel_code:
            qc_acq_strehl.append(header.get("HIERARCH ESO QC ACQ FIELD{0} STREHL".format(kw_t), np.nan))
        for kw_b in bsl_code:
            qc_tau0.append(header.get("HIERARCH ESO QC TAU0 OPDC{0}".format(kw_b), np.nan))
        self.qc_dict = {
            "qc_tau0": np.array(qc_tau0),
            "qc_acq_strehl": np.array(qc_acq_strehl),
            "ambi_fwhm": [header.get("HIERARCH ESO ISS AMBI FWHM START", np.nan),
                          header.get("HIERARCH ESO ISS AMBI FWHM END", np.nan)],
            "ambi_tau0": [header.get("HIERARCH ESO ISS AMBI TAU0 START", np.nan),
                          header.get("HIERARCH ESO ISS AMBI TAU0 END", np.nan)],
         }

    def get_extension(self, keyword, insname="ft", multiple=False, verbose=False):
        """
        Get the hdu according to the keyword and the insname.

        Parameters
        ----------
        keyword : string
            The keyword of the extension, e.g., "OI_VIS", case free.
        insname : string
            The instrument name, "ft" for fringe tracking, "sc" for science, or
            "aux" for auxiliary data without INSNAME keyword.
        multiple : bool
            Allow the output list to have more than one extensions, if True.
        verbose : bool
            Print more information if True.

        Returns
        -------
        extList : list
            The list of hdu(s) matched with the keyword.
        """
        hdulist = self.hdulist
        keyword = keyword.upper()
        insname = insname.upper()
        if not insname in self.__inslist:
            errortext = "The insname ({0}) is incorrect!".format(insname)
            raise KeyError(errortext)
        insname = "GRAVITY_{0}".format(insname)
        extList = []
        for hdu in hdulist:
            #--> Use INSNAME to determine the data to include
            if (hdu.name == keyword) & (insname in str(hdu.header.get('INSNAME', "GRAVITY_AUX"))):
                extList.append(hdu)
        extCount = len(extList)
        if extCount == 0:
            raise KeyError("The extension ({0}) is not found!".format(keyword))
        elif (extCount > 1) & (not multiple):
            raise KeyError("There are {0} same extensions {1} for {2}!".format(extCount, keyword, insname))
        return extList

    def get_data(self, keyword, insname="ft", verbose=False):
        """
        Get the data identified by the keyword.

        Parameters
        ----------
        keyword : string
            The keyword of the data, "extName:datName", e.g., "OI_VIS:VISDATA".
            The keyword is case free.
        insname : string
            The instrument name, "ft" for fringe tracking, "sc" for science, or
            "aux" for auxiliary data without INSNAME keyword.
        verbose : bool
            Print more information if True.

        Returns
        -------
        data : array
            The data array or None if the keyword is not found.
        """
        if ":" in keyword:
            keyword = keyword.upper()
            extName, datName = keyword.split(":")
        else:
            raise ValueError("The keyword ({0}) is not recognized!".format(keyword))
        hduList = self.get_extension(extName, insname=insname, multiple=False, verbose=verbose)
        try:
            data = hduList[0].data[datName].copy()
        except:
            if verbose:
                print("The data ({0}) is not found in the extension {0}.".format(datName, extName))
            return None
        dim2 = self.dim2.get(extName, None)
        if not dim2 is None:
            ndim = len(data.shape)
            if ndim == 1:
                data = data.reshape(-1, dim2)
            elif ndim == 2:
                data = data.reshape(-1, dim2, data.shape[1])
            else:
                raise ValueError("The shape of {0}:{1} ({2}) is not managable!".format(extName, datName, data.shape))
        return data

    def get_data_fulldim(self, keyword, insname="ft", verbose=False):
        """
        Get the data expanded in full dimention (time, baseline/telescope/triangle, channel).

        keyword : string
            The keyword of the data, "extName:datName", e.g., "OI_VIS:VISDATA".
            The keyword is case free.
        insname : string
            The instrument name, "ft", "sc", or their polarization components.
        verbose : bool
            Print more information if True.

        Returns
        -------
        data_array: array
            The data array expanded to its full dimension.
        """
        if ":" in keyword:
            keyword = keyword.upper()
            extName, datName = keyword.split(":")
        else:
            raise ValueError("The keyword ({0}) is not recognized!".format(keyword))
        if not extName in self.datakey_list:
            raise ValueError("The extension ({0}) is not supported to be expanded!".format(extName))
        else:
            ndim2 = self.dim2[extName]
        insname = insname.upper()
        if insname in self.__inslist[:-1]:
            nchn = self.get_dimlen("CHANNEL_{0}".format(insname))
        else:
            raise ValueError("Cannot recognize insname ({0})!".format(insname))
        data_array = self.get_data(keyword, insname=insname, verbose=verbose)
        if data_array is None:
            raise ValueError("The keyword ({0}) is not recognized!".format(keyword))
        dshape = data_array.shape
        if len(dshape) == 3: # The 0th dimension of P2VMRED data is not fixed.
            if dshape[2] == nchn:
                data_array_full = data_array
            else:
                raise ValueError("The third dimension is not channel!")
        else:
            if (len(dshape) == 2) & (dshape[1] == ndim2):
                data_array_full = np.zeros((dshape[0], dshape[1], nchn), dtype=np.float)
                for loop in range(nchn):
                    data_array_full[:, :, loop] = data_array
            else:
                raise ValueError("The shape of {0} ({1}) is not correct ({2})!".format(keyword, dshape, (-1, ndim2)))
        return data_array_full

    def get_data_flagged(self, keyword, mask=None, insname="ft", verbose=False):
        """
        Get the masked data according to the rejection flag.

        Parameters
        ----------
        keyword : string
            The keyword of the data, "extName:datName", e.g., "OI_VIS:VISDATA".
            The keyword is case free.
        mask : bool array (optional)
            The specified mask.  If None, the "REJECTION_FLAG" will be used to
            flag the data.
        insname : string
            The instrument name, "ft", "sc", or their polarization components.
        verbose : bool
            Print more information if True.

        Returns
        -------
        data_flagged: masked array
            The masked data array.
        """
        data_array = self.get_data_fulldim(keyword, insname=insname, verbose=verbose) # Error will be raised if the keyword is wrong.
        if mask is None:
            mask = self.get_data_fulldim("oi_vis:rejection_flag", insname=insname, verbose=verbose) > 0
        else:
            assert data_array.shape == mask.shape
        data_flagged = np.ma.array(data_array, mask=mask)
        return data_flagged

    def get_qc(self, keyword):
        """
        Get the qc data identified by the keyword.

        Parameters
        ----------
        keyword : string
            The keyword of the qc data.

        Returns
        -------
        The qc data or None if the keyword is not found.
        """
        return self.qc_dict.get(keyword, None)

    def get_dimlen(self, keyword):
        """
        Get the length of the given dimension.

        Parameters
        ----------
        keyword : string
            The keyword of the data dimension.
        """
        return gravity_DimLen(keyword, self.specres)


def gravity_DimLen(keyword, specres=None):
    """
    Get the dimension length of the GRAVITY data.

    Parameters
    ----------
    keyword : string
        The keyword of the data dimension.
    specres : string (optional)
        The mode of resolution only works for CHANNEL_SC data.
    """
    dimDict = {
        "BASELINE": 6,
        "TELESCOPE": 4,
        "TRIANGLE": 4,
        "CHANNEL_FT": 5,
        "CHANNEL_SC_LOW": 14,
        "CHANNEL_SC_MEDIUM": 210,
        "CHANNEL_SC_HIGH": 1800, # Not accurate!!!
        "OI_VIS:STA_INDEX": 2,
        "OI_VIS2:STA_INDEX": 2,
        "OI_FLUX:STA_INDEX": 1,
    }
    keyword = keyword.upper()
    #-> Test : connection
    kwsList = keyword.split(":")
    if len(kwsList) > 1:
        return dimDict[keyword]
    #-> Test _ connection
    kwsList = keyword.split("_")
    if len(kwsList) > 3:
        raise KeyError("The keyword ({0}) is not recognized!".format(keyword))
    elif len(kwsList) == 3:
        keyword = "_".join(kwsList[:-1])
    elif len(kwsList) == 1:
        return dimDict[keyword]
    #-> Deal with *_* situation
    kwsList = keyword.split("_")
    if kwsList[1] != "SC":
        return dimDict[keyword]
    else:
        if not specres is None:
            keyword = "{0}_{1}".format(keyword, specres.upper())
        return dimDict[keyword]
