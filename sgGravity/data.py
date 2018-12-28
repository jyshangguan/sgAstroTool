import datetime
import numpy as np
from astropy.io import fits
from astropy import units as u

__all__ = ["GravitySet", "GravityData", "GravityVis", "GravityP2VMRED", "readfits_ins"]

class GravitySet(object):
    """
    A set of gravity data observed at different times.
    """
    def __init__(self, gd_list=None, file_list=None, insname=None, dataList=None,
                 verbose=True):
        """
        Parameters
        ----------
        gd_list : list (optional)
            The list of GravityData.  If it is provided, the parameters to read
            from fits files will be ignored.
        file_list : list (optional)
            The list of fits file names.
        insname : string (optional)
            The keyword INSNAME that is used to select the data, case free.
        dataList : list (optional)
            The list of data keywords to read.
        verbose : bool, default: True
            Print notes if True.
        """
        if gd_list is None:
            assert not file_list is None
            assert not insname is None
            gd_list = []
            for filename in file_list:
                gd_list.append(GravityData(filename=filename, insname=insname, dataList=dataList))
        else:
            if verbose & (not ((file_list is None) & (insname is None) & (dataList is None))):
                print("The gd_list is used so the other parameters are ignored!")
        self.gd_list = gd_list

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
                    obsdate_e = "{0}-{1}-{2}T12:00:00".format(date_list[0], date_list[1], (eval(date_list[2])+1))
                    obsdate_s = datetime.datetime.strptime(obsdate_s, "%Y-%m-%dT%H:%M:%S")
                    obsdate_e = datetime.datetime.strptime(obsdate_e, "%Y-%m-%dT%H:%M:%S")
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

    def get_DataSeries_flagged(self, datakey="obsdate", obsdate=None, verbose=True, **kwargs):
        """
        Get the time series of data identified by data_key.
        """
        if obsdate is None:
            gd_list = self.gd_list
        else:
            gd_list = self.get_Data_obsdate(obsdate)
        dataList = []
        for gd in gd_list:
            if datakey == "obsdate":
                if verbose:
                    print("No flag is applied (unless I implement the time flag)!")
                dataList.append(gd.obsdate)
            elif datakey in gd.get_data_keys():
                dataList.append(gd.get_data_flagged(datakey, **kwargs))
            elif datakey in gd.get_qc_keys():
                if verbose:
                    print("No flag is applied (unless I implement the time flag)!")
                dataList.append(gd.get_qc(datakey, **kwargs))
            else:
                raise ValueError("Cannot find the data ({0})!".format(datakey))
        return dataList

    def get_DataSeries(self, datakey="obsdate", obsdate=None, verbose=True):
        """
        Get the time series of data identified by data_key.
        """
        if obsdate is None:
            gd_list = self.gd_list
        else:
            gd_list = self.get_Data_obsdate(obsdate)
        dataList = []
        for gd in gd_list:
            if datakey == "obsdate":
                dataList.append(gd.obsdate)
            elif datakey in gd.get_data_keys():
                dataList.append(gd.get_data(datakey))
            elif datakey in gd.get_qc_keys():
                dataList.append(gd.get_qc(datakey))
            else:
                raise ValueError("Cannot find the data ({0})!".format(datakey))
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
    def __init__(self, data_dict=None, filename=None, insname=None, dataList=None,
                 verbose=True):
        """
        Parameters
        ----------
        data_dict : dict (optional)
            The dictionary of data obtained from readfits_ins().  If data_dict is
            provided, the parameters to read from a fits file will be ignored.
        filename : string (optional)
            The fits file name.
        insname : string (optional)
            The keyword INSNAME that is used to select the data, case free.
        dataList : list (optional)
            The list of data keywords to read.
        verbose : bool, default: True
            Print notes if True.
        """
        #-> Prior properties
        self.__catglist_vis = ["SINGLE_SCI_VIS", "SINGLE_SCI_VIS_CALIBRATED", "SINGLE_CAL_VIS",
                               "DUAL_SCI_VIS"]
        self.__catglist_p2vmred = ["SINGLE_SCI_P2VMRED", "DUAL_SCI_P2VMRED"]
        self.ndim_ft = {
            "vis" : (1, 6, 5),
            "vis2": (1, 6, 5),
            "flux": (1, 4, 5),
            "t3"  : (1, 4, 5),
        }
        self.ndim_sc = {
            "vis" : (1, 6, 210),
            "vis2": (1, 6, 210),
            "flux": (1, 4, 210),
            "t3"  : (1, 4, 210),
        }
        #-> Read in the fits file data_dict is None
        if data_dict is None:
            assert not filename is None
            assert not insname is None
            data_dict = readfits_ins(filename, insname, dataList)
        else:
            if verbose & (not ((filename is None) & (insname is None) & (dataList is None))):
                print("The data_dict is used so the other parameters are ignored!")
        #-> Information from the header
        header = data_dict["HEADER"]
        self.header = header
        #--> Basical information
        self.catg = header.get("HIERARCH ESO PRO CATG", None)
        self.insname = data_dict["INSNAME"]
        self.obsdate = datetime.datetime.strptime(header["DATE-OBS"], '%Y-%m-%dT%H:%M:%S')
        self.object = header["OBJECT"]
        self.ra=header['RA']
        self.dec=header['DEC']
        #-> Get the data
        if self.catg in self.__catglist_vis:
            self.data = GravityVis(data_dict, verbose=verbose)
        elif self.catg in self.__catglist_p2vmred:
            self.data = GravityP2VMRED(data_dict, verbose=verbose)
        else:
            raise ValueError("The catg ({0}) is not supported!".format(self.catg))

    def get_time_dit(self, unit_read="s"):
        """
        Get the integration time of one DIT of the data.

        Parameters
        ----------
        unit_read : string, default: us
            The unit of the read-in data.  The default value of GRAVITY data is
            microsecond.

        Returns
        -------
        dit : Astropy Quantity
            The integration time of one DIT.
        """
        if self.insname == "GRAVITY_SC":
            det_num = "DET2"
        elif self.insname == "GRAVITY_FT":
            det_num = "DET3"
        else:
            raise ValueError("Cannot recognize the insname ({0})!".format(self.insname))
        keyword = "HIERARCH ESO {0} SEQ1 DIT".format(det_num)
        dit = self.header.get(keyword) * u.Unit(unit_read)
        return dit

    def get_time_start(self, unit_read="us"):
        """
        Get the start time of the data (first time of VISTIME).

        Parameters
        ----------
        unit_read : string, default: us
            The unit of the read-in data.  The default value of GRAVITY data is
            microsecond.

        Returns
        -------
        time_start : Astropy Quantity
            The start time of the data.
        """
        vis_time = self.get_data("vis_time")
        time_start = vis_time[0, 0] * u.Unit(unit_read)
        return time_start

    def get_time_end(self, unit_read="us"):
        """
        Get the end time of the data (last time of VISTIME).

        Parameters
        ----------
        unit_read : string, default: us
            The unit of the read-in data.  The default value of GRAVITY data is
            microsecond.

        Returns
        -------
        time_start : Astropy Quantity
            The start time of the data.
        """
        vis_time = self.get_data("vis_time")
        time_end = vis_time[-1, 0] * u.Unit(unit_read)
        return time_end

    def ruv_mas(self, flag=True, flag_kwargs={}):
        """
        Calculate the uv distance with units: milli-arcsec^1.
        """
        u, v = self.uv_mas(flag=flag, flag_kwargs=flag_kwargs)
        ruv = np.sqrt(u**2 + v**2)
        return ruv

    def uv_mas(self, flag=True, flag_kwargs={}):
        """
        Calculate the uv coordinates with units: milli-arcsec^1.
        """
        if flag:
            u = self.get_data_flagged("vis_ucoord", **flag_kwargs)
            v = self.get_data_flagged("vis_vcoord", **flag_kwargs)
        else:
            u = self.get_data_fulldim("vis_ucoord")
            v = self.get_data_fulldim("vis_vcoord")
        wavelength = self.get_data("wavelength")
        u_mas = u / wavelength / (180. / np.pi * 3.6e6)
        v_mas = v / wavelength / (180. / np.pi * 3.6e6)
        return (u_mas, v_mas)

    def get_baseline_UT(self, baseline_index):
        """
        Get the baseline of the telescope pairs in terms of telescope name.

        Parameters
        ----------
        baseline_index : int
            The index of the baseline, 0-5.
        """
        tel_name = self.data.data_dict["tel_name"]
        sta_index = list(self.data.data_dict["sta_index"])
        bsl_code = list(self.data.data_dict["vis_baseline"][0, baseline_index, :])
        bsl_list = []
        for bsl in bsl_code:
            bsl_list.append(tel_name[sta_index.index(bsl)])
        return bsl_list

    def get_baseline_Tidx(self, baseline_index):
        """
        Get the baseline of the telescope pairs in terms of index 0~3.

        Parameters
        ----------
        baseline_index : int
            The index of the baseline, 0-5.
        """
        tel_name = self.data.data_dict["tel_name"]
        sta_index = list(self.data.data_dict["sta_index"])
        bsl_code = list(self.data.data_dict["vis_baseline"][0, baseline_index, :])
        bsl_list = []
        for bsl in bsl_code:
            bsl_list.append(sta_index.index(bsl))
        return bsl_list

    def get_data_keys(self):
        """
        Get the keys of the data_dict.
        """
        return self.data.data_dict.keys()

    def get_data(self, keyword):
        """
        Get the data identified by the keyword.

        Parameters
        ----------
        keyword : string
            The keyword of the data.

        Returns
        -------
        data : array
            The data array or None if the keyword is not found.
        """
        data = self.data.get_data(keyword)
        return data

    def get_data_fulldim(self, keyword):
        """
        Get the data expanded in full dimention.

        keyword : string
            The keyword of the data.

        Returns
        -------
        data_array: array
            The data array expanded to its full dimension.
        """
        data_fulldim = self.data.get_data_fulldim(keyword)
        return data_fulldim

    def get_data_flagged(self, keyword, **kwargs):
        """
        Get the masked data according to the flag.

        Parameters
        ----------
        keyword : string
            The keyword of the data.
        **kwargs : Additional parameters for GravityP2VMRED objects.

        Returns
        -------
        data_flagged: masked array
            The masked data array.
        """
        data_flagged = self.data.get_data_flagged(keyword, **kwargs)
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


class GravityVis(object):
    """
    The object of gravity visibility data.
    """
    def __init__(self, data_dict=None, filename=None, insname=None, dataList=None,
                 verbose=True):
        """
        Parameters
        ----------
        data_dict : dict (optional)
            The dictionary of data obtained from readfits_ins().  If data_dict is
            provided, the parameters to read from a fits file will be ignored.
        filename : string (optional)
            The fits file name.
        insname : string (optional)
            The keyword INSNAME that is used to select the data, case free.
        dataList : list (optional)
            The list of data keywords to read.
        verbose : bool, default: True
            Print notes if True.
        """
        #-> Prior properties
        self.__catglist = ["SINGLE_SCI_VIS", "SINGLE_SCI_VIS_CALIBRATED", "SINGLE_CAL_VIS",
                           "DUAL_SCI_VIS"]
        self.ndim_ft = {
            "vis" : (1, 6, 5),
            "vis2": (1, 6, 5),
            "flux": (1, 4, 5),
            "t3"  : (1, 4, 5),
        }
        self.ndim_sc = {
            "vis" : (1, 6, 210),
            "vis2": (1, 6, 210),
            "flux": (1, 4, 210),
            "t3"  : (1, 4, 210),
        }
        self.datakey_list = ["vis", "vis2", "t3", "flux"]
        self.auxkey_list = ["station", "baseline", "triangle"]
        #-> Read in the fits file data_dict is None
        if data_dict is None:
            assert not filename is None
            assert not insname is None
            data_dict = readfits_ins(filename, insname, dataList)
        else:
            if verbose & (not ((filename is None) & (insname is None) & (dataList is None))):
                print("The data_dict is used so the other parameters are ignored!")
        #-> Information from the header
        header = data_dict["HEADER"]
        self.header = header
        #--> Basical information
        self.catg = header.get("HIERARCH ESO PRO CATG", None)
        if verbose & (not self.catg in self.__catglist):
            print("The catg ({0}) has not been tested before!".format(self.catg))
        self.insname = data_dict["INSNAME"]
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
        #-> Data content
        wavelength = data_dict["OI_WAVELENGTH"]["EFF_WAVE"]
        visdata = data_dict["OI_VIS"]
        vis2data = data_dict["OI_VIS2"]
        t3data = data_dict["OI_T3"]
        fluxdata = data_dict["OI_FLUX"]
        self.data_dict = {
            #--> General data
            "wavelength": wavelength,
            "bandwidth": data_dict["OI_WAVELENGTH"]["EFF_BAND"],
            "tel_name": data_dict["TEL_NAME"],
            "sta_name": data_dict["STA_NAME"],
            "sta_index": data_dict["STA_INDEX"],
            #--> Vis data
            "vis_flag"    : visdata["FLAG"],
            "vis_baseline": visdata["STA_INDEX"],
            "vis_ucoord"  : visdata["UCOORD"],
            "vis_vcoord"  : visdata["VCOORD"],
            "vis_time"    : visdata["TIME"],
            "vis_int_time": visdata["INT_TIME"],
            "vis_amp"     : visdata["VISAMP"],
            "vis_amp_err" : visdata["VISAMPERR"],
            "vis_phi"     : visdata["VISPHI"],
            "vis_phi_err" : visdata["VISPHIERR"],
            "vis_data"    : visdata["VISDATA"],
            "vis_err"     : visdata["VISERR"],
            "vis_r"       : visdata["RVIS"],
            "vis_r_err"   : visdata["RVISERR"],
            "vis_i"       : visdata["IVIS"],
            "vis_i_err"   : visdata["IVISERR"],
            #--> Vis2 data
            "vis2_flag"    : vis2data["FLAG"],
            "vis2_baseline": vis2data["STA_INDEX"],
            "vis2_ucoord"  : vis2data["UCOORD"],
            "vis2_vcoord"  : vis2data["VCOORD"],
            "vis2_data"    : vis2data["VIS2DATA"],
            "vis2_err"     : vis2data["VIS2ERR"],
            #--> T3 data
            "t3_flag"    : t3data["FLAG"],
            "t3_triangle": t3data["STA_INDEX"],
            "t3_u1coord" : t3data["U1COORD"],
            "t3_v1coord" : t3data["V1COORD"],
            "t3_u2coord" : t3data["U2COORD"],
            "t3_v2coord" : t3data["V2COORD"],
            "t3_amp"     : t3data["T3AMP"],
            "t3_amp_err" : t3data["T3AMPERR"],
            "t3_phi"     : t3data["T3PHI"],
            "t3_phi_err" : t3data["T3PHIERR"],
            #--> Flux data
            "flux_flag"   : fluxdata["FLAG"],
            "flux_station": fluxdata["STA_INDEX"],
            "flux_data"   : fluxdata["FLUX"],
            "flux_err"    : fluxdata["FLUXERR"],
        }

    def get_data(self, keyword):
        """
        Get the data identified by the keyword.

        Parameters
        ----------
        keyword : string
            The keyword of the data.

        Returns
        -------
        data : array
            The data array or None if the keyword is not found.
        """
        data = self.data_dict.get(keyword, None)
        if not data is None:
            data = data.copy()
        return data

    def get_data_fulldim(self, keyword):
        """
        Get the data expanded in full dimention.

        keyword : string
            The keyword of the data.

        Returns
        -------
        data_array: array
            The data array expanded to its full dimension.
        """
        kw_prf, kw_par = keyword.split("_")[:2]
        if not kw_prf in self.datakey_list:
            raise ValueError("The keyword ({0}) cannot be expanded!".format(keyword))
        if kw_par in self.auxkey_list:
            raise ValueError("The keyword ({0}) cannot be expanded!".format(keyword))
        if self.insname == "GRAVITY_FT":
            ndim = self.ndim_ft[kw_prf]
        elif self.insname == "GRAVITY_SC":
            ndim = self.ndim_sc[kw_prf]
        else:
            raise ValueError("Cannot recognize self.insname ({0})!".format(self.insname))
        data_array = self.get_data(keyword)
        if data_array is None:
            raise ValueError("The keyword ({0}) is not recognized!".format(keyword))
        dshape = data_array.shape
        if dshape == ndim:
            return data_array
        else:
            if (len(dshape) == 2) & (dshape[0] == ndim[0]) & (dshape[1] == ndim[1]):
                data_array_ext = np.zeros(ndim, dtype=np.float)
                for loop in range(ndim[2]):
                    data_array_ext[:, :, loop] = data_array
                return data_array_ext
            else:
                raise ValueError("The shape of {0} ({1}) is not correct ({2})!".format(keyword, dshape, ndim))

    def get_data_flagged(self, keyword):
        """
        Get the masked data according to the flag.

        Parameters
        ----------
        keyword : string
            The keyword of the data.

        Returns
        -------
        data_flagged: masked array
            The masked data array.
        """
        kw_prf, kw_par = keyword.split("_")[:2]
        if not kw_prf in self.datakey_list:
            raise ValueError("The keyword ({0}) cannot be flagged!".format(keyword))
        if kw_par in self.auxkey_list:
            raise ValueError("The keyword ({0}) cannot be flagged!".format(keyword))
        data_array = self.get_data_fulldim(keyword)
        kw_flag = "{0}_flag".format(kw_prf)
        flag = self.get_data_fulldim(kw_flag)
        data_flagged = np.ma.array(data_array, mask=flag)
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


class GravityP2VMRED(object):
    """
    The object of gravity p2vmred data.
    """
    def __init__(self, data_dict=None, filename=None, insname=None, dataList=None,
                 verbose=True):
        """
        Parameters
        ----------
        data_dict : dict (optional)
            The dictionary of data obtained from readfits_ins().  If data_dict is
            provided, the parameters to read from a fits file will be ignored.
        filename : string (optional)
            The fits file name.
        insname : string (optional)
            The keyword INSNAME that is used to select the data, case free.
        dataList : list (optional)
            The list of data keywords to read.
        verbose : bool, default: True
            Print notes if True.
        """
        #-> Prior properties
        self.__catglist = ["SINGLE_SCI_P2VMRED", "DUAL_SCI_P2VMRED"]
        self.ndim_ft = {
            "vis" : (-1, 6, 5),
            "vis2": (-1, 6, 5),
            "flux": (-1, 4, 5),
            "t3"  : (-1, 4, 5),
        }
        self.ndim_sc = {
            "vis" : (-1, 6, 210),
            "vis2": (-1, 6, 210),
            "flux": (-1, 4, 210),
            "t3"  : (-1, 4, 210),
        }
        self.datakey_list = ["vis", "vis2", "t3", "flux"]
        self.auxkey_list = ["station", "baseline", "triangle"]
        #-> Read in the fits file data_dict is None
        if data_dict is None:
            assert not filename is None
            assert not insname is None
            data_dict = readfits_ins(filename, insname, dataList)
        else:
            if verbose & (not ((filename is None) & (insname is None) & (dataList is None))):
                print("The data_dict is used so the other parameters are ignored!")
        #-> Information from the header
        header = data_dict["HEADER"]
        self.header = header
        #--> Basical information
        self.catg = header.get("HIERARCH ESO PRO CATG", None)
        if verbose & (not self.catg in self.__catglist):
            print("The catg ({0}) has not been tested before!".format(self.catg))
        self.insname = data_dict["INSNAME"]
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
        #-> Data content
        wavelength = data_dict["OI_WAVELENGTH"]["EFF_WAVE"]
        visdata = data_dict["OI_VIS"]
        fluxdata = data_dict["OI_FLUX"]
        self.data_dict = {
            #--> General data
            "wavelength": wavelength,
            "bandwidth": data_dict["OI_WAVELENGTH"]["EFF_BAND"],
            "tel_name": data_dict["TEL_NAME"],
            "sta_name": data_dict["STA_NAME"],
            "sta_index": data_dict["STA_INDEX"],
            #--> Vis data
            "vis_flag": visdata["FLAG"],
            "vis_baseline": visdata["STA_INDEX"],
            "vis_time": visdata["TIME"],
            "vis_int_time": visdata["INT_TIME"],
            "vis_ucoord": visdata["UCOORD"],
            "vis_vcoord": visdata["VCOORD"],
            "vis_data": visdata["VISDATA"],
            "vis_err": visdata["VISERR"],
            "vis_self_ref": visdata["SELF_REF"],
            "vis_phase_ref": visdata.get("PHASE_REF", None),
            "vis_f1f2": visdata["F1F2"],
            "vis_rejection_flag": visdata["REJECTION_FLAG"],
            "vis_gdelay": visdata["GDELAY"],
            "vis_gdelay_boot": visdata["GDELAY_BOOT"],
            "vis_snr": visdata["SNR"],
            "vis_snr_boot": visdata["SNR_BOOT"],
            #--> Flux data
            "flux_flag": fluxdata["FLAG"],
            "flux_station": fluxdata["STA_INDEX"],
            "flux_data": fluxdata["FLUX"],
            "flux_err": fluxdata["FLUXERR"],
        }
        #--> Align the coherent flux according to the reference phase
        if self.insname == "GRAVITY_FT":
            phref = self.data_dict["vis_self_ref"]
        elif self.insname == "GRAVITY_SC":
            phref = self.data_dict["vis_phase_ref"]
        else:
            phref = None
        if not phref is None:
            rvis = np.real(visdata["VISDATA"])
            ivis = np.imag(visdata["VISDATA"])
            rvis_align = np.cos(phref) * rvis - np.sin(phref) * ivis
            ivis_align = np.sin(phref) * rvis + np.cos(phref) * ivis
            self.data_dict["vis_data_aligned"] = rvis_align + 1j*ivis_align

    def get_data(self, keyword):
        """
        Get the data identified by the keyword.

        Parameters
        ----------
        keyword : string
            The keyword of the data.

        Returns
        -------
        data : array
            The data array or None if the keyword is not found.
        """
        data = self.data_dict.get(keyword, None)
        if not data is None:
            data = data.copy()
        return data

    def get_data_fulldim(self, keyword):
        """
        Get the data expanded in full dimention.

        keyword : string
            The keyword of the data.

        Returns
        -------
        data_array: array
            The data array expanded to its full dimension.
        """
        kw_prf, kw_par = keyword.split("_")[:2]
        if not kw_prf in self.datakey_list:
            raise ValueError("The keyword ({0}) cannot be expanded!".format(keyword))
        if kw_par in self.auxkey_list:
            raise ValueError("The keyword ({0}) cannot be expanded!".format(keyword))
        if self.insname == "GRAVITY_FT":
            ndim = self.ndim_ft[kw_prf]
        elif self.insname == "GRAVITY_SC":
            ndim = self.ndim_sc[kw_prf]
        else:
            raise ValueError("Cannot recognize self.insname ({0})!".format(self.insname))
        data_array = self.get_data(keyword)
        if data_array is None:
            raise ValueError("The keyword ({0}) is not recognized!".format(keyword))
        dshape = data_array.shape
        if dshape[1:] == ndim[1:]: # The 0th dimension of P2VMRED data is not fixed.
            return data_array
        else:
            if (len(dshape) == 2) & (dshape[1] == ndim[1]):
                data_array_ext = np.zeros((dshape[0], dshape[1], ndim[2]), dtype=np.float)
                for loop in range(ndim[2]):
                    data_array_ext[:, :, loop] = data_array
                return data_array_ext
            else:
                raise ValueError("The shape of {0} ({1}) is not correct ({2})!".format(keyword, dshape, ndim))

    def get_data_flagged(self, keyword, mask=None):
        """
        Get the masked data according to the rejection flag.

        Parameters
        ----------
        keyword : string
            The keyword of the data.
        mask : bool array (optional)
            The specified mask.  If None, the "REJECTION_FLAG" will be used to
            flag the data.

        Returns
        -------
        data_flagged: masked array
            The masked data array.
        """
        kw_prf, kw_par = keyword.split("_")[:2]
        if not kw_prf in self.datakey_list:
            raise ValueError("The keyword ({0}) cannot be flagged!".format(keyword))
        if kw_par in self.auxkey_list:
            raise ValueError("The keyword ({0}) cannot be flagged!".format(keyword))
        data_array = self.get_data_fulldim(keyword)
        if mask is None:
            mask = self.get_data_fulldim("vis_rejection_flag") > 0
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



reshapeDict = {
    "OI_VIS": 6, # Number of baselines
    "OI_VIS2": 6, # Number of baselines
    "OI_FLUX": 4, # Number of telescopes
    "OI_T3": 4, # Number of triangles
}
def readfits_ins(filename, insname, dataList=None):
    """
    Read the fits file with the select insname.

    Parameters
    ----------
    filename : string
        The fits file name.
    insname : string
        The keyword INSNAME that is used to select the data, case free.
    dataList : list (optional)
        The list of data keywords to read.

    Returns
    -------
    outDict : dict
        The dict of data read from the file.
    """
    hdulist = fits.open(filename, mode='readonly')
    header = hdulist[0].header
    insname = insname.upper()
    #-> Make sure the data is in single mode
    assert header["HIERARCH ESO INS POLA MODE"] == "COMBINED"
    outDict = {
        "HEADER": header,
        "INSNAME": insname,
        "TEL_NAME": hdulist["OI_ARRAY"].data["TEL_NAME"],
        "STA_NAME": hdulist["OI_ARRAY"].data["STA_NAME"],
        "STA_INDEX": hdulist["OI_ARRAY"].data["STA_INDEX"],
    }
    #-> Choose the proper extensions
    dataDict = {}
    for loop in range(len(hdulist)):
        hdu = hdulist[loop]
        hduname = hdu.name
        #--> Use INSNAME to determine the data to include
        if insname in str(hdu.header.get('INSNAME',[])):
            if hduname in dataDict.keys():
                raise ValueError("{0} is repeating!".format(hduname))
            dataDict[hduname] = {
                "extension": loop,
                "reshape": reshapeDict.get(hduname, 0),
            }
    if dataList is None:
        dataList = dataDict.keys()
    if len(dataList) == 0:
        raise RuntimeError("There is not data to be extracted!")
    #-> Go through the necessary extensions
    for kw in dataList:
        #--> Report error if the kw is not found in the data.
        ext = dataDict[kw]["extension"]
        nreshape = dataDict[kw]["reshape"]
        outDict[kw] = {
            "HEADER": hdulist[ext].header
        }
        hdudata = hdulist[ext].data
        #--> Go through all the columns
        for dkw in hdudata.columns.names:
            darray = hdudata[dkw]
            if nreshape > 0:
                ndim = len(darray.shape)
                if ndim == 1:
                    darray = darray.reshape(-1, nreshape)
                elif ndim == 2:
                    darray = darray.reshape(-1, nreshape, darray.shape[1])
                else:
                    ValueError("The shape of {0}-{1} ({2}) is not managable!".format(kw,
                               dkw, darray.shape))
            outDict[kw][dkw] = darray
    return outDict
