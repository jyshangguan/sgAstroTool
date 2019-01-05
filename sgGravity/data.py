import datetime
import numpy as np
from astropy.io import fits
from astropy import units as u

__all__ = ["GravitySet", "GravityData", "GravityVis", "GravityP2VMRED"]

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
                               "DUAL_SCI_VIS"]
        self.__catglist_p2vmred = ["SINGLE_SCI_P2VMRED", "DUAL_SCI_P2VMRED", "SINGLE_CAL_P2VMRED"]
        self.dims = {
            "BASELINE": 6,
            "TELESCOPE": 4,
            "TRIANGLE": 4,
            "CHANNEL_FT": 5,
            "CHANNEL_SC": 210,
            "OI_VIS:STA_INDEX": 2,
            "OI_VIS2:STA_INDEX": 2,
            "OI_FLUX:STA_INDEX": 1,
        }
        self.dim2 = {
            "OI_VIS" : self.dims["BASELINE"],
            "OI_VIS2": self.dims["BASELINE"],
            "OI_FLUX": self.dims["TELESCOPE"],
            "OI_T3": self.dims["TRIANGLE"],
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
            The instrument name, "ft" for fringe tracking or "sc" for science.
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
            The instrument name, "ft" for fringe tracking or "sc" for science.
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

    def ruv_mas(self, insname="ft", verbose=False):
        """
        Calculate the uv distance with units: milli-arcsec^1.

        Parameters
        ----------
        insname : string
            The instrument name, "ft" for fringe tracking or "sc" for science.
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
            The instrument name, "ft" for fringe tracking or "sc" for science.
        verbose : bool
            Print more information if True.

        Returns
        -------
        (u_mas, v_mas) : tuple of arrays
            The UV coordinates of each baseline and channel, units: milli-arcsec^-1.
        """
        if not insname.upper() in ["SC", "FT"]:
            raise ValueError("The insname ({0}) is not recognized!".format(insname))
        u = self.get_data_fulldim("OI_VIS:UCOORD", insname=insname, verbose=verbose)
        v = self.get_data_fulldim("OI_VIS:VCOORD", insname=insname, verbose=verbose)
        wavelength = self.get_data("OI_WAVELENGTH:EFF_WAVE", insname=insname, verbose=verbose)
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
        tel_name = self.get_data("OI_ARRAY:TEL_NAME", insname="aux")
        sta_index = list(self.get_data("OI_ARRAY:STA_INDEX", insname="aux"))
        vis_station = self.get_data("OI_VIS:STA_INDEX", insname="sc") # SC and FT provides the same information.
        if self.catg in self.__catglist_p2vmred:
            bsl_code = vis_station[0, baseline_index, :]
        elif self.catg in self.__catglist_vis:
            bsl_code = vis_station[baseline_index, :]
        else:
            raise ValueError("Self.catg ({0}) is not recognized!".format(self.catg))
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
        sta_index = list(self.get_data("OI_ARRAY:STA_INDEX", insname="aux"))
        vis_station = self.get_data("OI_VIS:STA_INDEX", insname="sc") # SC and FT provides the same information.
        if self.catg in self.__catglist_p2vmred:
            bsl_code = vis_station[0, baseline_index, :]
        elif self.catg in self.__catglist_vis:
            bsl_code = vis_station[baseline_index, :]
        else:
            raise ValueError("Self.catg ({0}) is not recognized!".format(self.catg))
        bsl_list = []
        for bsl in bsl_code:
            bsl_list.append(sta_index.index(bsl))
        return bsl_list

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
            The instrument name, "ft" for fringe tracking or "sc" for science.
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
            The instrument name, "ft" for fringe tracking or "sc" for science.
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
                           "DUAL_SCI_VIS"]
        self.dims = {
            "BASELINE": 6,
            "TELESCOPE": 4,
            "TRIANGLE": 4,
            "CHANNEL_FT": 5,
            "CHANNEL_SC": 210,
            "OI_VIS:STA_INDEX": 2,
            "OI_VIS2:STA_INDEX": 2,
            "OI_FLUX:STA_INDEX": 1,
        }
        self.dim2 = {
            "OI_VIS" : self.dims["BASELINE"],
            "OI_VIS2": self.dims["BASELINE"],
            "OI_FLUX": self.dims["TELESCOPE"],
            "OI_T3": self.dims["TRIANGLE"],
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

    def get_data(self, keyword, insname="ft", verbose=False):
        """
        Get a copy of the data identified by the keyword.

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
        hdulist = self.hdulist
        insname = insname.upper()
        if not ((insname == "FT") | (insname == "SC") | (insname == "AUX")):
            errortext = "The insname ({0}) is incorrect!  It should be ft, sc, or aux, case free.".format(insname)
            raise ValueError(errortext)
        insname = "GRAVITY_{0}".format(insname)
        extCount = 0
        for loop in range(len(hdulist)):
            hdu = hdulist[loop]
            #--> Use INSNAME to determine the data to include
            if (hdu.name == extName) & (insname in str(hdu.header.get('INSNAME', "GRAVITY_AUX"))):
                hdudata = hdu.data
                extCount += 1
        if extCount == 0:
            if verbose:
                print("The extension ({0}) is not found!".format(extName))
            return None
        elif extCount > 1:
            raise ValueError("There are {0} same extensions {1} for {2}!".format(extCount, extName, insname))
        try:
            data = hdudata[datName].copy()
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
            The instrument name, "ft" for fringe tracking or "sc" for science.
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
        if (insname == "FT") or (insname == "SC"):
            nchn = self.dims["CHANNEL_{0}".format(insname)]
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
            The instrument name, "ft" for fringe tracking or "sc" for science.
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
        self.dims = {
            "BASELINE": 6,
            "TELESCOPE": 4,
            "TRIANGLE": 4,
            "CHANNEL_FT": 5,
            "CHANNEL_SC": 210,
            "OI_VIS:STA_INDEX": 2,
            "OI_VIS2:STA_INDEX": 2,
            "OI_FLUX:STA_INDEX": 1,
        }
        self.dim2 = {
            "OI_VIS" : self.dims["BASELINE"],
            "OI_VIS2": self.dims["BASELINE"],
            "OI_FLUX": self.dims["TELESCOPE"],
            "OI_T3": self.dims["TRIANGLE"],
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
        hdulist = self.hdulist
        insname = insname.upper()
        if not ((insname == "FT") | (insname == "SC") | (insname == "AUX")):
            errortext = "The insname ({0}) is incorrect!  It should be ft, sc, or aux, case free.".format(insname)
            raise ValueError(errortext)
        insname = "GRAVITY_{0}".format(insname)
        extCount = 0
        for loop in range(len(hdulist)):
            hdu = hdulist[loop]
            #--> Use INSNAME to determine the data to include
            if (hdu.name == extName) & (insname in str(hdu.header.get('INSNAME', "GRAVITY_AUX"))):
                hdudata = hdu.data
                extCount += 1
        if extCount == 0:
            if verbose:
                print("The extension ({0}) is not found!".format(extName))
            return None
        elif extCount > 1:
            raise ValueError("There are {0} same extensions {1} for {2}!".format(extCount, extName, insname))
        try:
            data = hdudata[datName].copy()
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
            The instrument name, "ft" for fringe tracking or "sc" for science.
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
        if (insname == "FT") or (insname == "SC"):
            nchn = self.dims["CHANNEL_{0}".format(insname)]
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
            The instrument name, "ft" for fringe tracking or "sc" for science.
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
