import datetime
import numpy as np
from astropy.io import fits

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
        return gdList

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

    def get_DataSeries_flagged(self, datakey="obsdate", verbose=True, **kwargs):
        """
        Get the time series of data identified by data_key.
        """
        dataList = []
        for gd in self.gd_list:
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

    def get_DataSeries(self, datakey="obsdate", verbose=True):
        """
        Get the time series of data identified by data_key.
        """
        dataList = []
        for gd in self.gd_list:
            if datakey == "obsdate":
                dataList.append(gd.obsdate)
            elif datakey in gd.get_data_keys():
                dataList.append(gd.get_data(datakey))
            elif datakey in gd.get_qc_keys():
                dataList.append(gd.get_qc(datakey))
            else:
                raise ValueError("Cannot find the data ({0})!".format(datakey))
        return dataList





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
        self.__catglist = ["SINGLE_SCI_VIS", "SINGLE_SCI_P2VMRED"]
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
        if not self.catg in self.__catglist:
            raise ValueError("The catg ({0}) has not been tested before!".format(self.catg))
        self.insname = data_dict["INSNAME"]
        self.obsdate = datetime.datetime.strptime(header["DATE-OBS"], '%Y-%m-%dT%H:%M:%S')
        self.object = header["OBJECT"]
        self.ra=header['RA']
        self.dec=header['DEC']
        #-> Get the data
        if self.catg in ["SINGLE_SCI_VIS"]:
            self.data = GravityVis(data_dict, verbose=verbose)
        elif self.catg in ["SINGLE_SCI_P2VMRED"]:
            self.data = GravityP2VMRED(data_dict, verbose=verbose)
        else:
            raise ValueError("The catg ({0}) is not supported!".format(self.catg))

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
        The data array or None if the keyword is not found.
        """
        return self.data.get_data(keyword)

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
        return self.data.get_data_flagged(keyword, **kwargs)

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
        self.__catglist = ["SINGLE_SCI_VIS"]
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
        visdata = data_dict["OI_VIS"]
        vis2data = data_dict["OI_VIS2"]
        t3data = data_dict["OI_T3"]
        fluxdata = data_dict["OI_FLUX"]
        self.data_dict = {
            #--> General data
            "wavelength": data_dict["OI_WAVELENGTH"]["EFF_WAVE"],
            "tel_name": data_dict["TEL_NAME"],
            "sta_name": data_dict["STA_NAME"],
            "sta_index": data_dict["STA_INDEX"],
            #--> Vis data
            "vis_flag": visdata["FLAG"],
            "vis_baseline": visdata["STA_INDEX"],
            "vis_u": visdata["UCOORD"],
            "vis_v": visdata["VCOORD"],
            "vis_amp": visdata["VISAMP"],
            "vis_amp_err": visdata["VISAMPERR"],
            "vis_phi": visdata["VISPHI"],
            "vis_phi_err": visdata["VISPHIERR"],
            "vis_data": visdata["VISDATA"],
            "vis_err": visdata["VISERR"],
            #--> Vis2 data
            "vis2_flag": vis2data["FLAG"],
            "vis2_baseline": vis2data["STA_INDEX"],
            "vis2_u": vis2data["UCOORD"],
            "vis2_v": vis2data["VCOORD"],
            "vis2_data": vis2data["VIS2DATA"],
            "vis2_err": vis2data["VIS2ERR"],
            #--> T3 data
            "t3_flag": t3data["FLAG"],
            "t3_triangle": t3data["STA_INDEX"],
            "t3_u1": t3data["U1COORD"],
            "t3_v1": t3data["V1COORD"],
            "t3_u2": t3data["U2COORD"],
            "t3_v2": t3data["V2COORD"],
            "t3_amp": t3data["T3AMP"],
            "t3_amp_err": t3data["T3AMPERR"],
            "t3_phi": t3data["T3PHI"],
            "t3_phi_err": t3data["T3PHIERR"],
            #--> Flux data
            "flux_flag": fluxdata["FLAG"],
            "flux_station": fluxdata["STA_INDEX"],
            "flux_data": fluxdata["FLUX"],
            "flux_err": fluxdata["FLUXERR"],
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
        The data array or None if the keyword is not found.
        """
        return self.data_dict.get(keyword, None)

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
        data_array = self.data_dict[keyword]
        kw_flag = "{0}_flag".format(keyword.split("_")[0])
        flag = self.data_dict.get(kw_flag, None)
        if flag is None:
            raise ValueError("The flag keyword ({0}) is not recognized!".format(kw_flag))
        assert data_array.shape == flag.shape
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
        self.__catglist = ["SINGLE_SCI_P2VMRED"]
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
        visdata = data_dict["OI_VIS"]
        fluxdata = data_dict["OI_FLUX"]
        self.data_dict = {
            #--> General data
            "wavelength": data_dict["OI_WAVELENGTH"]["EFF_WAVE"],
            "tel_name": data_dict["TEL_NAME"],
            "sta_name": data_dict["STA_NAME"],
            "sta_index": data_dict["STA_INDEX"],
            #--> Vis data
            "vis_flag": visdata["FLAG"],
            "vis_baseline": visdata["STA_INDEX"],
            "vis_time": visdata["TIME"],
            "vis_u": visdata["UCOORD"],
            "vis_v": visdata["VCOORD"],
            "vis_data": visdata["VISDATA"],
            "vis_err": visdata["VISERR"],
            "vis_self_ref": visdata["SELF_REF"],
            "vis_phase_ref": visdata.get("PHASE_REF", None),
            "vis_f1f2": visdata["F1F2"],
            "vis_rejection_flag": visdata["REJECTION_FLAG"],
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
        The data array or None if the keyword is not found.
        """
        return self.data_dict.get(keyword, None)

    def get_data_flagged(self, keyword, kw_flag="vis_rejection_flag", threshold=1.):
        """
        Get the masked data according to the rejection flag.

        Parameters
        ----------
        keyword : string
            The keyword of the data.
        kw_flag : string, default: vis_rejection_flag
            The keyword of the flag.
        threshold : float, default: 1.
            The threshold above which the data are flagged.

        Returns
        -------
        data_flagged: masked array
            The masked data array.
        """
        data_array = self.get_data(keyword)
        if data_array is None:
            raise ValueError("The keyword ({0}) is not recognized!".format(keyword))
        flag = self.get_data(kw_flag)
        if flag is None:
            raise ValueError("Cannot find the flag data ({0})!".format(kw_flag))
        dshape = data_array.shape
        if len(dshape) == 2:
            mask = flag > threshold
        elif len(dshape) == 3:
            mask = np.zeros_like(data_array, dtype=bool)
            for loop in range(dshape[2]):
                mask[:, :, loop] = flag > threshold
        else:
            raise ValueError("The data shape ({0}) is wrong!".format(dshape))
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
    "OI_FLUX": 4, # Number of telescopes
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
