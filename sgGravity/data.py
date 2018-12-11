import datetime
import numpy as np
from astropy.io import fits

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
        self.obsdate = datetime.datetime.strptime(header["DATE-OBS"], '%Y-%m-%dT%H:%M:%S')
        self.object = header["OBJECT"]
        self.ra=header['RA']
        self.dec=header['DEC']
        self.insmode = header["INSMODE"].split(",")
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
        """
        data_array = self.data_dict[keyword]
        kw_flag = "{0}_flag".format(keyword.split("_")[0])
        flag = self.data_dict.get(kw_flag, None)
        if flag is None:
            raise ValueError("The flag keyword ({0}) is not recognized!".format(kw_flag))
        assert data_array.shape == flag.shape
        data_flagged = np.ma.array(data_array, mask=flag)
        return data_flagged


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
        self.obsdate = datetime.datetime.strptime(header["DATE-OBS"], '%Y-%m-%dT%H:%M:%S')
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

    def get_data_flagged(self, keyword, kw_flag="vis_rejection_flag", threshold=1):
        """
        Get the masked data according to the rejection flag.

        Parameters
        ----------
        keyword : string
            The keyword of the data.  It could be vis and flux data.
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
