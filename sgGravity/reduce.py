import numpy as np
from .data import GravityData

__all__ = ["gravi_vis_average_bootstrap", "correct_visdata"]

def sinc_modulation(tau, d):
    """
    The sinc function.
    """
    x = np.pi * tau / d
    return np.sin(x) / x

def VisLoss(gdelay, **kwargs):
    """
    Calculate the visibility loss.
    """
    vis = sinc_modulation(gdelay, **kwargs)
    return vis

def correct_visdata(p2vmred):
    """
    Correct the visdata of P2VMRED data.
    """
    visdata = p2vmred.get_data("OI_VIS:VISDATA", insname="ft")
    effwave = p2vmred.get_data("oi_wavelength:eff_wave", insname="ft")
    effband = p2vmred.get_data("oi_wavelength:eff_band", insname="ft")
    gdelay  = p2vmred.get_data("oi_vis:gdelay", insname="ft")
    coheLen = effwave**2 / effband
    nfrm, nbsl, nchn = visdata.shape # Get the number of frames, baselines and channels
    visloss = np.zeros([nfrm, nbsl, nchn])
    for loop in range(nchn):
        visloss[:, :, loop] = VisLoss(gdelay, d=coheLen[loop])
        visdata[:, :, loop] = visdata[:, :, loop] / visloss[:, :, loop]
    return visdata

def gravi_vis_average_bootstrap(p2vmred, mask=None, correction=False,
                                nboot=20, nseg_default=100):
    """
    Calculate average the P2VMRED data to obtain the SCIVIS data.  The function
    works with the same procedure as the function gravi_vis_average_bootstrap()
    in gvraid2:/home/grav/SVN/gravity_dev/gravityp/gravi/gravi_vis.c

    Parameters
    ----------
    p2vmred : GravityData
        The p2vmred data used for average, with the following contents:
        rejection_flag: Baseline wise rejection flag, (frame, baseline);
        visdata: Unnormalized complex visibility data, (frame, baseline, channel);
        viserr: The error of visdata, (frame, baseline, channel);
        self_ref: The self reference phase, (frame, baseline, channel);
        f1f2: The geometric flux, (frame, baseline, channel).
    nboot : int, default: 20
        The number of times to bootstrap the data.
    nseg_default : int, default: 100
        The number of segments to cut the data.
    """
    #-> Prepare the data
    #-#> Select the correct frames
    dit_sc = p2vmred.get_time_dit(insname="sc")
    time_start = p2vmred.get_time_start(insname="sc") - dit_sc * 0.5
    time_end   = p2vmred.get_time_end(insname="sc") + dit_sc * 0.5
    timeList = p2vmred.get_data("OI_VIS:TIME", insname="ft")[:, 0]
    fltr_time = (timeList > time_start.to("us").value) & (timeList < time_end.to("us").value)
    rejflag = p2vmred.get_data("OI_VIS:REJECTION_FLAG", insname="ft")
    if not mask is None:
        rejflag[mask] = 999
    rejflag = rejflag[fltr_time, :]
    visdata = p2vmred.get_data("OI_VIS:VISDATA", insname="ft")[fltr_time, :, :]
    viserr  = p2vmred.get_data("OI_VIS:VISERR", insname="ft")[fltr_time, :, :]
    selref  = p2vmred.get_data("OI_VIS:SELF_REF", insname="ft")[fltr_time, :, :]
    f1f2    = p2vmred.get_data("OI_VIS:F1F2", insname="ft")[fltr_time, :, :]
    ucoord  = p2vmred.get_data("OI_VIS:UCOORD", insname="ft")[fltr_time, :]
    vcoord  = p2vmred.get_data("OI_VIS:VCOORD", insname="ft")[fltr_time, :]
    inttime = p2vmred.get_data("OI_VIS:INT_TIME", insname="ft")[fltr_time, :]
    wavelength = p2vmred.get_data("OI_WAVELENGTH:EFF_WAVE", insname="ft")
    #--> Correct the visibility loss
    if correction:
        nfrm, nbsl, nchn = visdata.shape # Get the number of frames, baselines and channels
        effwave = p2vmred.get_data("oi_wavelength:eff_wave", insname="ft")
        effband = p2vmred.get_data("oi_wavelength:eff_band", insname="ft")
        gdelay  = p2vmred.get_data("oi_vis:gdelay", insname="ft")[fltr_time, :]
        coheLen = effwave**2 / effband
        visloss = np.zeros([nfrm, nbsl, nchn])
        for loop in range(nchn):
            visloss[:, :, loop] = VisLoss(gdelay, d=coheLen[loop])
            visdata[:, :, loop] = visdata[:, :, loop] / visloss[:, :, loop]
    else:
        visloss = None
    #-> Segment the data into nseg blocks
    visR_seg = []
    visI_seg = []
    visP_seg = []
    f12_seg  = []
    f1f2_seg = []
    ucoord_seg = []
    vcoord_seg = []
    inttime_seg = []
    nfrm, nbsl, nchn = visdata.shape # Get the number of frames, baselines and channels
    nsegList = []
    for loop_b in range(nbsl):
        fltr = rejflag[:, loop_b] == 0
        nvalid = np.sum(fltr)
        #--> Wisely choose the number of rows in each segment, 1 row per segment
        #if the number of frames is less than nseg_default.
        nrow_per_seg = np.max([nvalid / np.min([nfrm, nseg_default]), 1])
        nseg = int(nvalid / nrow_per_seg) # The number of segment finally used.
        nsegList.append(nseg)
        ncut = nrow_per_seg * nseg
        #--> Reshape the data into segments, (nrow_per_seg, nseg, nchn)
        visdata_rs = visdata[fltr, loop_b, :][:ncut, :].reshape(nrow_per_seg, nseg, nchn)
        viserr_rs = viserr[fltr, loop_b, :][:ncut, :].reshape(nrow_per_seg, nseg, nchn)
        selref_rs = selref[fltr, loop_b, :][:ncut, :].reshape(nrow_per_seg, nseg, nchn)
        f1f2_rs = f1f2[fltr, loop_b, :][:ncut, :].reshape(nrow_per_seg, nseg, nchn)
        f12sq_rs = f1f2_rs.copy() # Exactly following the pipeline
        f12sq_rs[f12sq_rs < 1e-15] = 0.0
        ucoord_seg.append(ucoord[fltr, loop_b][:ncut])
        vcoord_seg.append(vcoord[fltr, loop_b][:ncut])
        inttime_seg.append(inttime[fltr, loop_b][:ncut])
        #--> Calculate the intermediate quantities in the segments.
        mR = np.real(visdata_rs)
        mI = np.imag(visdata_rs)
        eR = np.real(viserr_rs)
        eI = np.imag(viserr_rs)
        tR = np.sum(np.cos(selref_rs) * mR - np.sin(selref_rs) * mI, axis=0)
        tI = np.sum(np.sin(selref_rs) * mR + np.cos(selref_rs) * mI, axis=0)
        tP = np.sum(mR**2 + mI**2 - eR**2 - eI**2, axis=0)
        tF1F2 = np.sum(f1f2_rs, axis=0)
        tF12  = np.sum(np.sqrt(f12sq_rs), axis=0)
        visR_seg.append(tR)
        visI_seg.append(tI)
        visP_seg.append(tP)
        f1f2_seg.append(tF1F2)
        f12_seg.append(tF12)
    #-> Bootstrapping
    visamp     = []
    visphi     = []
    vis2       = []
    visR       = []
    visI       = []
    visamp_err = []
    visphi_err = []
    vis2_err   = []
    visR_err   = []
    visI_err   = []
    u_mas      = []
    v_mas      = []
    inttime_total = []
    for loop_b in range(nbsl):
        visR_boot = []
        visI_boot = []
        visP_boot = []
        f1f2_boot = []
        f12_boot  = []
        nseg = nsegList[loop_b]
        for boot in range(nboot):
            if boot == 0:
                segList = np.arange(nseg)
            else:
                segList = np.random.randint(nseg, size=nseg)
            visR_boot.append(np.sum(visR_seg[loop_b][segList, :], axis=0))
            visI_boot.append(np.sum(visI_seg[loop_b][segList, :], axis=0))
            visP_boot.append(np.sum(visP_seg[loop_b][segList, :], axis=0))
            f12_boot.append(np.sum(f12_seg[loop_b][segList, :], axis=0))
            f1f2_boot.append(np.sum(f1f2_seg[loop_b][segList, :], axis=0))
            f1f2_boot[-1][f1f2_boot[-1] < 0.0] = 1e-15
        visR_boot = np.array(visR_boot)
        visI_boot = np.array(visI_boot)
        visP_boot = np.array(visP_boot)
        f1f2_boot = np.array(f1f2_boot)
        f12_boot  = np.array(f12_boot)
        #--> Calculate the final physical quantities: visamp, vis2 and visphi
        visamp_boot = np.sqrt(visR_boot**2 + visI_boot**2) / f12_boot
        vis2_boot   = visP_boot / f1f2_boot
        visphi_boot = np.angle(visR_boot + 1j*visI_boot)
        visphi_boot[1:, :] = np.angle(np.exp(1j*(visphi_boot[0, :]-visphi_boot[1:, :])))
        visamp.append(visamp_boot[0, :])
        visphi.append(visphi_boot[0, :])
        vis2.append(vis2_boot[0, :])
        visR.append(visR_boot[0, :])
        visI.append(visI_boot[0, :])
        visamp_err.append(np.std(visamp_boot, axis=0, ddof=1))
        visphi_err.append(np.std(visphi_boot, axis=0, ddof=1) * 180./np.pi)
        vis2_err.append(np.std(vis2_boot, axis=0, ddof=1))
        visR_err.append(np.std(visR_boot, axis=0, ddof=1))
        visI_err.append(np.std(visI_boot, axis=0, ddof=1))
        #--> Calculate the coordinate and integration time
        inttime_total = np.sum(inttime_seg[loop_b])
        u_mean = np.sum(ucoord_seg[loop_b] * inttime_seg[loop_b]) / inttime_total
        v_mean = np.sum(vcoord_seg[loop_b] * inttime_seg[loop_b]) / inttime_total
        u_mas.append(u_mean / wavelength / (180. / np.pi * 3.6e6))
        v_mas.append(v_mean / wavelength / (180. / np.pi * 3.6e6))
    scivis_dict = {
        "visamp": np.array(visamp),
        "visphi": np.array(visphi),
        "vis2": np.array(vis2),
        "visR": np.array(visR),
        "visI": np.array(visI),
        "visamp_err": np.array(visamp_err),
        "visphi_err": np.array(visphi_err),
        "vis2_err": np.array(vis2_err),
        "visR_err": np.array(visR_err),
        "visI_err": np.array(visI_err),
        "u_mas": np.array(u_mas),
        "v_mas": np.array(v_mas),
        "inttime_total": inttime_total,
        "fltr_time": fltr_time,
        "rejection_flag": rejflag,
        "visloss": visloss,
    }
    return scivis_dict
