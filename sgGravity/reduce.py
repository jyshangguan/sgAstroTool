import numpy as np

__all__ = ["gravi_vis_average_bootstrap"]

def gravi_vis_average_bootstrap(p2vmred_dict, nboot=20, nseg_default=100):
    """
    Calculate average the P2VMRED data to obtain the SCIVIS data.  The function
    works with the same procedure as the function gravi_vis_average_bootstrap()
    in gvraid2:/home/grav/SVN/gravity_dev/gravityp/gravi/gravi_vis.c

    Parameters
    ----------
    p2vmred_dict : dict
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
    rejflag = p2vmred_dict["rejection_flag"]
    visdata = p2vmred_dict["visdata"]
    viserr  = p2vmred_dict["viserr"]
    selref  = p2vmred_dict["self_ref"]
    f1f2    = p2vmred_dict["f1f2"]
    #-> Segment the data into nseg blocks
    visR_seg = []
    visI_seg = []
    visP_seg = []
    f12_seg  = []
    f1f2_seg = []
    nfrm, nbase, nchn = visdata.shape # Get the number of frames, baselines and channels
    for loop_b in range(nbase):
        fltr = rejflag[:, loop_b] == 0
        nvalid = np.sum(fltr)
        #--> Wisely choose the number of rows in each segment, 1 row per segment
        #if the number of frames is less than nseg_default.
        nrow_per_seg = np.max([nvalid / np.min([nfrm, nseg_default]), 1])
        nseg = int(nvalid / nrow_per_seg) # The number of segment finally used.
        ncut = nrow_per_seg * nseg
        #--> Reshape the data into segments, (nrow_per_seg, nseg, nchn)
        visdata_rs = visdata[fltr, loop_b, :][:ncut, :].reshape(nrow_per_seg, nseg, nchn)
        viserr_rs = viserr[fltr, loop_b, :][:ncut, :].reshape(nrow_per_seg, nseg, nchn)
        selref_rs = selref[fltr, loop_b, :][:ncut, :].reshape(nrow_per_seg, nseg, nchn)
        f1f2_rs = f1f2[fltr, loop_b, :][:ncut, :].reshape(nrow_per_seg, nseg, nchn)
        f12sq_rs = f1f2_rs.copy() # Exactly following the pipeline
        f12sq_rs[f12sq_rs < 1e-15] = 0.0
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
    visR_seg = np.array(visR_seg)
    visI_seg = np.array(visI_seg)
    visP_seg = np.array(visP_seg)
    f1f2_seg = np.array(f1f2_seg)
    f12_seg  = np.array(f12_seg)
    #-> Bootstrapping
    visR_boot = []
    visI_boot = []
    visP_boot = []
    f1f2_boot = []
    f12_boot  = []
    for boot in range(nboot):
        if boot == 0:
            segList = np.arange(nseg)
        else:
            segList = np.random.randint(nseg, size=nseg)
        visR_boot.append(np.sum(visR_seg[:, segList, :], axis=1))
        visI_boot.append(np.sum(visI_seg[:, segList, :], axis=1))
        visP_boot.append(np.sum(visP_seg[:, segList, :], axis=1))
        f12_boot.append(np.sum(f12_seg[:, segList, :], axis=1))
        f1f2_boot.append(np.sum(f1f2_seg[:, segList, :], axis=1))
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
    visphi_boot[1:, :, :] = np.angle(np.exp(1j*(visphi_boot[0, :, :]-visphi_boot[1:, :, :])))
    scivis_dict = {
        "visamp": visamp_boot[0, :, :],
        "visphi": visphi_boot[0, :, :] * 180./np.pi,
        "vis2": vis2_boot[0, :, :],
        "visR": visR_boot[0, :, :],
        "visI": visI_boot[0, :, :],
        "visamp_std": np.std(visamp_boot, axis=0, ddof=1),
        "visphi_std": np.std(visphi_boot, axis=0, ddof=1) * 180./np.pi,
        "vis2_std" : np.std(vis2_boot, axis=0, ddof=1),
        "visR_std" : np.std(visR_boot, axis=0, ddof=1),
        "visI_std" : np.std(visI_boot, axis=0, ddof=1),
    }
    return scivis_dict
