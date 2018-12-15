import numpy as np

__all__ = ["gravi_vis_average_bootstrap"]

def gravi_vis_average_bootstrap(p2vmred_dict, nboot=20, nseg=100):
    """
    Calculate average the P2VMRED data to obtain the SCIVIS data.  The function
    works with the same procedure as the function gravi_vis_average_bootstrap()
    in gvraid2:/home/grav/SVN/gravity_dev/gravityp/gravi/gravi_vis.c
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
    nf, nb = rejflag.shape
    nc = visdata.shape[2]
    for loop_b in range(nb):
        fltr = rejflag[:, loop_b] == 0
        nr = np.sum(fltr)
        nrow = nr / nseg
        nrange = nrow * nseg
        #
        data = visdata[fltr, loop_b, :]
        visdata_rs = data[:nrange, :].reshape(-1, nseg, nc)
        #
        data = viserr[fltr, loop_b, :]
        viserr_rs = data[:nrange, :].reshape(-1, nseg, nc)
        #
        data = selref[fltr, loop_b, :]
        selref_rs = data[:nrange, :].reshape(-1, nseg, nc)
        #
        data = f1f2[fltr, loop_b, :]
        f1f2_rs = data[:nrange, :].reshape(-1, nseg, nc)
        f12sq_rs = f1f2_rs.copy() # Exactly following the pipeline
        f12sq_rs[f12sq_rs < 1e-15] = 0.0
        #
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
        #print segList
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
    visamp_boot = np.sqrt(visR_boot**2 + visI_boot**2) / f12_boot
    vis2_boot   = visP_boot / f1f2_boot
    scivis_dict = {
        "visamp": visamp_boot[0, :, :],
        "vis2": vis2_boot[0, :, :],
        "visR": visR_boot[0, :, :],
        "visI": visI_boot[0, :, :],
        "visamp_std": np.std(visamp_boot, axis=0, ddof=1),
        "vis2_std" : np.std(vis2_boot, axis=0, ddof=1),
        "visR_std" : np.std(visR_boot, axis=0, ddof=1),
        "visI_std" : np.std(visI_boot, axis=0, ddof=1),
    }
    return scivis_dict
