import os
import glob
import numpy as np
import datetime
from optparse import OptionParser
import sgGravity
now = datetime.datetime.now()

#->Parse the commands
usage = """Flag the P2VMRED data based on GDELAY and (optionally) F1F2.

       %prog [options] arg1 arg2
"""
parser = OptionParser(usage=usage)
parser.add_option("-n", "--filename", dest="filename", default="./*p2vmred.fits",
                  help="The files to be processed.")
parser.add_option("-d", "--dirname", dest="dirname", default="./flagged",
                  help="The name of the directory to save the flagged data.")
parser.add_option("-s", "--subname", dest="subname", default="flagged",
                  help="The subscript of the name of the flagged data.")
parser.add_option("-g", "--gdelay", dest="gdly_thrd", default="40",
                  help="The threshold (unit: micron) to select the DITs based on the group delay (GDELAY).")
parser.add_option("-f", "--f1f2", dest="f1f2_thrd", default=None,
                  help="The threshold to select the DITs based on the geometric flux (F1F2).")
parser.add_option("-c", "--correct", dest="visloss", action="store_true", default=False,
                  help="Correct the visibility loss according to the group delay (GDELAY).")
parser.add_option("-o", "--overwrite", dest="overwrite", action="store_true", default=False,
                  help="Overwrite the fits file.")
parser.add_option("-q", "--quite", dest="quite", action="store_true", default=False,
                  help="Stop print auxiliary information.")
parser.add_option("--flag-code", dest="flagcode", default="999",
                  help="The code used to flag the data.  It should work with any positive int.")
parser.add_option("--nolog", dest="logflag", action="store_false", default=True,
                  help="Turn off the log.")
parser.add_option("--ignore-channels", dest="channels", default=None,
                  help="Provide the list of channel indices (0~4) to be ignored. Example: '[0, 4]'.")
#--> Setup the options
(options, args) = parser.parse_args()
fileName  = options.filename
dirName   = options.dirname
subName   = options.subname
gdly_thrd = eval(options.gdly_thrd) # Threshold of GDELAY
f1f2_thrd = options.f1f2_thrd # Threshold of F1F2
visloss   = options.visloss # Flag to correct the visibility loss
overwrite = options.overwrite # Overwrite the FITS file
verbose   = not options.quite # Print the auxiliary information
flagcode  = eval(options.flagcode) # The code of flagged data here
#---> Setup the log
logflag   = options.logflag
if logflag:
    logList = ["The data are flagged on: {0}.".format(now.now())]
else:
    logList = None
#---> Generate the list of ignored channels
if options.channels is None:
    ignored_channels = []
else:
    exec("ignored_channels = {0}".format(options.channels))

#-> Check the number of data
fileList = sorted(glob.glob(fileName))
nFile = len(fileList)
if nFile == 0:
    raise ValueError("[GRAV_FLAG: ERROR] There is not file found with {0}.".format(fileName))
else:
    logtext = "[GRAV_FLAG: NOTICE] There are {0} files found!".format(nFile)
    if logflag:
        logList.append(logtext)
    if verbose:
        print(logtext)

#-> Setup the directory to save the data
if not os.path.isdir(dirName):
    os.mkdir(dirName)
elif len(glob.glob("{0}/*".format(dirName))) > 0:
    if overwrite:
        logtext = "[GRAV_FLAG: WARNING] The target directory is not empty but we may overwrite the files!"
        if logflag:
            logList.append(logtext)
        if verbose:
            print(logtext)
    else:
        raise RuntimeError("[GRAV_FLAG: ERROR] The target directory exists and is not empty!\n  {0}/".format(dirName))
else:
    pass

#-> Setup the f1f2_thrd
if not f1f2_thrd is None:
    f1f2_thrd = eval(f1f2_thrd)

#-> Summarize the treatments
if logflag:
    logList.append("[GRAV_FLAG: SUMMARY]")
    logList.append("  {0:>17}: {1}".format("Target DIR", dirName))
    logList.append("  {0:>17}: {1}".format("Subscript name", subName))
    logList.append("  {0:>17}: {1}".format("GDELAY threshold", gdly_thrd))
    logList.append("  {0:>17}: {1}".format("F1F2 threshold", f1f2_thrd))
    logList.append("  {0:>17}: {1}".format("Correct vis. loss", visloss))
    logList.append("  {0:>17}: {1}".format("Overwrite", overwrite))
    logList.append("  {0:>17}: {1}".format("Flag code", flagcode))
    logList.append("  {0:>17}: {1}".format("Ignored channels", ", ".join(np.array(ignored_channels, dtype="str"))))
if verbose:
    print("[GRAV_FLAG: SUMMARY]")
    print("  {0:>17}: {1}".format("Target DIR", dirName))
    print("  {0:>17}: {1}".format("Subscript name", subName))
    print("  {0:>17}: {1}".format("GDELAY threshold", gdly_thrd))
    print("  {0:>17}: {1}".format("F1F2 threshold", f1f2_thrd))
    print("  {0:>17}: {1}".format("Correct vis. loss", visloss))
    print("  {0:>17}: {1}".format("Overwrite", overwrite))
    print("  {0:>17}: {1}".format("Flag code", flagcode))
    print("  {0:>17}: {1}".format("Ignored channels", ", ".join(np.array(ignored_channels, dtype="str"))))

def flagp2vmred(gp2vm, insname, gdly_thrd, f1f2_thrd=None, loglist=None, visloss=False,
                flagcode=999, ignoredchannels=None, verbose=False):
    """
    Flag the p2vmred data.

    Parameters
    ----------
    gp2vm : GravityData (P2VMRED)
        The Gravity P2VMRED data.
    insname : string
        The instrument name, "ft", "sc", or their polarization components.
    gdly_thrd : float
        The threshold to make selection based on GDELAY, units: micron.
    f1f2_thrd : float (optional)
        The threshold to make selection based on F1F2, units: micron.
    loglist : list (optional)
        The list to record all the log information.
    visloss : bool, default: False
        To correct the visibility loss if True.
    flagcode : int, default: 999
        The number to be replaced into the REJECTION_FLAG array for the flagged
        data.
    ignoredchannels : list (optional)
        The list of channels ignored in selecting the data if the channel
        dimension is involved, as the REJECTION_FLAG do not have the dimension
        of channel.
    verbose : bool, default: False
        Print the auxiliary data if True.

    Returns
    -------
    gp2vm : GravityData (P2VMRED)
        The Gravity P2VMRED data after the REJECTION_FLAG array of FT data revised.
    loglist : list (optional)
        The list to record all the log information.
    """
    fitsName = gp2vm.filename().split("/")[-1]
    logtext = "[GRAV_FLAG: NOTICE] Processing {0} data of {1}...".format(insname, fitsName)
    if verbose:
        print(logtext)
    if not loglist is None:
        loglist.append(logtext)
    #-> Get the headers
    prihdr = gp2vm.get_extension("primary", insname="aux")[0].header
    vishdr = gp2vm.get_extension("oi_vis", insname=insname)[0].header
    #-> Revise the rejection_flag
    rejflag = gp2vm.get_data("oi_vis:rejection_flag", insname=insname)
    #--> Select on GDELAY
    if verbose:
        print("[GRAV_FLAG: NOTICE] |GDELAY|<{0}".format(gdly_thrd))
    gdlyList = gp2vm.get_data("oi_vis:gdelay", insname=insname) * 1e6 # Convert the units to micron
    fltr_gooddata = np.abs(gdlyList) < gdly_thrd
    prihdr["history"] = "OI_VIS:REJECTION_FLAG ({0}) is revised to select |GDELAY|<{1} micron.".format(insname, gdly_thrd)
    vishdr["history"] = "REJECTION_FLAG is revised to select |GDELAY|<{0} micron.".format(gdly_thrd)
    #--> Setup the included channels
    nchn = sgGravity.gravity_DimLen("CHANNEL_FT")
    includechannels = range(nchn)
    if not ignoredchannels is None:
        for cn in ignoredchannels:
            includechannels.remove(cn)
    #--> Select on F1F2
    if f1f2_thrd is None:
        pass # No selection on F1F2
    else:
        if verbose:
            print("[GRAV_FLAG: NOTICE] F1F2>{0}".format(f1f2_thrd))
        f1f2List = gp2vm.get_data("oi_vis:f1f2", insname=insname)
        fltr_f1f2 = np.sum((f1f2List[:, :, includechannels] > f1f2_thrd), axis=2) == len(includechannels)
        fltr_gooddata = fltr_gooddata & fltr_f1f2
        prihdr["history"] = "OI_VIS:REJECTION_FLAG ({0}) is revised to select F1F2>{1}.".format(insname, f1f2_thrd)
        vishdr["history"] = "REJECTION_FLAG is revised to select F1F2>{0}.".format(f1f2_thrd)
    logtext = "[GRAV_FLAG: NOTICE] Included DITs in each baseline: {0}".format(", ".join(np.sum(fltr_gooddata, axis=0).astype("str")))
    if not loglist is None:
        loglist.append(logtext)
    if verbose:
        print(logtext)
    rejflag[~fltr_gooddata] = flagcode # Asign the non-gooddata a non-zero flag.
    gp2vm.update_data("oi_vis:rejection_flag", rejflag.reshape(-1), insname=insname)
    #-> Correct the visibility loss
    if visloss:
        if verbose:
            print("[GRAV_FLAG: NOTICE] Correct the visibility loss!")
        visdata = sgGravity.correct_visdata(gp2vm)
        gp2vm.update_data("oi_vis:visdata", visdata.reshape(-1, nchn), insname=insname)
        prihdr["history"] = "OI_VIS:VISDATA ({0}) is revised to correct the visibility loss.".format(insname)
        vishdr["history"] = "VISDATA is revised to correct the visibility loss."
    return gp2vm, loglist


gp2vmSet = sgGravity.GravitySet(file_list=fileList)
count = 0
for gp2vm in gp2vmSet:
    if gp2vm.polamode == "COMBINED":
        gp2vm, logList = flagp2vmred(gp2vm, insname="ft", gdly_thrd=gdly_thrd, f1f2_thrd=f1f2_thrd,
                                     loglist=logList, visloss=visloss, flagcode=flagcode,
                                     ignoredchannels=ignored_channels, verbose=verbose)
    elif gp2vm.polamode == "SPLIT":
        for insname in ["FT_P1", "FT_P2"]:
            gp2vm, logList = flagp2vmred(gp2vm, insname=insname, gdly_thrd=gdly_thrd,
                                         f1f2_thrd=f1f2_thrd, loglist=logList, visloss=visloss,
                                         flagcode=flagcode, ignoredchannels=ignored_channels,
                                         verbose=verbose)
    else:
        raise KeyError("Cannot recognize the polarization mode ({0})!".format(gp2vm.polamode))
    #-> Write the file
    fitsName = gp2vm.filename()
    fitsName = ".".join(fitsName.split("/")[-1].split(".")[:-1])
    writeName = "{0}/{1}_{2}.fits".format(dirName, fitsName, subName)
    gp2vm.writeto(writeName, overwrite=True)
    count += 1
    logtext = "[GRAV_FLAG: NOTICE] {0}/{1} is finished!".format(count, nFile)
    if logflag:
        logList.append(logtext)
    if verbose:
        print(logtext)

#-> write the log
if logflag:
    with open("{0}/gravi_flag.log".format(dirName), "w") as f:
        for li in logList:
            f.write("{0}\n".format(li))
