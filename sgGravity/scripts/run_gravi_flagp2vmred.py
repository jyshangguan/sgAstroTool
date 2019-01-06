import os
import glob
import numpy as np
import sgGravity
from optparse import OptionParser

#->Parse the commands
parser = OptionParser()
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
(options, args) = parser.parse_args()
fileName  = options.filename
dirName   = options.dirname
subName   = options.subname
gdly_thrd = eval(options.gdly_thrd) # Threshold of GDELAY
f1f2_thrd = eval(options.f1f2_thrd) # Threshold of F1F2
visloss   = options.visloss # Flag to correct the visibility loss
overwrite = options.overwrite # Overwrite the FITS file
verbose   = not options.quite # Print the auxiliary information

#-> Check the number of data
fileList = sorted(glob.glob(fileName))
nFile = len(fileList)
if nFile == 0:
    raise ValueError("[GRAV_FLAG: ERROR] There is not file found with {0}.".format(fileName))
else:
    if verbose:
        print("[GRAV_FLAG: NOTICE] There are {0} files found!".format(nFile))

#-> Setup the directory to save the data
if dirName is None:
    dirName = "{0}/flagged".format("/".join(fileName.split("/")[:-1]))
if not os.path.isdir(dirName):
    os.mkdir(dirName)
elif len(glob.glob("{0}/*".format(dirName))) > 0:
    if overwrite:
        if verbose:
            print("[GRAV_FLAG: WARNING] The target directory is not empty but we may overwrite the files!")
    else:
        raise RuntimeError("[GRAV_FLAG: ERROR] The target directory exists and is not empty!\n  {0}/".format(dirName))
else:
    pass

#-> Setup the subscript of the output file
if subName is None:
    subName = "flagged"

#-> Setup the GDELAY threshold
if gdly_thrd is None:
    gdly_thrd = 40 # micron

#-> Summarize the treatments
if verbose:
    print("[GRAV_FLAG: SUMMARY]")
    print("  {0:>17}: {1}".format("Target DIR", dirName))
    print("  {0:>17}: {1}".format("Subscript name", subName))
    print("  {0:>17}: {1}".format("GDELAY threshold", gdly_thrd))
    print("  {0:>17}: {1}".format("F1F2 threshold", f1f2_thrd))
    print("  {0:>17}: {1}".format("Correct vis. loss", visloss))
    print("  {0:>17}: {1}".format("Overwrite", overwrite))

#-> Load the file
nchn = 5
gp2vmSet = sgGravity.GravitySet(file_list=fileList)
count = 0
for gp2vm in gp2vmSet:
    #-> Revise the rejection_flag
    rejflag = gp2vm.get_data("oi_vis:rejection_flag", insname="ft")
    #--> Select on GDELAY
    gdlyList = gp2vm.get_data("oi_vis:gdelay", insname="ft") * 1e6 # Convert the units to micron
    fltr_gooddata = np.abs(gdlyList) < gdly_thrd
    #--> Select on F1F2
    if f1f2_thrd is None:
        pass # No selection on F1F2
    else:
        f1f2List = gp2vm.get_data("oi_vis:f1f2", insname="ft")
        fltr_f1f2 = np.sum((f1f2List[:, :, 1:-1] > f1f2_thrd), axis=2) == (nchn - 2) # Ignore the two channels on the end
        fltr_gooddata = fltr_gooddata & fltr_f1f2
    rejflag[~fltr_gooddata] = 999 # Asign the non-gooddata a non-zero flag
    gp2vm.update_data("oi_vis:rejection_flag", rejflag.reshape(-1), insname="ft")
    #-> Correct the visibility loss
    if visloss:
        if verbose:
            print("[GRAV_FLAG: NOTICE] Correct the visibility loss!")
        visdata = sgGravity.correct_visdata(gp2vm)
        gp2vm.update_data("oi_vis:visdata", visdata.reshape(-1, nchn), insname="ft")
    #-> Write the file
    fitsName = gp2vm.filename()
    fitsName = ".".join(fitsName.split("/")[-1].split(".")[:-1])
    writeName = "{0}/{1}_{2}.fits".format(dirName, fitsName, subName)
    gp2vm.writeto(writeName, overwrite=True)
    if verbose:
        count += 1
        print("[GRAV_FLAG: NOTICE] {0}/{1} is finished!".format(count, nFile))
