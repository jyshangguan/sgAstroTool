import sys
import os
import numpy as np
from astropy.io import fits
from glob import glob
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Change the keyword of the FITS file')
    parser.add_argument('files', type=str, nargs='*', default=None, help='The filename to be modified.')
    args = parser.parse_args()

    if len(args.files) == 0:
        args.files = glob('./*.fits')

    for f in args.files:
        print('Updating {} for MiRA...'.format(f))
        hdl = fits.open(f, mode='update')
        hdl[0].header['ESO FT ROBJ NAME'] = 'target_ft'
        hdl[0].header['ESO INS SOBJ NAME'] = 'target_sc'

        # Change the revision number in the header of OI_ARRAY
        # to 1 since we dont have FOV or FOVTYPE
        hdl['OI_ARRAY'].header['OI_REVN'] = 1
        hdl['OI_TARGET'].data['TARGET'][0] = 'target_ft'
        hdl['OI_TARGET'].data['TARGET'][1] = 'target_sc'
            
        for ext in hdl[1:]:
            if ext.header['EXTNAME'] == 'OI_FLUX':
                
                # Change the column name in OI_FLUX from FLUX to FLUXDATA
                ext.header['TTYPE5'] = 'FLUXDATA'
            
        hdl.flush()
