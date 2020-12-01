import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.nddata import CCDData
from .utils import get_model_from_header

__all__ = ['imgblock_standard']

class imgblock_standard(object):
    '''
    GALFIT standard output.  E.g., output from `galfit -o2 <file> (standard img. block)`
    '''
    def __init__(self, filename, unit='adu'):
        '''
        Parameters
        ----------
        filename : string
            Path of the GALFIT output file.
        '''
        self.extensions = []
        for loop in range(3):
            self.extensions.append(CCDData.read(filename, hdu=loop+1, unit=unit))

        self.model_info = get_model_from_header(self.extensions[1].header)

    def plot_extension(self, n):
        '''
        Plot one extenstion.
        '''
