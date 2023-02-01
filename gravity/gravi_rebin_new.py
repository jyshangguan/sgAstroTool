#!/usr/bin/env python3
'''
Rebin GRAVITY data in the spectral dimension

To rebin all files in a directory:

  ./gravi_rebin_new.py

The script accepts file names as input arguments and will only rebin
those files if provided.

By default, the output file name is made from the input file name
appending _rebinned_<binsize>.

See ./gravi_rebin_new.py --help for useful options.

This is a revised rebin script based on gravi_rebin.py

'''
# Changelog:
# 2023-01-17 shangguan  Create the script


import glob
import copy
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from astropy.convolution import Gaussian1DKernel, convolve

import matplotlib as mpl
mpl.rc("xtick", direction="in", labelsize=16)
mpl.rc("ytick", direction="in", labelsize=16)
mpl.rc("xtick.major", width=1., size=8)
mpl.rc("ytick.major", width=1., size=8)
mpl.rc("xtick.minor", width=1., size=5)
mpl.rc("ytick.minor", width=1., size=5)

import argparse


def filter_files(fn):
    '''True if fn is the name of a suitable file'''
    # Remove files that can't be opened as FITS
    try:
        header=fits.open(fn)[0].header
    except (IOError):
        try:
            header=fits.open("./"+fn)[0].header
        except (IOError):
            return False
    # All good, valid file
    return True


class GRAVITY_data(object):
    '''
    GRAVITY data.
    '''
    def __init__(self, filename):
        '''
        Load the data.
        '''
        try:
            self.hdul = fits.open(filename)
        except(IOError):
            filename = "./" + filename
            self.hdul = fits.open(fname)

        self.header = self.hdul[0].header

        if self.header['TELESCOP'] == 'ESO-VLTI-U1234':
            self.tel = 'UTS'
        elif self.header['TELESCOP'] == 'ESO-VLTI-A1234':
            self.tel = 'ATS'
        else:
            raise KeyError('Cannot recognize TELESCOP: {}!'.format(self.header['TELESCOP']))
        
        self.pol = self.header.get('ESO INS POLA MODE', None)
        if self.pol not in ['SPLIT', 'COMBINED']:
            raise KeyError('Cannot recognize ESO INS POLA MODE: {}!'.format(self.header['ESO INS POLA MODE']))

        self.res = self.header['ESO INS SPEC RES']
        if self.res not in ['LOW', 'MEDIUM', 'HIGH']:
            raise KeyError('Cannot recognize ESO INS SPEC RES: {}!'.format(self.header['ESO INS SPEC RES']))
        
        # Split the file name
        self.filename = filename.split('/')[-1]

        # GRAVITY data labels
        self.telescope_name = ['UT4', 'UT3', 'UT2', 'UT1']
        self.baseline_name = ['UT4$-$UT3', 'UT4$-$UT2', 'UT4$-$UT1', 'UT3$-$UT2', 'UT3$-$UT1', 'UT2$-$UT1']
        self.triangle_name = ['UT4$-$UT3$-$UT2', 'UT4$-$UT3$-$UT1', 'UT4$-$UT2$-$UT1', 'UT3$-$UT2$-$UT1']
        self.polar_name = ['P', 'S']


    def get_extensions(self):
        '''
        Get the extension list.
        '''
        if self.pol == 'COMBINED':
            exts = [10]
        elif self.pol == 'SPLIT':
            exts = [11, 12]
        else:
            raise KeyError('Cannot recognize self.pol ({})!'.format(self.pol))
        
        return exts


    def plot_OI_FLUX(self, axs=None, **kwargs):
        '''
        Plot the FLUX of the OI_FLUX data.

        Parameters
        ----------
        axs : Axes
            Axes to plot the VIS2DATA.
        **kwargs : parameters to plt.plot()

        Returns
        -------
        axs : Axes
            Axes plotting the VIS2DATA.
        '''
        if axs is None:
            if self.pol == 'COMBINED':
                fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True, sharey=True)
                axs = axs.reshape(4, 1)
                fig.subplots_adjust(hspace=0.05)
            else:
                fig, axs = plt.subplots(4, 2, figsize=(20, 12), sharex=True, sharey=True)
                fig.subplots_adjust(wspace=0.01, hspace=0.05)

            axo = fig.add_subplot(111, frameon=False) # The out axis
            axo.tick_params(axis='y', which='both', left=False, labelleft=False)
            axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            axo.set_xlabel(r"Wavelength ($\mu$m)", fontsize=24, labelpad=24)
            axo.set_title(r"FLUX", fontsize=24)

            plain = False
        else:
            plain = True
            
        if 'marker' not in kwargs:
            kwargs['marker'] = '.'
        
        for loop_x, ext in enumerate(self.get_extensions()):
            wave = self.hdul['OI_WAVELENGTh', ext].data['EFF_WAVE'] * 1e6
            vis2 = self.hdul['OI_FLUX', ext].data['FLUX']

            for idx, ax in enumerate(axs[:, loop_x]):
                ax.plot(wave, vis2[idx, :], **kwargs)

            if not plain:
                for idx, ax in enumerate(axs[:, loop_x]):
                    if self.pol == 'SPLIT':
                        txt = f'{self.telescope_name[idx]} ({self.polar_name[loop_x]})'
                    else:
                        txt = f'{self.telescope_name[idx]}'
                    ax.text(0.02, 0.95, txt, fontsize=16, va='top', ha='left', transform=ax.transAxes)

                    ax.minorticks_on()

        return axs


    def plot_OI_VIS(self, axs=None, **kwargs):
        '''
        Plot the VISAMP and VISPHI of the OI_VIS data.

        Parameters
        ----------
        axs : Axes
            Axes to plot the VISAMP (left) and VISPHI (right).
        **kwargs : parameters to plt.plot()

        Returns
        -------
        axs : Axes
            Axes plotting the VISAMP (left) and VISPHI (right).
        '''
        if axs is None:
            if self.pol == 'COMBINED':
                fig, axs = plt.subplots(6, 2, figsize=(20, 12), sharex=True, sharey=False)
                axs = axs.reshape(6, 2, 1)
                fig.subplots_adjust(wspace=0.1, hspace=0.05)
            else:
                fig, axs = plt.subplots(6, 4, figsize=(20, 12), sharex=True, sharey=False)
                axs = axs.reshape(6, 2, 2).swapaxes(1, 2)
                fig.subplots_adjust(wspace=0.2, hspace=0.05)

            axo = fig.add_subplot(111, frameon=False) # The out axis
            axo.tick_params(axis='y', which='both', left=False, labelleft=False)
            axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            axo.set_xlabel(r"Wavelength ($\mu$m)", fontsize=24, labelpad=24)
            axo.set_title(r"VISAMP & VISPHI ($^\circ$)", fontsize=24)

            plain = False
        else:
            plain = True
            
        if 'marker' not in kwargs:
            kwargs['marker'] = '.'
        
        for loop_x, ext in enumerate(self.get_extensions()):
            wave = self.hdul['OI_WAVELENGTh', ext].data['EFF_WAVE'] * 1e6
            vamp = self.hdul['OI_VIS', ext].data['VISAMP']
            vphi = self.hdul['OI_VIS', ext].data['VISPHI']

            for idx, ax in enumerate(axs[:, 0, loop_x]):
                ax.plot(wave, vamp[idx, :], **kwargs)
        
            for idx, ax in enumerate(axs[:, 1, loop_x]):
                ax.plot(wave, vphi[idx, :], **kwargs)

            if not plain:
                for idx, ax in enumerate(axs[:, 0, loop_x]):
                    if self.pol == 'SPLIT':
                        txt = f'{self.baseline_name[idx]} ({self.polar_name[loop_x]})'
                    else:
                        txt = f'{self.baseline_name[idx]}'
                    ax.text(0.02, 0.95, txt, fontsize=16, va='top', ha='left', transform=ax.transAxes)

                    ax.set_ylim([0, 1.4])
                    ax.minorticks_on()
                
                for idx, ax in enumerate(axs[:, 1, loop_x]):
                    ax.set_ylim([-190, 190])
                    ax.minorticks_on()

        return axs
        

    def plot_OI_VIS2(self, axs=None, **kwargs):
        '''
        Plot the VIS2DATA of the OI_VIS2 data.

        Parameters
        ----------
        axs : Axes
            Axes to plot the VIS2DATA.
        **kwargs : parameters to plt.plot()

        Returns
        -------
        axs : Axes
            Axes plotting the VIS2DATA.
        '''
        if axs is None:
            if self.pol == 'COMBINED':
                fig, axs = plt.subplots(6, 1, figsize=(12, 12), sharex=True, sharey=True)
                axs = axs.reshape(6, 1)
                fig.subplots_adjust(hspace=0.05)
            else:
                fig, axs = plt.subplots(6, 2, figsize=(20, 12), sharex=True, sharey=True)
                fig.subplots_adjust(wspace=0.01, hspace=0.05)

            axo = fig.add_subplot(111, frameon=False) # The out axis
            axo.tick_params(axis='y', which='both', left=False, labelleft=False)
            axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            axo.set_xlabel(r"Wavelength ($\mu$m)", fontsize=24, labelpad=24)
            axo.set_title(r"VIS2", fontsize=24)

            plain = False
        else:
            plain = True
            
        if 'marker' not in kwargs:
            kwargs['marker'] = '.'
        
        for loop_x, ext in enumerate(self.get_extensions()):
            wave = self.hdul['OI_WAVELENGTh', ext].data['EFF_WAVE'] * 1e6
            vis2 = self.hdul['OI_VIS2', ext].data['VIS2DATA']

            for idx, ax in enumerate(axs[:, loop_x]):
                ax.plot(wave, vis2[idx, :], **kwargs)

            if not plain:
                for idx, ax in enumerate(axs[:, loop_x]):
                    if self.pol == 'SPLIT':
                        txt = f'{self.baseline_name[idx]} ({self.polar_name[loop_x]})'
                    else:
                        txt = f'{self.baseline_name[idx]}'
                    ax.text(0.02, 0.95, txt, fontsize=16, va='top', ha='left', transform=ax.transAxes)

                    ax.set_ylim([0, 1.4])
                    ax.minorticks_on()

        return axs


    def plot_OI_T3(self, axs=None, **kwargs):
        '''
        Plot the T3AMP and T3PHI of the OI_T3 data.

        Parameters
        ----------
        axs : Axes
            Axes to plot the T3AMP (left) and T3PHI (right).
        **kwargs : parameters to plt.plot()

        Returns
        -------
        axs : Axes
            Axes to plot the T3AMP (left) and T3PHI (right).
        '''
        if axs is None:
            if self.pol == 'COMBINED':
                fig, axs = plt.subplots(4, 2, figsize=(20, 12), sharex=True, sharey=False)
                axs = axs.reshape(4, 2, 1)
                fig.subplots_adjust(wspace=0.1, hspace=0.05)
            else:
                fig, axs = plt.subplots(4, 4, figsize=(20, 12), sharex=True, sharey=False)
                axs = axs.reshape(4, 2, 2).swapaxes(1, 2)
                fig.subplots_adjust(wspace=0.2, hspace=0.05)

            axo = fig.add_subplot(111, frameon=False) # The out axis
            axo.tick_params(axis='y', which='both', left=False, labelleft=False)
            axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            axo.set_xlabel(r"Wavelength ($\mu$m)", fontsize=24, labelpad=24)
            axo.set_title(r"T3AMP & T3PHI ($^\circ$)", fontsize=24)

            plain = False
        else:
            plain = True
            
        if 'marker' not in kwargs:
            kwargs['marker'] = '.'
        
        for loop_x, ext in enumerate(self.get_extensions()):
            wave = self.hdul['OI_WAVELENGTh', ext].data['EFF_WAVE'] * 1e6
            t3amp = self.hdul['OI_T3', ext].data['T3AMP']
            t3phi = self.hdul['OI_T3', ext].data['T3PHI']

            for idx, ax in enumerate(axs[:, 0, loop_x]):
                ax.plot(wave, t3amp[idx, :], **kwargs)
        
            for idx, ax in enumerate(axs[:, 1, loop_x]):
                ax.plot(wave, t3phi[idx, :], **kwargs)

            if not plain:
                for idx, ax in enumerate(axs[:, 0, loop_x]):
                    if self.pol == 'SPLIT':
                        txt = f'{self.triangle_name[idx]} ({self.polar_name[loop_x]})'
                    else:
                        txt = f'{self.triangle_name[idx]}'
                    ax.text(0.02, 0.95, txt, fontsize=16, va='top', ha='left', transform=ax.transAxes)

                    ax.set_ylim([0, 1.4])
                    ax.minorticks_on()
                
                for idx, ax in enumerate(axs[:, 1, loop_x]):
                    ax.set_ylim([-190, 190])
                    ax.minorticks_on()

        return axs


    def rebin_all(self, ncombine, nsigma=None):
        '''
        Rebin all components.

        Parameters
        ----------
        ncombine (optional) : int
            Number of channels to be combined.
        '''
        self.rebin_OI_WAVELENGTH(ncombine, nsigma=nsigma)
        self.rebin_OI_FLUX(ncombine, nsigma=nsigma)
        self.rebin_OI_VIS(ncombine, nsigma=nsigma)
        self.rebin_OI_VIS2(ncombine, nsigma=nsigma)
        self.rebin_OI_T3(ncombine, nsigma=nsigma)


    def rebin_OI_WAVELENGTH(self, ncombine, nsigma=None):
        '''
        Rebin the OI_WAVELENGTH data.

        Parameters
        ----------
        ncombine : int
            Number of channels to be combined.
        '''
        for ext in self.get_extensions():
            hdu_data = copy.deepcopy(self.hdul['OI_WAVELENGTH', ext].data)
            dwave = np.diff(hdu_data['EFF_WAVE'])[0]
            wave_bin = rebin_wave(hdu_data['EFF_WAVE'], ncombine)
            hdu_data.resize(len(wave_bin))
            hdu_data['EFF_WAVE'] = wave_bin

            # Revise the EFF_BAND
            if nsigma is not None:
                band = np.sqrt(hdu_data['EFF_BAND'][0]**2 + (2.3548 * nsigma * dwave)**2)
                band_bin = np.ones_like(wave_bin) * band
                hdu_data['EFF_BAND'] = band_bin

            self.hdul['OI_WAVELENGTh', ext].data = hdu_data


    def rebin_OI_FLUX(self, ncombine, nsigma=None):
        '''
        Rebin the OI_FLUX data.

        Parameters
        ----------
        ncombine : int
            Number of channels to be combined.
        '''
        colList = ['FLUX', 'FLUXDATA', 'FLUXERR', 'FLAG']

        for ext in self.get_extensions():
            hdu_data = copy.deepcopy(self.hdul['OI_FLUX', ext].data)
            flag = hdu_data['FLAG']

            try:
                flux = np.ma.array(hdu_data['FLUX'], mask=flag)
                flux_colname = 'FLUX'
            except:
                flux = np.ma.array(hdu_data['FLUXDATA'], mask=flag)
                flux_colname = 'FLUXDATA'

            fluxerr = np.ma.array(hdu_data['FLUXERR'], mask=flag)

            # Smooth if nsigma is provided
            if nsigma is not None:
                flux = smooth(flux, nsigma)

            # Rebin data
            nrecs = flag.shape[1] // ncombine
            new_data = resize_FITS_rec(hdu_data, nrecs, colList)
            flux_bin, fluxerr_bin = rebin_vis(flux, fluxerr, ncombine)
            flag_bin = rebin_flag(flag, ncombine)

            new_data[flux_colname] = flux_bin
            new_data['FLUXERR'] = fluxerr_bin
            new_data['FLAG'] = flag_bin

            self.hdul['OI_FLUX', ext].data = new_data


    def rebin_OI_VIS(self, ncombine, nsigma=None):
        '''
        Rebin the OI_VIS data.

        Parameters
        ----------
        ncombine : int
            Number of channels to be combined.
        '''
        colList = ['VISDATA', 'VISERR', 'RVIS', 'RVISERR', 'IVIS', 'IVISERR',
                   'VISAMP', 'VISAMPERR', 'VISPHI', 'VISPHIERR', 'FLAG']

        for ext in self.get_extensions():
            # Get data
            hdu_data = self.hdul['OI_VIS', ext].data
            flag = hdu_data['FLAG']
            visdata = np.ma.array(hdu_data['VISDATA'], mask=flag)
            viserr = np.ma.array(hdu_data['VISERR'], mask=flag)
            visamp = np.ma.array(hdu_data['VISAMP'], mask=flag)
            visamperr = np.ma.array(hdu_data['VISAMPERR'], mask=flag)
            visphi = np.ma.array(hdu_data['VISPHI'], mask=flag)
            visphierr = np.ma.array(hdu_data['VISPHIERR'], mask=flag)

            # Smooth if nsigma is provided
            if nsigma is not None:
                visdata = smooth(visdata, nsigma)
                visdata_cmb = smooth(visamp * np.exp(1j * visphi * np.pi / 180), nsigma)
                visamp = np.absolute(visdata_cmb)
                visphi = np.angle(visdata_cmb, deg=True)

            # Rebin data
            nrecs = flag.shape[1] // ncombine
            new_data = resize_FITS_rec(hdu_data, nrecs, colList)
            visdata_bin, viserr_bin = rebin_vis(visdata, viserr, ncombine)
            visamp_bin, visamperr_bin, visphi_bin, visphierr_bin = rebin_vis_components(visamp, visamperr, visphi, visphierr, ncombine)
            flag_bin = rebin_flag(flag, ncombine)

            new_data['VISDATA'] = visdata_bin
            new_data['VISERR'] = viserr_bin
            new_data['RVIS'] = visdata_bin.real
            new_data['RVISERR'] = viserr_bin.real
            new_data['IVIS'] = visdata_bin.imag
            new_data['IVISERR'] = viserr_bin.imag
            new_data['VISAMP'] = visamp_bin
            new_data['VISAMPERR'] = visamperr_bin
            new_data['VISPHI'] = visphi_bin
            new_data['VISPHIERR'] = visphierr_bin
            new_data['FLAG'] = flag_bin

            self.hdul['OI_VIS', ext].data = new_data


    def rebin_OI_VIS2(self, ncombine, nsigma=None):
        '''
        Rebin the OI_VIS2 data.

        Parameters
        ----------
        ncombine : int
            Number of channels to be combined.
        '''
        colList = ['VIS2DATA', 'VIS2ERR', 'FLAG']

        for ext in self.get_extensions():
            # Get data
            hdu_data = self.hdul['OI_VIS2', ext].data
            flag = hdu_data['FLAG']
            vis2data = np.ma.array(hdu_data['VIS2DATA'], mask=flag)
            vis2err = np.ma.array(hdu_data['VIS2ERR'], mask=flag)

            # Smooth if nsigma is provided
            if nsigma is not None:
                vis2data = smooth(vis2data, nsigma)

            # Rebin data
            nrecs = flag.shape[1] // ncombine
            new_data = resize_FITS_rec(hdu_data, nrecs, colList)
            vis2data_bin, vis2err_bin = rebin_vis(vis2data, vis2err, ncombine)
            flag_bin = rebin_flag(flag, ncombine)

            new_data['VIS2DATA'] = vis2data_bin
            new_data['VIS2ERR'] = vis2err_bin
            new_data['FLAG'] = flag_bin

            self.hdul['OI_VIS2', ext].data = new_data


    def rebin_OI_T3(self, ncombine, nsigma=None):
        '''
        Rebin the OI_T3 data.

        Parameters
        ----------
        ncombine : int
            Number of channels to be combined.
        '''
        colList = ['T3AMP', 'T3AMPERR', 'T3PHI', 'T3PHIERR', 'FLAG']

        for ext in self.get_extensions():
            # Get data
            hdu_data = self.hdul['OI_T3', ext].data
            flag = hdu_data['FLAG']
            t3amp = np.ma.array(hdu_data['T3AMP'], mask=flag)
            t3amperr = np.ma.array(hdu_data['T3AMPERR'], mask=flag)
            t3phi = np.ma.array(hdu_data['T3PHI'], mask=flag)
            t3phierr = np.ma.array(hdu_data['T3PHIERR'], mask=flag)

            # Smooth if nsigma is provided
            if nsigma is not None:
                t3data = smooth(t3amp * np.exp(1j * t3phi * np.pi / 180), nsigma)
                t3amp = np.absolute(t3data)
                t3phi = np.angle(t3data, deg=True)

            # Rebin data
            nrecs = flag.shape[1] // ncombine
            new_data = resize_FITS_rec(hdu_data, nrecs, colList)
            t3amp_bin, t3amperr_bin, t3phi_bin, t3phierr_bin = rebin_t3(t3amp, t3amperr, t3phi, t3phierr, ncombine)
            flag_bin = rebin_flag(flag, ncombine)

            new_data['T3AMP'] = t3amp_bin
            new_data['T3AMPERR'] = t3amperr_bin
            new_data['T3PHI'] = t3phi_bin
            new_data['T3PHIERR'] = t3phierr_bin
            new_data['FLAG'] = flag_bin

            self.hdul['OI_T3', ext].data = new_data


    def save(self, filename, overwrite=False):
        '''
        Save the FITS file.
        '''
        self.hdul.writeto(filename, overwrite=overwrite)


def rebin_average(data, ncombine, error=None):
    '''
    Rebin the data by averaging.
    
    Parameters
    ----------
    data : 2D array
        Data to be rebinned.  Assume the data to be NxM.  The N dimension 
        is for baselines while the M dimension is for channels.
    ncombine : int
        Number of channels to bin together.
    '''
    nbsl, nchn = data.shape
    
    nnew = nchn // ncombine
    ndrp = nchn - nnew * ncombine
    ndrp_pre = ndrp // 2
    ndrp_end = ndrp - ndrp_pre
    
    data_reshape = data[:, ndrp_pre:-ndrp_end].reshape(nbsl, nnew, ncombine)
    
    if error is None:
        data_rebin = np.average(data_reshape, axis=2)
        error_rebin = None
    else:
        assert error.shape == data.shape, 'The shapes of data and error are not consistent!'
        weight_reshape = error[:, ndrp_pre:-ndrp_end].reshape(nbsl, nnew, ncombine)**-2
        fltr = np.isnan(weight_reshape) | ~np.isfinite(weight_reshape)
        weight_reshape[fltr] = 0
        data_rebin, weight_rebin = np.average(data_reshape, axis=2, weights=weight_reshape, returned=True)
        error_rebin = weight_rebin**(-0.5)
    
    return data_rebin, error_rebin


def rebin_error_simple(error, ncombine):
    '''
    Rebin the error in a simple-mind approach.
    '''
    nbsl, nchn = error.shape
    
    nnew = nchn // ncombine
    ndrp = nchn - nnew * ncombine
    ndrp_pre = ndrp // 2
    ndrp_end = ndrp - ndrp_pre
    
    error_reshape = error[:, ndrp_pre:-ndrp_end].reshape(nbsl, nnew, ncombine)
    error_rebin = np.sqrt((error_reshape**2).sum(axis=2)) / ncombine
    return error_rebin
    

def rebin_wave(wave, ncombine):
    '''
    Rebin the wavelength data.
    '''
    wave_bin = rebin_average(np.atleast_2d(wave), ncombine)[0][0, :]
    return wave_bin


def rebin_flux(flux, fluxerr, ncombine):
    '''
    Rebin the FLUX.
    '''
    flux_bin, fluxerr_bin = rebin_average(flux, ncombine, fluxerr)
    return flux_bin, fluxerr_bin


def rebin_vis(visdata, viserr, ncombine):
    '''
    Rebin the VISDATA.
    '''
    visdata_real, viserr_real = rebin_average(visdata.real, ncombine, viserr.real)
    visdata_imag, viserr_imag = rebin_average(visdata.imag, ncombine, viserr.imag)
    visdata_bin = visdata_real + 1j*visdata_imag
    viserr_bin = viserr_real + 1j*viserr_imag
    return visdata_bin, viserr_bin


def rebin_vis_components(visamp, visamperr, visphi, visphierr, ncombine):
    '''
    Rebin VISAMP and VISPHI.
    '''
    visdata = visamp * np.exp(1j * visphi * np.pi / 180)
    visdata_real, _ = rebin_average(visdata.real, ncombine)
    visdata_imag, _ = rebin_average(visdata.imag, ncombine)
    visdata_bin = visdata_real + 1j*visdata_imag
    
    visamp_bin = np.absolute(visdata_bin)
    visphi_bin = np.angle(visdata_bin, deg=True)
    
    visamperr_bin = rebin_error_simple(visamperr, ncombine)
    visphierr_bin = rebin_error_simple(visphierr, ncombine)
    return visamp_bin, visamperr_bin, visphi_bin, visphierr_bin


def rebin_vis2(vis2data, vis2err, ncombine):
    '''
    Rebin the VIS2DATA.
    '''
    vis2data_bin, vis2err_bin = rebin_average(vis2data, ncombine, vis2err)
    return vis2data_bin, vis2err_bin


def rebin_t3(t3amp, t3amperr, t3phi, t3phierr, ncombine):
    '''
    Rebin the T3.
    '''
    t3data = t3amp * np.exp(1j * t3phi * np.pi / 180)
    t3data_real, _ = rebin_average(t3data.real, ncombine)
    t3data_imag, _ = rebin_average(t3data.imag, ncombine)
    t3data_bin = t3data_real + 1j*t3data_imag
    
    t3amp_bin = np.absolute(t3data_bin)
    t3phi_bin = np.angle(t3data_bin, deg=True)
    
    t3amperr_bin = rebin_error_simple(t3amperr, ncombine)
    t3phierr_bin = rebin_error_simple(t3phierr, ncombine)
    return t3amp_bin, t3amperr_bin, t3phi_bin, t3phierr_bin


def rebin_flag(flag, ncombine):
    '''
    Rebin the flag. Consider the bin good as long as there 
    is one good data in this bin.
    
    Note
    ----
    True means bad data.
    '''
    nbsl, nchn = flag.shape
    
    nnew = nchn // ncombine
    ndrp = nchn - nnew * ncombine
    ndrp_pre = ndrp // 2
    ndrp_end = ndrp - ndrp_pre
    
    flag_reshape = flag[:, ndrp_pre:-ndrp_end].reshape(nbsl, nnew, ncombine)
    flag_rebin = flag_reshape.sum(axis=2) == ncombine
    return flag_rebin


def resize_FITS_rec(table, nrecs, colnames=None):
    '''
    Copied from the old gravi_rebin.py
    
    Duplicate a FITS_rec table an resize some columns

    This method will create a deep copy of a
    astropy.io.fits.fitsrec.FITS_rec table with size NRECS for the
    columns specified in COLNAMES. The resized columns are filled with
    zeroes while the other columns retain their original content.
    '''
    # Duplicate input table with different size for some columns.
    new_coldefs=copy.deepcopy(table.columns.columns)

    # Change size ouf output columns
    snrecs=str(nrecs)
    for c in range(len(new_coldefs)):
        if (new_coldefs[c].name in colnames):
            new_coldefs[c].format = (
                fits.column._ColumnFormat(snrecs+new_coldefs[c].format[-1:]))
            new_coldefs[c].array = None
    return fits.FITS_rec.from_columns(new_coldefs)


def smooth(data, nsigma):
    '''
    Smooth the data in the last dimension.

    Parameters
    ----------
    data : 2D array.
        Data to be smoothed.
    nsigma : float
        Gaussian smooth sigma in the unit of channel.
    '''
    kernel = Gaussian1DKernel(stddev=nsigma)

    if isinstance(np.mean(data), complex):
        data_real = []
        data_imag = []
        for loop in range(data.shape[0]):
            data_real.append(convolve(data[loop, :].real, kernel))
            data_imag.append(convolve(data[loop, :].imag, kernel))
        data_sm = np.ma.array(data_real, mask=data.mask) + 1j * np.ma.array(data_imag, mask=data.mask)
    else:
        data_sm = []
        for loop in range(data.shape[0]):
            data_sm.append(convolve(data[loop, :], kernel))
        data_sm = np.ma.array(data_sm, mask=data.mask)

    return data_sm


if __name__ == '__main__':
    # Mainly follow the old script
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     prefix_chars='-+')
    parser.add_argument('fname', nargs='*')
    parser.add_argument('-b', '--binsize', help=
                        'Number of wavelength bins to group together [16]',
                        type=int, default=16)
    parser.add_argument('-s', '--sigma', help=
                        'Number of channels as the sigma of the Gaussian kernel. Do not convolve if 0 and match LOW if 23.5 [0]',
                        type=float, default=0)
    parser.add_argument('-O', '--overwrite', help='overwrite without prompting.', 
                        action='store_true', default=False)
    parser.add_argument('-o', '--output', help='output filename.', 
                        type=str, default=None)
    parser.add_argument('-p', '--plot', help='Plot the rebinning results', 
                        action='store_true', default=False)

    args = parser.parse_args()

    # If file list is emtpy, use "./GRAVI*.fits"
    if len(args.fname) == 0:
        args.fname=sorted(glob.glob("./GRAVI*.fits"))

        # Filter file list
        args.fname = list(filter(filter_files, args.fname))
    
    if len(args.fname) == 0:
        raise RuntimeError("No valid file found!")
    
    # Go through each file
    for fname in args.fname:
        g = GRAVITY_data(fname)

        if args.sigma == 0:
            nsigma = None
        else:
            nsigma = args.sigma
        g.rebin_all(args.binsize, nsigma)
        g.hdul[0].header["ESO INS SPEC RES"] += "_REBINNED_"+str(args.binsize)

        # Save file
        if args.output is not None:
            out_fname = args.output
        else:
            if (len(fname) > 5) & (fname[-5:]=='.fits'):
                out_fname = fname[:-5]
            else:
                out_fname = fname
            out_fname += '_rebinned_' + str(args.binsize)
        g.save(f'{out_fname}.fits', overwrite=args.overwrite)

        if args.plot:
            from matplotlib.backends.backend_pdf import PdfPages
            
            g_o = GRAVITY_data(fname)
            pdf = PdfPages(f'{out_fname}.pdf')


            # FLUX
            axs = g_o.plot_OI_FLUX(c='C0', marker='.')
            g.plot_OI_FLUX(axs=axs, c='C3', marker='o')
            
            pdf.savefig()
            plt.close()


            # VIS
            axs = g_o.plot_OI_VIS(c='C0', marker='.')
            g.plot_OI_VIS(axs=axs, c='C3', marker='o')
            
            pdf.savefig()
            plt.close()


            # VIS2
            axs = g_o.plot_OI_VIS2(c='C0', marker='.')
            g.plot_OI_VIS2(axs=axs, c='C3', marker='o')
            
            pdf.savefig()
            plt.close()


            # T3
            axs = g_o.plot_OI_T3(c='C0', marker='.')
            g.plot_OI_T3(axs=axs, c='C3', marker='o')
            
            pdf.savefig()
            plt.close()

            pdf.close()



