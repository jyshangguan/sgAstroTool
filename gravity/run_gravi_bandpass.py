import numpy as np
from astropy.io import fits
from astropy.modeling import models, fitting
from astropy.modeling.core import Fittable1DModel
from astropy.modeling.parameters import Parameter
from astropy.modeling.physical_models import BlackBody
import astropy.units as units
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy import fft
from scipy.stats import binned_statistic
from scipy.ndimage import binary_dilation as dilation
import tqdm
import sys


baseline_name = ['G12', 'G13', 'G14', 'G23', 'G24', 'G34']
nbsl = len(baseline_name)
telescope_name = ['G1', 'G2', 'G3', 'G4']
ntel = len(telescope_name)


class Coherence_sinc(Fittable1DModel):
    '''
    The model of the coherence envelope. A top hat bandpass gives a sinc envelope.

    Parameters
    ----------
    amplitude (float): The amplitude of the coherence envelope.
    lambda_coh (float): The coherence length.
    opd0 (float): The offset of the OPD.
    absolute (bool): Whether to take the absolute value of the sinc function. Default is False.
    '''
    amplitude = Parameter(default=1, bounds=(0, None))
    lambda_coh = Parameter(default=1, bounds=(1e-16, None))
    opd0 = Parameter(default=0, bounds=(None, None))
    
    def __init__(self, amplitude=amplitude, lambda_coh=lambda_coh, opd0=opd0, absolute=False, **kwargs):
        super().__init__(amplitude=amplitude, lambda_coh=lambda_coh, opd0=opd0, **kwargs)
        
        self._absolute = absolute
    
    def evaluate(self, x, amplitude, lambda_coh, opd0):
        """
        The sinc function model.
        """
        if self._absolute:
            sinc = np.abs(np.sinc((x - opd0) / lambda_coh))  # Note: np.sinc(x) = sin(pi * x) / (pi * x) !!
        else:
            sinc = np.sinc((x - opd0) / lambda_coh)
            
        return amplitude * sinc


def quick_fit(opd, visamp, amplitude=None, lambda_coh=None, lambda_coh_exp=None, plot=True):
    '''
    Quick fit the coherence envelope.

    Parameters
    ----------
    opd (array-like): The optical path difference.
    visamp (array-like): The visibility amplitude.
    amplitude (float): The amplitude of the coherence envelope. Default is None.
    lambda_coh (float): The coherence length. Default is None.
    lambda_coh_exp (float): The expected coherence length. Default is None.

    Returns
    -------
    m_fit (astropy.modeling.Model): The fitted model.
    '''
    if amplitude is None:
        amplitude = np.max(visamp)
        
    if lambda_coh is None:
        if lambda_coh_exp is None:
            lambda_coh = 10
        else:
            lambda_coh = lambda_coh_exp
    
    m_init = Coherence_sinc(amplitude=amplitude, lambda_coh=lambda_coh, opd0=0, absolute=True)
    fitter = fitting.LevMarLSQFitter()

    m_fit = fitter(m_init, opd, visamp, maxiter=10000)
    
    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(opd, visamp)
        ax.plot(opd, m_fit(opd))
        ax.minorticks_on()
        ax.set_xlabel(r'OPD ($\mu$m)', fontsize=24)
        ax.set_ylabel(r'VISAMP', fontsize=24)
    
        txtList = []
        if lambda_coh_exp:
            ratio = m_fit.lambda_coh.value / lambda_coh_exp
            txtList.append(f'Expected lambda_coh: {lambda_coh_exp:.2e}')
            txtList.append(f'Fitted lambda_coh: {m_fit.lambda_coh.value:.2e}')
            txtList.append(f'Ratio: {ratio:.2e}')
        else:
            txtList.append(f'Fitted lambda_coh: {m_fit.lambda_coh.value:.2e}')
        txt = '\n'.join(txtList)
        ax.text(0.05, 1, txt, fontsize=16, transform=ax.transAxes, va='bottom', ha='left')
    
    return m_fit


def gen_bandpass(wave_step=0.002, wmin=1.9, wmax=2.5, wmin_bp=2.10, wmax_bp=2.11):
    '''
    Generate bandpass.

    Parameters
    ----------
    wave_step (float): The step size for generating the wave array. Default is 0.002.
    wmin (float): The minimum value of the wave array. Default is 1.9.
    wmax (float): The maximum value of the wave array. Default is 2.5.
    wmin_bp (float): The minimum value of the bandpass range. Default is 2.10.
    wmax_bp (float): The maximum value of the bandpass range. Default is 2.11.

    Returns
    -------
    bp (ndarray): The bandpass array.
    wave (ndarray): The wave array.
    wnum (ndarray): The wavenumber array.
    '''
    wave = np.arange(wmin, wmax, wave_step)
    wnum_step = wave_step / wave.mean()**2
    wnum = np.arange(1/wave.max(), 1/wave.min(), wnum_step)
    wave = 1 / wnum
    bp = np.zeros_like(wnum)
    fltr = (wave > wmin_bp) & (wave < wmax_bp)
    bp[fltr] = 1
    return bp, wave, wnum


def fft_bandpass(wnum, bp):
    '''
    FFT the bandpass.

    Parameters
    ----------
    wnum (array-like): The wavenumber array.
    bp (array-like): The bandpass array.

    Returns
    -------
    tuple: A tuple containing the wavenumber array and the FFT of the bandpass array.
    '''
    F_bp = fft.fftshift(fft.fft(fft.fftshift(bp)))
    opd = fft.fftshift(fft.fftfreq(len(wnum), np.diff(wnum)[0]))
    return opd, F_bp


def ifft_F_bandpass(opd, Fbp):
    '''
    IFFT the Fourier transformed bandpass.

    Parameters
    ----------
    opd (array-like): The optical path difference.
    Fbp (array-like): The Fourier transformed bandpass.

    Returns
    -------
    tuple: A tuple containing the wavenumber and the inverse Fourier transformed bandpass.
    '''
    bp = fft.fftshift(fft.ifft(fft.fftshift(Fbp)))
    wnum = fft.fftshift(fft.fftfreq(len(opd), np.diff(opd)[0]))
    return wnum, bp


def self_ref(datinc, ndilate=1):
    '''
    Calculate the self-calibrated visibility phase.

    Parameters
    ----------
    datinc : array
        The input incoherent data.
    ndilate : int
        The number of dilation iterations.

    Returns
    -------
    datcoh : array
        The self-referenced (coherent) data.
    '''
    nl = len(datinc)
    w = 1. - dilation(np.identity(nl), iterations=ndilate)
    model = (datinc[:, None] * w).sum(axis=0)
    model = np.conj(model) / (np.abs(model) + 1e-10)
    datcoh = datinc * model
    return datcoh


def plot_F_bandpass(opd, Fbp, fig=None, axs=None):
    '''
    Plot the fourier transformed bandpass.

    Parameters
    ----------
    opd (array-like): The optical path difference.
    Fbp (array-like): The Fourier transformed bandpass.
    fig (matplotlib.figure.Figure): The figure object. Default is None.
    axs (matplotlib.axes.Axes): The axes object. Default is None.

    Returns
    -------
    tuple: A tuple containing the figure and axes objects.
    '''
    if axs is None:
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.3, wspace=0.45)
        
    visamp = np.absolute(Fbp)
    visphi = np.angle(Fbp, deg=True)
    
    axs[0, 0].plot(opd, np.real(Fbp))
    axs[1, 0].plot(opd, np.imag(Fbp))
    axs[0, 1].plot(opd, visamp)
    axs[1, 1].plot(opd, visphi)
    
    axs[0, 0].set_ylabel(r'Real', fontsize=18)
    axs[1, 0].set_ylabel(r'Imag', fontsize=18)
    axs[0, 1].set_ylabel(r'VISAMP', fontsize=18)
    axs[1, 1].set_ylabel(r'VISPHI', fontsize=18)
    
    for ax in axs.flatten():
        ax.set_xlabel(r'OPD ($\mu$m)', fontsize=18)
        ax.minorticks_on()
    return fig, axs


def interpolate_visdata(opd, visdata, oversample=1):
    '''
    Interpolate the visibility data.

    Parameters
    ----------
    opd (array-like): The OPD array.
    visdata (array-like): The visibility data.
    oversample (int): The oversampling factor. Default is 1.

    Returns
    -------
    tuple: A tuple containing the interpolated OPD and visibility data.
    '''
    opd_step = np.abs(np.diff(opd)).mean() / oversample
    idx_min = np.argmin(opd)
    idx_max = np.argmax(opd)
    idx0 = np.min([idx_min, idx_max])
    idx1 = np.max([idx_min, idx_max])
    opd = opd[idx0:idx1]
    visdata = visdata[idx0:idx1]

    opd_shift = opd - opd.mean()
    idx_sort = np.argsort(opd_shift)
    opd_fin = opd_shift[idx_sort]
    visabs_fin = visdata[idx_sort]
    
    opd_interp = np.arange(opd_fin.min(), opd_fin.max(), opd_step)
    visamp_interp = np.interp(opd_interp, opd_fin, np.absolute(visabs_fin))
    visphi_interp = np.interp(opd_interp, opd_fin, np.exp(1j*np.angle(visabs_fin)))
    visdata_interp = visamp_interp * np.exp(1j * np.angle(visphi_interp))
    return opd_interp, visdata_interp


def convert_opd_tel2bsl(time_vis, time_opd, opd_tel):
    '''
    Convert the fiber OPD of each telescope to the baseline OPD.

    Parameters
    ----------
    time_vis (numpy.ndarray): Array of shape (N, B) representing the time visibility for each baseline.
    time_opd (numpy.ndarray): Array of shape (M, T) representing the time OPD for each telescope.
    opd_tel (numpy.ndarray): Array of shape (M, T) representing the fiber OPD for each telescope.

    Returns
    -------
    opd_bsl (numpy.ndarray): Array of shape (N, B) representing the baseline OPD.
    '''
    opd_bsl = []
    baseline_telescopes = [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]
    for bsl, (t1, t2) in enumerate(baseline_telescopes):
        topd_t1 = time_opd[:, t1]
        topd_t2 = time_opd[:, t2]
        opd_t1 = np.interp(time_vis[:, bsl], topd_t1, opd_tel[:, t1], left=np.nan, right=np.nan)
        opd_t2 = np.interp(time_vis[:, bsl], topd_t2, opd_tel[:, t2], left=np.nan, right=np.nan)
        opd_bsl.append(opd_t2 - opd_t1)
    opd_bsl = np.array(opd_bsl).T
    return opd_bsl


def find_opd_segments(time, opd, order=10, mode='clip', threshold=0.9):
    '''
    Deprecated!!!!

    Find the segments of the OPD array.

    Parameters
    ----------
    opd (array-like): The OPD array.
    order (int): The order of the extrema. Default is 10.

    Returns
    -------
    segments (array-like): The segments of the OPD array.
    '''
    fltr = opd > np.quantile(opd, threshold)
    x_max = opd[fltr]
    max_indice = argrelextrema(x_max, np.greater, order=order, mode=mode)[0]
    time_max = time[fltr][max_indice]

    fltr = opd < np.quantile(opd, 1-threshold)
    x_min = opd[fltr]
    min_indice = argrelextrema(x_min, np.less, order=order, mode=mode)[0]
    time_min = time[fltr][min_indice]
    print(time_max)
    print(time_min)

    if time_max[0] < time_min[0]:
        start = time_max
        end = time_min
    else:
        start = time_min
        end = time_max
    
    idx_start = [time.tolist().index(t) for t in start]
    idx_end = [time.tolist().index(t) for t in end]
    
    nseg = np.min([len(idx_start), len(idx_end)])

    segments = np.array([[idx_start[ii], idx_end[ii]] for ii in range(nseg)])
    return segments


def read_FT_from_p2vmred(filename, extver=20, opd_step=0.05, plot=True, 
                         fig=None, axs=None, opd_no_move=10, ref_channel=2):
    '''
    Read the time, OPD, and visibility data from the p2vmred file.

    Parameters
    ----------
    filename (str): The filename of the p2vmred file.
    plot (bool): Whether to plot the OPD and visibility data. Default is True.
    fig (matplotlib.figure.Figure): The figure object. Default is None.
    axs (matplotlib.axes.Axes): The axes object. Default is None.
    signal_baselines (list; deprecated): The list of baselines with useful signals. Default is None.
    opd_no_move (float) : The threshold indicating that the baseline do not have OPD offset. Default is 1 micron.
    chn_plot (int): The channel to plot. Default is None.

    Returns
    -------
    tuple: A tuple containing the time, OPD, and visibility data.
    '''
    d = fits.open(filename)
    tvis = d['OI_VIS', extver].data['TIME'].reshape(-1, nbsl)
    vis = d['OI_VIS', extver].data['VISDATA'].reshape(tvis.shape[0], nbsl, -1)
    gdelay = d['OI_VIS', extver].data['GDELAY'].reshape(tvis.shape[0], nbsl) * 1e6

    topd = d['OI_VIS_MET'].data['TIME'].reshape(-1, ntel)
    opd_tel = d['OI_VIS_MET'].data['OPD_FC'].reshape(-1, ntel) * 1e6
    opd = convert_opd_tel2bsl(tvis, topd, opd_tel)

    fltr = ~np.isnan(opd[:, 0])
    time = tvis[fltr, 0]
    vis = vis[fltr, :, :]
    gdelay = gdelay[fltr, :]
    opd = opd[fltr, :]
    nchn = vis.shape[-1]

    try:
        opd_bin, vis_bin = average_opd_vis(opd, vis, opd_step=opd_step, 
                                           opd_no_move=opd_no_move)
    except ValueError as e:
        print(e)
        opd_bin, vis_bin = None, None
    
    opd_move_flag = np.array([True if np.std(opd[:, bsl]) > opd_no_move else False for bsl in range(nbsl)])

    if plot:
        if axs is None:
            fig, axs = plt.subplots(nbsl, 2, figsize=(12, 12), sharex=True)

        for ax in axs[1:, 0]:
            ax.sharey(axs[0, 0])
        for ax in axs[1:, 1]:
            ax.sharey(axs[0, 1])
    
        for bsl in range(nbsl):

            if opd_move_flag[bsl]:
                color = 'C3'
            else:
                color = 'k'
            
            ax = axs[bsl, 0]
            y = opd[:, bsl]
            y = y - y.mean()
            ax.plot(time, y, color=color)
            ax.text(0.03, 0.95, baseline_name[bsl], fontsize=14, transform=ax.transAxes, va='top', ha='left')

            if bsl < nbsl - 1:
                ax.set_xticklabels([])
            
            ax = axs[bsl, 1]
            y = np.absolute(vis[:, bsl, ref_channel])
            y = y / np.abs(y).max()
            ax.plot(time, y, color=color)

            if bsl < nbsl - 1:
                ax.set_xticklabels([])
    
        for ax in axs[-1, :]:
            ax.set_xlabel(r'Time ($\mu$s)', fontsize=20)
            ax.minorticks_on()
    
        axs[0, 0].set_title(r'OPD ($\mu$m)', fontsize=20)
        axs[0, 1].set_title('Normalized VISAMP', fontsize=20)
        axs[0, 1].set_yticklabels([])

    data = dict(time=time, opd=opd, vis=vis, axs=axs, 
                opd_move_flag=opd_move_flag, 
                opd_bin=opd_bin, vis_bin=vis_bin, nchn=nchn)
    return data


def read_SC_from_astroreduced(filename, extver=11, plot=True, axs=None, 
                              opd_no_move=10, ref_channel=7):
    '''
    Read the time, OPD, and visibility data from the astroreduced file.

    Parameters
    ----------
    filename (str): The filename of the p2vmred file.
    plot (bool): Whether to plot the OPD and visibility data. Default is True.
    axs (matplotlib.axes.Axes): The axes object. Default is None.
    signal_baselines (list; deprecated): The list of baselines with useful signals. Default is None.
    opd_no_move (float) : The threshold indicating that the baseline do not have OPD offset. Default is 1 micron.
    ref_channel (int): The channel as a reference to find the zero OPD and to plot. Default is 7.

    Returns
    -------
    tuple: A tuple containing the time, OPD, and visibility data.
    '''
    d = fits.open(filename)
    time = d['OI_VIS', extver].data['TIME'].reshape(-1, nbsl)
    f1f2 = d['OI_VIS', extver].data['F1F2'].reshape(time.shape[0], nbsl, -1)

    vfac = d['OI_VIS', extver].data['V_FACTOR'].reshape(time.shape[0], nbsl, -1)
    vis = d['OI_VIS', extver].data['VISDATA'].reshape(time.shape[0], nbsl, -1) / np.sqrt(f1f2)
    vamp = np.absolute(vis) / np.sqrt(vfac)
    nchn = vis.shape[-1]

    gd_disp = d['OI_VIS', extver].data['GDELAY_DISP'].reshape(time.shape[0], nbsl) * 1e6
    gd_ft = d['OI_VIS', extver].data['GDELAY_FT'].reshape(time.shape[0], nbsl) * 1e6
    opd_disp = d['OI_VIS', extver].data['OPD_DISP'].reshape(time.shape[0], nbsl, -1) * 1e6
    opd_met_fc = d['OI_VIS'].data['OPD_met_FC'].reshape(-1, nbsl) * 1e6

    # Calculate the opd
    opdm = np.zeros(nbsl)
    for bsl in range(nbsl):
        opdm[bsl] = gd_disp[np.argmax(vamp[:, bsl, ref_channel]), bsl]
    opd = gd_ft[:, :, np.newaxis] + opd_disp - opdm[np.newaxis, :, np.newaxis]

    opd_move_flag = np.array([True if np.std(opd[:, bsl]) > opd_no_move else False for bsl in range(nbsl)])

    if plot:
        if axs is None:
            fig, axs = plt.subplots(nbsl, 2, figsize=(12, 12), sharex=True)

        for ax in axs[1:, 0]:
            ax.sharey(axs[0, 0])
        for ax in axs[1:, 1]:
            ax.sharey(axs[0, 1])
    
        for bsl in range(nbsl):

            if opd_move_flag[bsl]:
                color = 'C3'
            else:
                color = 'k'
            
            ax = axs[bsl, 0]
            y = opd[:, bsl, ref_channel]
            ax.plot(time, y, color=color)

            ax.text(0.03, 0.95, baseline_name[bsl], fontsize=14, transform=ax.transAxes, va='top', ha='left')

            if bsl < nbsl - 1:
                ax.set_xticklabels([])
            
            ax = axs[bsl, 1]
            y = np.absolute(vis[:, bsl, ref_channel])
            y = y / np.abs(y).max()
            ax.plot(time, y, color=color)

            if bsl < nbsl - 1:
                ax.set_xticklabels([])
    
        for ax in axs[-1, :]:
            ax.set_xlabel(r'Time ($\mu$s)', fontsize=20)
            ax.minorticks_on()
    
        axs[0, 0].set_title(r'OPD ($\mu$m)', fontsize=20)
        axs[0, 1].set_title('Normalized VISAMP', fontsize=20)
        axs[0, 1].set_yticklabels([])

    data = dict(time=time, opd=opd, vis=vis, gd_disp=gd_disp, vamp=vamp, 
                nchn=nchn, opdm=opdm, gd_ft=gd_ft, opd_disp=opd_disp, 
                opd_met_fc=opd_met_fc, opd_move_flag=opd_move_flag, 
                axs=axs)
    return data


def average_opd_vis(opd, vis, opd_step=0.1, opd_no_move=10, ref_channel=2):
    '''
    Average the OPD and visibility data.

    Parameters
    ----------
    opd (array-like): The OPD array.
    vis (array-like): The visibility array.
    opd_step (float): The step size of the OPD array. Default is 0.1.
    opd_no_move (float): The threshold to select the baselines that has real data.
    '''
    opd_shift = np.zeros_like(opd)
    vis_shift = np.zeros_like(vis)
    for bsl in range(nbsl):
        idx_opd = np.argsort(opd[:, bsl], axis=0)
        opd_shift[:, bsl] = opd[idx_opd, bsl]
        for chn in range(vis.shape[-1]):
            vis_shift[:, bsl, chn] = vis[idx_opd, bsl, chn]
    vamp = np.absolute(vis_shift)
        
    #opd_shift -= opd_shift.mean(axis=0, keepdims=True)
    opdm = np.zeros(nbsl)
    for bsl in range(nbsl):
        opdm[bsl] = opd_shift[np.argmax(vamp[:, bsl, ref_channel]), bsl]
    opd_shift -= opdm[np.newaxis, :]

    opd_min = np.array([opd_shift[:, bsl].min() for bsl in range(nbsl)])
    opd_max = np.array([opd_shift[:, bsl].max() for bsl in range(nbsl)])
    fltr = opd_max > opd_no_move

    if np.sum(fltr) == 0:
        raise ValueError(f'No baselines have OPD offset more than {opd_no_move} micron!')

    opd_min = np.max(opd_min[fltr])
    opd_max = np.min(opd_max[fltr])

    opd_binedges = np.arange(opd_min, opd_max, opd_step)
    opd_bin = (opd_binedges[:-1] + opd_binedges[1:]) / 2
    
    vis_bin = []
    for bsl in range(nbsl):
        vis_bin.append([])
        for chn in range(vis.shape[-1]):
            x_real = vis_shift[:, bsl, chn].real
            x_imag = vis_shift[:, bsl, chn].imag
            v_real, _, _ = binned_statistic(opd_shift[:, bsl], x_real, bins=opd_binedges, statistic='median')
            v_imag, _, _ = binned_statistic(opd_shift[:, bsl], x_imag, bins=opd_binedges, statistic='median')
            vis_bin[bsl].append(v_real + 1j * v_imag)
    vis_bin = np.swapaxes(np.array(vis_bin), 1, 2)
    vis_bin = np.swapaxes(vis_bin, 0, 1)
    opd_bin = np.ones_like(vis_bin, dtype=float) * opd_bin[:, np.newaxis, np.newaxis]

    return opd_bin, vis_bin


def gen_wavenumber(opd_min, opd_max, nsample, shift=True):
    '''
    '''
    dwn = (opd_max - opd_min) / nsample
    wavenumber = np.fft.fftfreq(nsample, d=dwn)

    if shift:
        wavenumber = np.fft.fftshift(wavenumber)

    return wavenumber


def convert_vis_to_bandpass(opd, vis, wavenumber):
    '''
    Convert the visibility to the bandpass.
    '''
    wavenum = np.zeros([len(wavenumber), vis.shape[1], vis.shape[2]], dtype=np.complex128)
    bandpass = np.zeros([len(wavenumber), vis.shape[1], vis.shape[2]], dtype=np.complex128)

    for bsl in range(nbsl):
        for chn in range(vis.shape[-1]):
            wavenum[:, bsl, chn], bandpass[:, bsl, chn] = non_uniform_fourier_transform(opd[:, bsl, chn], vis[:, bsl, chn], wavenumber)

    return wavenum, bandpass
    

def bandpass_from_vis(opd, vis, opd_step=0.1, opd_threshold=10):
    '''
    Generate the bandpass from the visibility data.

    Parameters
    ----------
    opd (array-like): The OPD array.
    vis (array-like): The visibility array.
    opd_step (float): The step size of the OPD array. Default is 0.1.

    Returns
    -------
    tuple: A tuple containing the wavelength and the bandpass.
    '''
    opd_bin, vis_bin = average_opd_vis(opd, vis, opd_step=opd_step, opd_threshold=opd_threshold)

    wave_step = 1 / (opd_bin.max() - opd_bin.min())
    wave_min, wave_max = -0.5 / opd_step, 0.5 / opd_step
    wavenumber = np.arange(wave_min, wave_max, wave_step)
    wnum, bp = convert_vis_to_bandpass(opd_bin, vis_bin, wavenumber)

    wave = 1 / wnum.mean(axis=1)
    bp_average = np.nanmean(bp, axis=1)
    return wave, bp_average


def effband_from_vis(opd, vis, eff_wave, opd_range=[-25, 25], opd_step=0.1, opd_threshold=10, plot=True):
    '''
    Measure the effective bandwidth.

    Parameters
    ----------
    opd (array-like): Array of OPD (Optical Path Difference) values.
    vis (array-like): Array of visibility values.
    eff_wave (array-like): Array of effective wavelengths.
    opd_range (list, optional): Range of OPD values to consider. Defaults to [-25, 25].

    Returns
    -------
    array-like: Array of average effective bandwidth values.
    '''
    opd_bin, vis_bin = average_opd_vis(opd, vis, opd_step=opd_step, opd_threshold=opd_threshold)
    fltr = (opd_bin > opd_range[0]) & (opd_bin < opd_range[1])
    x = opd_bin[fltr]

    dlamList = []
    for bs in range(vis_bin.shape[1]):
        dlam = []
        for cn in range(vis_bin.shape[2]):
            y = np.absolute(vis_bin[fltr, bs, cn])

            fltr_nan = ~np.isnan(y)
            if len(x[fltr_nan]) < 3:
                dlam.append(np.nan)
                continue
            
            m_fit = quick_fit(x[fltr_nan], y[fltr_nan], lambda_coh=30, lambda_coh_exp=None, plot=plot)
            dlam.append(eff_wave[cn]**2 / m_fit.lambda_coh.value)
        dlamList.append(dlam)
    dlamList = np.array(dlamList)
    
    dlam_average = np.nanmean(dlamList, axis=0)
    return dlam_average


def non_uniform_fourier_transform(x, y, frequencies):
    '''
    Compute the non-uniform Fourier transform.

    Parameters
    ----------
    x (array-like): The x array.
    y (array-like): The y array.
    frequencies (array-like): The frequencies array.

    Returns
    -------
    result (array-like): The Fourier transformed array.
    '''
    kern = y[:, np.newaxis] * np.exp(-1j * 2 * np.pi * frequencies[np.newaxis, :] * x[:, np.newaxis])
    result = np.sum(kern, axis=0)
    return frequencies, result


def Bandpass_SC(filename, extver=11, plot=True, axs=None, opd_no_move=10, 
                ref_channel=7, wave_range=None, wnum_step=None, page_size=None,
                return_complex=False):
    '''
    Read the time, OPD, and visibility data from the astroreduced file.

    Parameters
    ----------
    filename (str): The filename of the p2vmred file.
    extver (int): The extension version. Default is 11.
    plot (bool): Whether to plot the data and bandpass. Default is True.
    axs (matplotlib.axes.Axes): The axes object. Default is None.
    opd_no_move (float) : The threshold indicating that the baseline do not have OPD offset. Default is 10 micron.
    ref_channel (int): The channel as a reference to find the zero OPD and to plot. Default is 7.
    wave_range (list): The wavelength range. Default is None.
    wnum_step (float; optional): The step size of the wavenumber. 
        If not provided we calculate the step size based on the opd range.
    page_size (tuple): The size of the page. Default is None.

    Returns
    -------
    tuple: A tuple containing the time, OPD, and visibility data.
    '''
    if plot:
        if axs is None:
            fig, axs = plt.subplots(nbsl, 3, figsize=(12, 12))
            fig.subplots_adjust(hspace=0.02, wspace=0.02)
            axo = fig.add_subplot(111, frameon=False) # The out axis
            axo.tick_params(axis='y', which='both', left=False, labelleft=False)
            axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            fn = filename.split('/')[-1]
            axo.set_title(fn, fontsize=14, pad=30, color='gray')

            if page_size is not None:
                fig.set_size_inches(page_size, forward=True)

        axs_data = axs[:, 0:2]
        axs_bp = axs[:, 2]
        for ax in axs_bp[1:]:
            ax.sharex(axs_bp[0])
            ax.sharey(axs_bp[0])
    else:
        axs_data = None
        axs_bp = None

    data = read_SC_from_astroreduced(filename, extver=extver, plot=plot, axs=axs_data, 
                                     opd_no_move=opd_no_move, ref_channel=ref_channel)

    vis = data['vis']
    opd = data['opd']
    opd_move_flag = data['opd_move_flag']
    nchn = data['nchn']

    if wave_range is None:
        wave_range = [1.9, 2.6]

    if wnum_step is None:
        wnum_step = 1 / (opd.max() - opd.min())

    wavenumber = np.arange(1/wave_range[0], 1/wave_range[1], -wnum_step)
    wavenum, bp = convert_vis_to_bandpass(opd, vis, wavenumber)

    for bsl in range(nbsl):
        if not opd_move_flag[bsl]:
            bp[:, bsl, :] = np.nan
        else:
            for chn in range(nchn):
                bp_ref = self_ref(bp[:, bsl, chn])
                bp[:, bsl, chn] = bp_ref / np.max(np.absolute(bp_ref))

    if not return_complex:
        bp = np.absolute(bp)
    wave = 1 / wavenumber
        
    if plot:
        axs_bp[0].set_title(f'Bandpass P{extver}', fontsize=20)
        colors = plt.cm.coolwarm(np.linspace(0, 1, nchn))

        for ii, ax in enumerate(axs_bp):
            for nc in range(nchn):
                b = bp[:, ii, nc]
                if return_complex:
                    ax.plot(wave, b.real, color=colors[nc], label=f'ch{nc}', ls='-')
                    ax.plot(wave, b.imag, color=colors[nc], ls='--')
                    ax.set_ylim(-1.1, 1.1)
                else:
                    ax.plot(wave, b, color=colors[nc], label=f'ch{nc}', ls='-')
                    ax.set_ylim(-0.1, 1.1)

        ax.minorticks_on()
        ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=20)
        ax.set_yticklabels([])
        
    return wave, bp


def Bandpass_FT(filename, extver=11, plot=True, axs=None, opd_no_move=10, 
                ref_channel=2, wave_range=None, wnum_step=None, page_size=None, 
                return_complex=False):
    '''
    Read the time, OPD, and visibility data from the p2vmred file.

    Parameters
    ----------
    filename (str): The filename of the p2vmred file.
    extver (int): The extension version. Default is 11.
    plot (bool): Whether to plot the data and bandpass. Default is True.
    axs (matplotlib.axes.Axes): The axes object. Default is None.
    opd_no_move (float) : The threshold indicating that the baseline do not have OPD offset. Default is 10 micron.
    ref_channel (int): The channel as a reference to find the zero OPD and to plot. Default is 7.
    wave_range (list): The wavelength range. Default is None.
    wnum_step (float; optional): The step size of the wavenumber. 
        If not provided we calculate the step size based on the opd range.
    page_size (tuple): The size of the page. Default is None.

    Returns
    -------
    tuple: A tuple containing the time, OPD, and visibility data.
    '''
    if plot:
        if axs is None:
            fig, axs = plt.subplots(nbsl, 3, figsize=(12, 12))
            fig.subplots_adjust(hspace=0.02, wspace=0.02)

            if page_size is not None:
                fig.set_size_inches(page_size, forward=True)

        axs_data = axs[:, 0:2]
        axs_bp = axs[:, 2]
        for ax in axs_bp[1:]:
            ax.sharex(axs_bp[0])
            ax.sharey(axs_bp[0])
    else:
        axs_data = None
        axs_bp = None

    data = read_FT_from_p2vmred(filename, extver=extver, plot=plot, axs=axs_data, 
                                opd_no_move=opd_no_move, ref_channel=ref_channel)

    vis = data['vis_bin']
    opd = data['opd_bin']
    opd_move_flag = data['opd_move_flag']
    nchn = data['nchn']

    # If there is no useful data
    if vis is None:
        ax = axs_bp[0]
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        return None, None

    # Otherwise, calculate the bandpass
    if wave_range is None:
        wave_range = [1.9, 2.6]

    if wnum_step is None:
        wnum_step = 1 / (opd.max() - opd.min())

    wavenumber = np.arange(1/wave_range[0], 1/wave_range[1], -wnum_step)
    wavenum, bp = convert_vis_to_bandpass(opd, vis, wavenumber)

    for bsl in range(nbsl):
        if not opd_move_flag[bsl]:
            bp[:, bsl, :] = np.nan
        else:
            for chn in range(nchn):
                bp_ref = self_ref(bp[:, bsl, chn])
                bp[:, bsl, chn] = bp_ref / np.max(np.absolute(bp_ref))

    if not return_complex:
        bp = np.absolute(bp)
    wave = 1 / wavenumber
        
    if plot:
        axs_bp[0].set_title(f'Bandpass P{extver}', fontsize=20)
        colors = plt.cm.coolwarm(np.linspace(0, 1, nchn))

        for ii, ax in enumerate(axs_bp):
            for nc in range(nchn):
                b = bp[:, ii, nc]
                #ax.plot(wave, b, label=f'ch{nc}', marker='.')
                if return_complex:
                    ax.plot(wave, b.real, color=colors[nc], label=f'ch{nc}', ls='-')
                    ax.plot(wave, b.imag, color=colors[nc], ls='--')
                    ax.set_ylim(-1.1, 1.1)
                else:
                    ax.plot(wave, b, color=colors[nc], label=f'ch{nc}', ls='-')
                    ax.set_ylim(-0.1, 1.1)
        ax.minorticks_on()
        ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=20)
        ax.set_yticklabels([])
        
    return wave, bp


def correct_lampspec(wave, bp, temperature):
    '''
    Correct the bandpass for the lamp spectrum.

    Parameters
    ----------
    wave (array-like): The wavelength array.
    bp (array-like): The bandpass array. It should have [nwave, nbsl, nchn].
    temperature (float): The temperature of the lamp.

    Returns
    -------
    array-like: The corrected bandpass.
    '''
    bb = BlackBody(temperature * units.K)
    bbspec = bb(wave * units.micron)
    bbspec = bbspec / bbspec.max()

    if len(bp.shape) == 3:
        bp_corr = bp / bbspec[:, np.newaxis, np.newaxis]
    elif len(bp.shape) == 2:
        bp_corr = bp / bbspec[:, np.newaxis]
    else:
        raise ValueError(f'The dimension of bp ({bp.shape}) is not compatible!')
    bp_corr /= bp_corr.max(axis=0, keepdims=True)
    return bp_corr


def fit_bandpass_gauss(wave, bp, window=0.2, stddev=0.05, plot=False, ax=None, 
                       data_kwargs=None, model_kwargs=None):
    '''
    Fit a Gaussian model to the bandpass and return the FWHM.

    Parameters
    ----------
    wave : array
        Wavelength array
    bp : array
        Bandpass array. The data should be [wave, channel].
    window : float
        Wavelength window to fit the Gaussian model.
    stddev : float
        Initial guess for the standard deviation of the Gaussian model.
    plot : bool
        If True, plot the bandpass and the model.
    ax : matplotlib axis
        Axis to plot the bandpass and the model.
    data_kwargs : dict
        Keyword arguments for the bandpass plot.
    model_kwargs : dict
        Keyword arguments for the model plot.
    
    Returns
    -------
    gList : list
        List of Gaussian models for each channel.
    tb : astropy.table.Table
        Table of the wavelength and FWHM for each channel.
    '''
    g_init = models.Gaussian1D(amplitude=1, stddev=stddev)
    fitter = fitting.LevMarLSQFitter()

    wList = []
    fwhm = []
    gList = []
    for nc in range(bp.shape[-1]):
        wcen = wave[np.argmax(bp[:, nc])]
        wmin = wcen - window / 2
        wmax = wcen + window / 2
        fltr = (wave > wmin) & (wave < wmax)
        x_fit = wave[fltr]
        y_fit = bp[fltr, nc]

        g_init.mean = wcen
        gList.append(fitter(g_init, x_fit, y_fit))
        fwhm.append(gList[-1].fwhm)
        wList.append(np.average(wave, weights=bp[:, nc]))

    chn = [f'CH{ii}' for ii in range(bp.shape[-1])]
    tb = Table([chn, wList, fwhm], names=('Channel', 'Wavelength', 'FWHM'))
    tb['Wavelength'].format = '%.3f'
    tb['FWHM'].format = '%.3f'

    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))

        if data_kwargs is None:
            data_kwargs = dict(marker='.', lw=2)
        if model_kwargs is None:
            model_kwargs = dict(color='C3', lw=1)

        for nc in range(bp.shape[-1]):
            ax.plot(wave, bp[:, nc], **data_kwargs)
            ax.plot(wave, gList[nc](wave), **model_kwargs)
        ax.minorticks_on()
    return gList, tb


if __name__ == '__main__':
    import glob
    import argparse
    from matplotlib.backends.backend_pdf import PdfPages

    parser = argparse.ArgumentParser(
        description='The script take the astroreduced files or p2vmred files to generate the bandpass.')
    parser.add_argument('fits_files', nargs='*', default=None, 
                        help='List of FITS files')
    parser.add_argument('-f', '--fiber_type', default='SC', 
                        help='The fiber type. Default is SC')
    parser.add_argument('-p', '--plot', action='store_true', default=False,
                        help='Whether to plot the bandpass. Default is False')
    parser.add_argument('--wmin', default=1.9, 
                        help='The minimum wavelength. Default is 1.9 micron')
    parser.add_argument('--wmax', default=2.6, 
                        help='The maximum wavelength. Default is 2.6 micron')
    parser.add_argument('--wnum_step', default=0.001, 
                        help='The step size of the wavenumber. Default is 0.001 micron^-1')
    parser.add_argument('-c', '--color_correction', action='store_true', default=False,
                        help='Whether to correct the lamp spectrum. Default is False')
    parser.add_argument('-t', '--temperature', default=800,
                        help='The blackbody temperature of the lamp. Default is 800 K')
    parser.add_argument('-a', '--average_type', default='COMPLEX', 
                        help='Averaging the complex bandpass if [COMPLEX] is used, while averaging the absolute bandpass if [ABSOLUTE] is used.')
    parser.add_argument('-d', '--debug', action='store_true', default=False, 
                        help='Turn on the debugging mode.')

    args = parser.parse_args()

    fits_file_list = args.fits_files
    fiber_type = args.fiber_type
    plot=args.plot
    wave_range = [args.wmin, args.wmax]
    wnum_step = float(args.wnum_step)
    color_correction = args.color_correction
    temperature = float(args.temperature)
    average_type = args.average_type
    debug = args.debug
    page_size = (10, 12)  # A4 size

    if fiber_type not in ['SC', 'FT']:
        raise ValueError(f'The fiber type ({fiber_type}) is not recognized.')

    if len(fits_file_list) == 0:
        if fiber_type == 'SC':
            fits_file_list = glob.glob('*astroreduced.fits')
        else:
            fits_file_list = glob.glob('*p2vmred.fits')
    fits_file_list.sort()

    
    print(f'Fiber type: {fiber_type}')
    print(f'Plot: {plot}')
    print(f'Wavelength range: {wave_range}')
    print(f'Wavenumber step: {wnum_step} micron^-1')
    print(f'Averaging type: {average_type}')
    print(f'Debugging mode: {debug}')
    print(f'Color correction: {color_correction}')
    if color_correction:
        print(f'Temperature: {temperature} K')

    if average_type == 'COMPLEX':
        use_complex = True
    elif average_type == 'ABSOLUTE':
        use_complex = False
    else:
        raise ValueError(f'Cannot recognize the average_type ({average_type})!')

    if len(fits_file_list) == 0:
        print('No FITS files found.')
        sys.exit()

    print(f'Found {len(fits_file_list)} files:')
    for loop, f in enumerate(fits_file_list):
        print(f'  {f}')
        if loop == 0:
            header = fits.getheader(f, ext=0)
            pol = header['ESO INS POLA MODE']
            res = header['ESO INS SPEC RES']
        else:
            header = fits.getheader(f, ext=0)
            if pol != header['ESO INS POLA MODE']:
                raise ValueError('The polarization mode is different!')
            if res != header['ESO INS SPEC RES']:
                raise ValueError('The spectral resolution is different!')

    if plot:
        pdf = PdfPages(f'bandpass_{fiber_type}.pdf')

    if pol == 'SPLIT':
        bp1List = []
        bp2List = []
        for loop, f in tqdm.tqdm(enumerate(fits_file_list)):
            fn = f.split('/')[-1][:-5]

            if fiber_type == 'SC':
                wave, bp1 = Bandpass_SC(
                    f, extver=11, wave_range=wave_range, wnum_step=wnum_step, 
                    return_complex=use_complex, plot=plot, page_size=page_size)
                if plot:
                    pdf.savefig(bbox_inches='tight')
                    plt.close()

                _, bp2 = Bandpass_SC(
                    f, extver=12, wave_range=wave_range, wnum_step=wnum_step, 
                    return_complex=use_complex, plot=plot, page_size=page_size)
                if plot:
                    pdf.savefig(bbox_inches='tight')
                    plt.close()

            else:
                wave, bp1 = Bandpass_FT(
                    f, extver=21, wave_range=wave_range, wnum_step=wnum_step, 
                    return_complex=use_complex, plot=plot, page_size=page_size, )
                if plot:
                    pdf.savefig(bbox_inches='tight')
                    plt.close()

                _, bp2 = Bandpass_FT(
                    f, extver=22, wave_range=wave_range, wnum_step=wnum_step, 
                    return_complex=use_complex, plot=plot, page_size=page_size)
                if plot:
                    pdf.savefig(bbox_inches='tight')
                    plt.close()

            bp1List.append(bp1)
            bp2List.append(bp2)
        bp1List = np.array(bp1List)
        bp2List = np.array(bp2List)

        # Calculate the average bandpass
        bp_ave_all = np.absolute(np.nanmean(np.concatenate([bp1List, bp2List]), axis=(0, 2)))
        
        if color_correction:
            bp_ave_all = correct_lampspec(wave, bp_ave_all, temperature)
        
        bp_ave_all /= np.max(bp_ave_all, axis=0, keepdims=True)
        
        if debug:
            bp_ave_bsl_1 = np.absolute(np.nanmean(bp1List, axis=0))
            bp_ave_bsl_2 = np.absolute(np.nanmean(bp2List, axis=0))
            bp_ave_all_1 = np.absolute(np.nanmean(bp1List, axis=(0, 2)))
            bp_ave_all_2 = np.absolute(np.nanmean(bp2List, axis=(0, 2)))

            if color_correction:
                # Correct the lamp spectrum
                bp_ave_bsl_1 = correct_lampspec(wave, bp_ave_bsl_1, temperature)
                bp_ave_bsl_2 = correct_lampspec(wave, bp_ave_bsl_2, temperature)
                bp_ave_all_1 = correct_lampspec(wave, bp_ave_all_1, temperature)
                bp_ave_all_2 = correct_lampspec(wave, bp_ave_all_2, temperature)

            bp_ave_bsl_1 /= np.max(bp_ave_bsl_1, axis=0, keepdims=True)
            bp_ave_bsl_2 /= np.max(bp_ave_bsl_2, axis=0, keepdims=True)
            bp_ave_all_1 /= np.max(bp_ave_all_1, axis=0, keepdims=True)
            bp_ave_all_2 /= np.max(bp_ave_all_2, axis=0, keepdims=True)

    elif pol == 'COMBINED':
        bpList = []
        for loop, f in tqdm.tqdm(enumerate(fits_file_list)):
            fn = f.split('/')[-1][:-5]

            if fiber_type == 'SC':
                wave, bp = Bandpass_SC(
                    f, extver=10, wave_range=wave_range, wnum_step=wnum_step, 
                    return_complex=use_complex, plot=plot, page_size=page_size)
                if plot:
                    pdf.savefig(bbox_inches='tight')
                    plt.close()

            else:
                wave, bp = Bandpass_FT(
                    f, extver=20, wave_range=wave_range, wnum_step=wnum_step, 
                    return_complex=use_complex, plot=plot, page_size=page_size)
                if plot:
                    pdf.savefig(bbox_inches='tight')
                    plt.close()

            bpList.append(bp)
        bpList = np.array(bpList)
    
        # Calculate the average bandpass
        bp_ave_all = np.absolute(np.nanmean(bpList, axis=(0, 2)))

        if color_correction:
            # Correct the lamp spectrum
            bp_ave_all = correct_lampspec(wave, bp_ave_all, temperature)
        
        bp_ave_all /= np.max(bp_ave_all, axis=0, keepdims=True)

        if debug:
            bp_ave_bsl = np.absolute(np.nanmean(bpList, axis=0))

            if color_correction:
                # Correct the lamp spectrum
                bp_ave_bsl = correct_lampspec(wave, bp_ave_bsl, temperature)
            
            bp_ave_bsl /= np.max(bp_ave_bsl, axis=0, keepdims=True)
    else:
        raise ValueError(f'Cannot recognize the polarization ({pol})!')

    # Save the output
    hdr = fits.Header()
    hdr['ESO INS POLA MODE'] = pol
    hdr['ESO INS SPEC RES'] = res
    hdr['CHANNEL'] = (f'{fiber_type}', 'SC or FT Channel')
    hdr['TYPAVG'] = (f'{average_type}', 'Average type, COMPLEX or ABSOLUTE')
    hdr['CLCORR'] = (f'{color_correction}', 'If applied color correction')
    hdr['TEMPBB'] = (f'{temperature}', 'Lamp blackbody temperature (K)')

    hduList = [fits.PrimaryHDU([1], header=hdr),
               fits.BinTableHDU.from_columns(
                   [fits.Column(name='WAVELENGTH', format='D', array=wave),
                    fits.Column(name='EFF_TRANS', format=f'{bp_ave_all.shape[-1]}D', array=bp_ave_all)])]

    hduList[1].header['EXTNAME'] = 'OI_BANDPATH'
    hduList[1].header['TTYPE1'] = ('WAVELENGTH', 'Wavelength of effective transmission curves')
    hduList[1].header['TTYPE2'] = ('EFF_TRANS', 'Averaged effective transmission over all baselines and polarization')

    if debug:
        hduList[0].header['DEBUG'] = (True, 'Debugging mode')

        if pol == 'SPLIT':
            hduList += [fits.BinTableHDU.from_columns(
                           [fits.Column(name='EFF_TRANS_1', format=f'{bp_ave_all_1.shape[-1]}D', array=bp_ave_all_1),
                            fits.Column(name='EFF_TRANS_2', format=f'{bp_ave_all_2.shape[-1]}D', array=bp_ave_all_2)]),
                       fits.ImageHDU(name='EFF_TRANS_1', data=bp_ave_bsl_1),
                       fits.ImageHDU(name='EFF_TRANS_2', data=bp_ave_bsl_2)]
            hduList[2].header['EXTNAME'] = 'BP_POL'
            hduList[2].header['TTYPE1'] = ('BP_POL_1', 'Averaged bandpass in polarization 1')
            hduList[2].header['TTYPE2'] = ('BP_POL_2', 'Averaged bandpass in polarization 2')
            hduList[3].header['EXTNAME'] = ('BP_POL_BSL_1', 'Bandpass per baseline in polarization 1')
            hduList[4].header['EXTNAME'] = ('BP_POL_BSL_2', 'Bandpass per baseline in polarization 2')

        else:
            hduList.append(fits.ImageHDU(name='BP_BSL', data=bp_ave_bsl))
            hduList[2].header['EXTNAME'] = ('BP_BSL', 'Bandpass per baseline')

    hduList[0].header['HISTORY'] = f'Reduced the following {len(fits_file_list)} files'
    for f in fits_file_list:
        hduList[0].header['HISTORY'] = f

    hdul = fits.HDUList(hduList)
    hdul.writeto(f'bandpass_{fiber_type}.fits', overwrite=True)

    if plot:
        if pol == 'SPLIT':
            colors = plt.cm.coolwarm(np.linspace(0, 1, bp1List.shape[-1]))

            # Plot the complex bandpass for each channel
            for chn in range(bp1List.shape[-1]):
                fig, axs = plt.subplots(6, 2, figsize=page_size, sharex=True, sharey=True)
                fig.subplots_adjust(hspace=0, wspace=0.05)
                axo = fig.add_subplot(111, frameon=False) # The out axis
                axo.tick_params(axis='y', which='both', left=False, labelleft=False)
                axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                axo.set_xlabel(r"Wavelength ($\mu$m)", fontsize=24, labelpad=20)
                axo.set_ylabel(fr"Complex bandpass", fontsize=24, labelpad=32) #
                axo.set_title(f'Complex bandpass of channel {chn}', fontsize=14, pad=30, color='gray')

                for bsl, (ax1, ax2) in enumerate(axs):
                    for ii in range(bp1List.shape[0]):
                        if ii == 0:
                            l1 = 'Real'
                            l2 = 'Imag'
                        else:
                            l1 = None
                            l2 = None

                        ax1.plot(wave, bp1List[ii, :, bsl, chn].real, c='C0', lw=0.5, label=l1)
                        ax1.plot(wave, bp1List[ii, :, bsl, chn].imag, c='C3', lw=0.5, label=l2)
                        ax2.plot(wave, bp2List[ii, :, bsl, chn].real, c='C0', lw=0.5)
                        ax2.plot(wave, bp2List[ii, :, bsl, chn].imag, c='C3', lw=0.5)
                    ax1.text(0.01, 0.95, baseline_name[bsl], transform=ax1.transAxes, 
                             fontsize=16, ha='left', va='top')
                axs[0, 0].set_title('P11', fontsize=14)
                axs[0, 1].set_title('P12', fontsize=14)
                axs[0, 0].legend(loc='lower right', fontsize=12, ncols=2)
                ax1.set_ylim([-1.1, 1.1])
                ax1.minorticks_on()
                ax2.minorticks_on()

                pdf.savefig(bbox_inches='tight')
                plt.close()


            if debug:
                # Plot the averaged bandpass for each baseline
                fig, axs = plt.subplots(6, 2, figsize=page_size, sharex=True, sharey=True)
                fig.subplots_adjust(hspace=0, wspace=0.05)
                axo = fig.add_subplot(111, frameon=False) # The out axis
                axo.tick_params(axis='y', which='both', left=False, labelleft=False)
                axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                axo.set_xlabel(r"Wavelength ($\mu$m)", fontsize=24, labelpad=20)
                axo.set_title(r'Averaged bandpass per baseline', fontsize=14, pad=30, color='gray')
            
                if color_correction:
                    axo.set_ylabel(fr"Bandpass (T={temperature} $K$)", fontsize=24, labelpad=32) #
                else:
                    axo.set_ylabel(fr"Bandpass (no correction)", fontsize=24, labelpad=32) #
        
                for ii in range(6):
                    for nc in range(bp_ave_bsl_1.shape[-1]):
                        axs[ii, 0].plot(wave, bp_ave_bsl_1[:, ii, nc], color=colors[nc], label=f'ch{nc}')
                        axs[ii, 1].plot(wave, bp_ave_bsl_2[:, ii, nc], color=colors[nc], label=f'ch{nc}')
                    axs[ii, 0].text(0.01, 0.95, baseline_name[ii], transform=axs[ii, 0].transAxes, 
                            fontsize=16, ha='left', va='top')
                axs[0, 0].set_title(r'P11', fontsize=20)
                axs[0, 1].set_title(r'P12', fontsize=20)
                axs[0, 0].minorticks_on()
                axs[0, 0].set_ylim(-0.1, 1.1)

                pdf.savefig(bbox_inches='tight')
                plt.close()


                # Plot the averaged bandpass for all baselines
                fig, axs = plt.subplots(2, 1, figsize=page_size, sharex=True, sharey=True)
                fig.subplots_adjust(hspace=0)
                axo = fig.add_subplot(111, frameon=False) # The out axis
                axo.tick_params(axis='y', which='both', left=False, labelleft=False)
                axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                axo.set_xlabel(r"Wavelength ($\mu$m)", fontsize=24, labelpad=20)

                if color_correction:
                    axo.set_title(fr'Averaged bandpass (T={temperature} $K$)', fontsize=14, pad=30, color='gray')
                else:
                    axo.set_title(r'Averaged bandpass (no correction)', fontsize=14, pad=30, color='gray')
    
                ax = axs[0]
                for nc in range(bp_ave_all_1.shape[-1]):
                    b = bp_ave_all_1[:, nc]
                    ax.plot(wave, b, color=colors[nc], label=f'ch{nc}')
                ax.axhline(y=0, ls='--', lw=0.5, color='k', zorder=4)
                ax.set_ylabel(r"P11", fontsize=24) #
            
                ax = axs[1]
                for nc in range(bp_ave_all_2.shape[-1]):
                    b = bp_ave_all_2[:, nc]
                    ax.plot(wave, b, color=colors[nc], label=f'ch{nc}')
                ax.set_ylabel(fr"P12", fontsize=24) #
                ax.axhline(y=0, ls='--', lw=0.5, color='k', zorder=4)
    
                ax.minorticks_on()
                pdf.savefig(bbox_inches='tight')
                plt.close()
                
            # Plot the averaged bandpass for all baselines
            fig, ax = plt.subplots(figsize=page_size, sharex=True, sharey=True)

            if color_correction:
                axo.set_title(fr'Averaged bandpass (T={temperature} $K$)', fontsize=14, pad=30, color='gray')
            else:
                axo.set_title(r'Averaged bandpass (no correction)', fontsize=14, pad=30, color='gray')
    
            for nc in range(bp_ave_all.shape[-1]):
                b = bp_ave_all[:, nc]
                ax.plot(wave, b, color=colors[nc], label=f'ch{nc}')
            ax.axhline(y=0, ls='--', lw=0.5, color='k', zorder=4)
            ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=24, labelpad=20)
            ax.set_ylabel(r"Bandpass", fontsize=24) #
    
            ax.minorticks_on()
            pdf.savefig(bbox_inches='tight')
            plt.close()
    
        else:
            colors = plt.cm.coolwarm(np.linspace(0, 1, bpList.shape[-1]))

            # Plot the complex bandpass for each channel
            for chn in range(bpList.shape[-1]):
                fig, axs = plt.subplots(6, 1, figsize=page_size, sharex=True, sharey=True)
                fig.subplots_adjust(hspace=0, wspace=0.05)
                axo = fig.add_subplot(111, frameon=False) # The out axis
                axo.tick_params(axis='y', which='both', left=False, labelleft=False)
                axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                axo.set_xlabel(r"Wavelength ($\mu$m)", fontsize=24, labelpad=20)
                axo.set_ylabel(fr"Complex bandpass", fontsize=24, labelpad=32) #
                axo.set_title(f'Complex bandpass of channel {chn}', fontsize=14, pad=30, color='gray')

                for bsl, ax in enumerate(axs):
                    for ii in range(bpList.shape[0]):
                        if ii == 0:
                            l1 = 'Real'
                            l2 = 'Imag'
                        else:
                            l1 = None
                            l2 = None

                        ax.plot(wave, bpList[ii, :, bsl, chn].real, c='C0', lw=0.5, label=l1)
                        ax.plot(wave, bpList[ii, :, bsl, chn].imag, c='C3', lw=0.5, label=l2)
                    ax.text(0.01, 0.95, baseline_name[bsl], transform=ax.transAxes, 
                             fontsize=16, ha='left', va='top')
                axs[0].legend(loc='lower right', fontsize=12, ncols=2)
                ax.set_ylim([-1.1, 1.1])
                ax.minorticks_on()

                pdf.savefig(bbox_inches='tight')
                plt.close()

            if debug:
                # Plot the averaged bandpass for each baseline
                fig, axs = plt.subplots(6, 1, figsize=page_size, sharex=True, sharey=True)
                fig.subplots_adjust(hspace=0)
                axo = fig.add_subplot(111, frameon=False) # The out axis
                axo.tick_params(axis='y', which='both', left=False, labelleft=False)
                axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                axo.set_xlabel(r"Wavelength ($\mu$m)", fontsize=24, labelpad=20)
                axo.set_ylabel(r"Measured bandpass", fontsize=24, labelpad=32) #
                axo.set_title(r'Averaged bandpass per baseline', fontsize=14, pad=30, color='gray')
        
                for ii, ax in enumerate(axs):
                    for nc in range(bp_ave_bsl.shape[-1]):
                        b = bp_ave_bsl[:, ii, nc]
                        ax.plot(wave, b, color=colors[nc], label=f'ch{nc}')
                    ax.text(0.01, 0.95, baseline_name[ii], transform=ax.transAxes, 
                            fontsize=16, ha='left', va='top')
                ax.minorticks_on()
                ax.set_ylim(-0.1, 1.1)

                pdf.savefig(bbox_inches='tight')
                plt.close()

            # Plot the averaged bandpass for all baselines
            fig, ax = plt.subplots(figsize=page_size, sharex=True, sharey=True)
            ax.set_xlabel(r"Wavelength ($\mu$m)", fontsize=24, labelpad=20)
            
            if color_correction:
                ax.set_title(fr'Averaged bandpass (T={temperature} $K$)', fontsize=14, pad=30, color='gray')
            else:
                ax.set_title(r'Averaged bandpass (no correction)', fontsize=14, pad=30, color='gray')
    
            for nc in range(bp_ave_all.shape[-1]):
                b = bp_ave_all[:, nc]
                ax.plot(wave, b, color=colors[nc], label=f'ch{nc}')
            ax.axhline(y=0, ls='--', lw=0.5, color='k', zorder=4)
            ax.set_ylabel(r"Measured bandpass", fontsize=24) #
            
            ax.minorticks_on()
            pdf.savefig(bbox_inches='tight')
            plt.close(fig)
    
        pdf.close()
    