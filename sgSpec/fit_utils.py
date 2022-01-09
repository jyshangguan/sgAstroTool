import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.modeling import fitting

__all__ = ['decomposer', 'plot_fit', 'multidecomposer', 'spec_perturb',
           'decomopse_Hbeta', 'integrate_Hbeta', 'plot_ref_lines', 
           'mask_spectrum']


def multidecomposer(m_init, wavelist, fluxlist, weightlist, pool=None, maxiter=10000):
    '''
    Decompose the AGN optical spectra.
    '''
    argList = []
    for loop, (wave, flux, weight) in enumerate(zip(wavelist, fluxlist, weightlist)):
        d = dict(model=m_init.copy(), wave=wave, flux=flux, weights=weight,
                 maxiter=maxiter, stamp=loop)
        argList.append(d)

    if pool is None:
        result = [decomposer(arg) for arg in tqdm(argList)]
    else:
        result = list(tqdm(pool.imap(decomposer, argList), total=len(argList)))

    return result


def decomposer(d):
    '''
    Decompose the AGN optical spectrum.
    '''
    m_init = d['model']
    wave = d['wave']
    flux = d['flux']
    weights = d.get('weights', None)
    maxiter = d.get('maxiter', 10000)
    stamp = d.get('stamp', None)
    print_model = d.get('print_model', False)

    fitter = fitting.LevMarLSQFitter()
    m_fit = fitter(m_init, wave, flux, weights=weights, maxiter=maxiter)

    if weights is None:
        weights = np.ones_likes(wave)
        
    chisq = np.sum((flux - m_fit(wave))**2 * weights**2)
    dof = len(flux) - len(m_fit.parameters)

    if print_model:
        for m in m_fit:
            print(m.__repr__())
        print('Chisq: {0:.2f}'.format(chisq))
        print('dof: {0}'.format(dof))

    res = {
        'data': {
            'wave': wave,
            'flux': flux,
            'weights': weights
        },
        'model': m_fit,
        'chisq': chisq,
        'dof': dof,
        'stamp': stamp
    }
    return res


def plot_fit(res, axs=None, remove_masked=False, ymin=-1, ymax=1):
    '''
    Plot the fitting results.
    '''

    if axs is None:
        fig = plt.figure(figsize=(15, 6))
        ax = fig.add_axes([0.05, 0.25, 0.98, 0.73])
        axr = fig.add_axes([0.05, 0.05, 0.98, 0.2])
    else:
        ax, axr = axs

    wave = res['data']['wave']
    flux = res['data']['flux']
    weights = res['data']['weights']
    m_fit = res['model']

    # Plot data
    ax.step(wave, flux, color='k', label='Data')

    # Plot model total
    x = wave
    y = m_fit(x)
    ax.plot(x, y, label='Total')

    # Plot model components
    if m_fit.n_submodels > 1:
        if 'Extinction' in m_fit.submodel_names:
            ext = m_fit['Extinction'](wave)
            ignList = ['Extinction']
        else:
            ext = 1
            ignList = []

        for m in m_fit:
            if m.name in ignList:
                continue
            x = wave
            y = m(x) * ext
            ax.plot(x, y, label=m.name)

    ax.legend(loc='upper left', fontsize=16, ncol=3)
    ax.set_ylabel(r'$F_\lambda\,(\times 10^{-15}\,\mathrm{erg\,s^{-1}\,cm^{-2}\,\AA^{-1}})$', fontsize=20)
    ax.minorticks_on()


    # Plot weights
    if weights is not None:
        axt = ax.twinx()
        axt.step(wave, weights, color='k', ls=':')
        axt.set_ylabel('Weights', fontsize=20)
        axt.minorticks_on()


    # Plot residual
    res = flux - m_fit(x)
    if remove_masked:
        res[weights==0] = 0

    axr.step(wave, res, color='k')

    axr.axhline(y=0., ls='--', color='gray')
    axr.axhspan(ymin=ymin, ymax=ymax, ls='--', facecolor='none', edgecolor='gray')
    axr.minorticks_on()
    axr.set_xlabel(r'Wavelength ($\mathrm{\AA}$)', fontsize=20)
    axr.set_ylabel(r'Res.', fontsize=24)

    return ax, axr


def spec_perturb(flux, ferr):
    '''
    Perturb the spectral flux.
    '''
    f = flux + ferr * np.random.randn(len(flux))
    return f


def plot_ref_lines(ax, alpha=0.6, ymin=0, ymax=1, ls='--', fontsize=12, 
                   ha='right', va='top', text_y=None, lineDict=None):
    '''
    Plot the typical emission lines of AGNs as a reference.  The input figure 
    should be in the restframe of the target.
    
    Parameters
    ----------
    ax : Axes 
        The axis of the figure to add the reference lines.
    alpha : float (default: 0.6)
        The opacity of the lines.
    ymin : float (default: 0)
        The min of the vertical line.  Range [0, 1].
    ymax : float (default: 1)
        The max of the vertical line.  Range [0, 1].
    ls : string (default: '--')
        The linestyle of the vertical line.
    fontsize : float (default: 12)
        The fontsize of the line label.
    ha : string (default: 'right')
        The horizontal alignment of the line label.
    va : string (default: 'top')
        The vertical alignment of the line label.
    text_y (optional) : float
        The vertical position of the line label.
    lineDict (optional) : dict 
        The information of the line. The key is the line label (name). 
        The content main contain:
            x : wavelength (units: Angstrom).
            x_off (optional) : horizontal offset of the label, a fraction of the wavelength.
            y_off (optional) : vertical offset of the label, a fraction of 'text_y'.
    '''
    if lineDict is None:
        lineDict = { # These are wavelength in the air (NIST)
            r'[Ne V] 3426':   dict(x=3425.881),
            r'[O II] 3727':   dict(x=3727),
            r'[Fe VII] 3759': dict(x=3758.920, y_off=-0.2),
            r'[Ne III] 3869': dict(x=3868.760),
            r'He I 3889':     dict(x=3888.647, y_off=-0.2),
            r'H $\epsilon$':  dict(x=3970.079),
            r'[S II] 4069':   dict(x=4068.600),
            r'H $\delta$':    dict(x=4101.742, y_off=-0.2),
            r'H $\gamma$':    dict(x=4340.471),
            r'[O III] 4363':  dict(x=4363.210, y_off=-0.2),
            r'He I 4471':     dict(x=4471.479),
            r'He II 4686':    dict(x=4685.710),
            r'He I 4713':     dict(x=4713),
            r'H $\beta$':     dict(x=4861.333),
            r'He I 4922':     dict(x=4922),
            r'[O III]4959':   dict(x=4958.911),
            r'[O III]5007':   dict(x=5006.843),
            r'He I 5016':     dict(x=5016, y_off=-0.2),
            r'[N I] 5200':    dict(x=5200.257),
            r'[Fe VI] 5176':  dict(x=5176.040),
        }
    
    if text_y is None:
        ylim = ax.get_ylim()
        text_y = (ylim[0] + 3 * ylim[1]) / 4
        
    xlim = ax.get_xlim()
    for loop, k in enumerate(lineDict.keys()):
        c = 'C{0}'.format(loop%10)
        x = lineDict[k]['x']
        if (x < xlim[0]) or (x > xlim[1]):
            continue
        
        ax.axvline(x=x, ymin=ymin, ymax=ymax, ls=ls, color=c, alpha=alpha)
        
        x_off = lineDict[k].get('x_off', 0)
        y_off = lineDict[k].get('y_off', 0)
        x *= (1 + x_off)
        y = text_y * (1 + y_off)
        ax.text(x=x, y=y, s=k, rotation=90, fontsize=fontsize, color=c, ha=ha, 
                va=va, transform=ax.transData)
    
    return ax
    

def mask_spectrum(mask_list, wave, flux, ferr=None):
    '''
    Filter out the data that are masked.
    
    Parameters
    ----------
    mask_list : list
        The list of starting and end wavelengths of the range to mask.
    wave : 1D array
        The wavelength.
    flux : 1D array
        The flux.
    ferr (optional) : 1D array
        The uncertainty of the flux.
    '''
    mask = np.ones_like(wave, dtype=bool)
    for w1, w2 in mask_list:
        fltr = (wave >= w1) & (wave < w2)
        mask[fltr] = False
    
    if np.sum(mask) == 0:
        raise ValueError('No data is left!')
        
    if ferr is None:
        return wave[mask], flux[mask]
    else:
        return wave[mask], flux[mask], ferr[mask]
