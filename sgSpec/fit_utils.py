import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.modeling import fitting

__all__ = ['decomposer', 'plot_fit', 'multidecomposer', 'spec_perturb',
           'decomopse_Hbeta', 'integrate_Hbeta']


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


def plot_fit(res, axs=None, remove_masked=False):
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
    axr.axhspan(ymin=-1, ymax=1., ls='--', facecolor='none', edgecolor='gray')
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
