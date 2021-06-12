import numpy as np
from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model
from scipy.optimize import minimize
import extinction
ls_km = 2.99792458e5  # km/s

__all__ = ['gen_hbo3', 'add_hbo3', 'find_line_peak', 'line_fwhm',
           'extinction_ccm89']

Hbeta = 4862.721
OIII_4959 = 4960.295
OIII_5007 = 5008.239
r_OIII = 2.98  # Storey & Zeippen (2000) from Ruancun's paper
TOLERANCE = 1e-6

def gen_hbo3(z=0.001, amp_hb=1, amp_o5007=1, stddev_o5007=1, wave_o5007=None,
             amp_o5007_list=None, std_o5007_list=None, wav_o5007_list=None,
             nwind=0, stddev_bounds=(TOLERANCE, 500), scale_down=10):
    '''
    Get the fitting model of the narrow H-beta and [OIII] components.
    '''

    if wave_o5007 is None:
        mean_hb = Hbeta * (1 + z)
        mean_o4959 = OIII_4959 * (1 + z)
        mean_o5007 = OIII_5007 * (1 + z)
    else:
        mean_o5007 = wave_o5007
        mean_o4959 = OIII_4959 / OIII_5007 * wave_o5007
        mean_hb = Hbeta / OIII_5007 * wave_o5007

    stddev_hb = stddev_o5007 / mean_o5007 * mean_hb
    stddev_o4959 = stddev_o5007 / mean_o5007 * mean_o4959
    amp_o4959 = amp_o5007 / r_OIII

    bounds = dict(
        amplitude=(TOLERANCE, None),
        stddev=stddev_bounds
    )
    m_init = (models.Gaussian1D(amplitude=amp_o5007, mean=mean_o5007, stddev=stddev_o5007, name='[OIII]5007 C', bounds=bounds) +
              models.Gaussian1D(name='[OIII]4959 C', bounds=bounds) +
              models.Gaussian1D(amplitude=amp_hb, name='Hbeta NC', bounds=bounds))

    name_o31 = '[OIII]5007 C'
    name_o32 = '[OIII]4959 C'
    name_hb = 'Hbeta NC'
    m_init[name_o32].amplitude.tied = tie_ampl_4959C
    m_init[name_o32].stddev.tied = tie_std_4959C
    m_init[name_o32].mean.tied = tie_wave_4959C
    m_init[name_hb].stddev.tied = tie_std_hbNC
    m_init[name_hb].mean.tied = tie_wave_hbNC

    m_init = add_hbo3(m_init, amp_o5007_list, std_o5007_list, wav_o5007_list,
                 nwind, stddev_bounds, scale_down)

    return m_init

def add_hbo3(model, amp_o5007_list=None, std_o5007_list=None, wav_o5007_list=None,
             nwind=0, stddev_bounds=(TOLERANCE, 500), scale_down=10):
    '''
    Get the fitting model of the narrow H-beta and [OIII] components.
    '''
    bounds = dict(
        amplitude=(TOLERANCE, None),
        stddev=stddev_bounds
    )
    if amp_o5007_list is not None:
        nwind = len(amp_o5007_list)

    n_exist = 0
    for loop in range(nwind):
        if amp_o5007_list is None:
            amp_o5007 = model['[OIII]5007 C'].amplitude.value / (scale_down+loop)
        else:
            amp_o5007 = amp_o5007_list[loop]
        if std_o5007_list is None:
            std_o5007 = model['[OIII]5007 C'].stddev.value
        else:
            assert len(std_o5007_list) == nwind
            std_o5007 = std_o5007_list[loop]
        if wav_o5007_list is None:
            wav_o5007 = model['[OIII]5007 C'].mean.value
        else:
            assert len(wav_o5007_list) == nwind
            wav_o5007 = wav_o5007_list[loop]

        name_o31w = '[OIII]5007 W{0}'.format(n_exist + loop)
        while (loop == 0) & (n_exist < 1000):
            if name_o31w in model.submodel_names:
                n_exist += 1
                name_o31w = '[OIII]5007 W{0}'.format(n_exist + loop)
            else:
                break

        model_index = n_exist + loop
        name_o31w = '[OIII]5007 W{0}'.format(model_index)
        name_o32w = '[OIII]4959 W{0}'.format(model_index)
        name_hbw = 'Hbeta NW{0}'.format(model_index)
        model += models.Gaussian1D(amplitude=amp_o5007, mean=wav_o5007, stddev=std_o5007, name=name_o31w, bounds=bounds)
        model += models.Gaussian1D(name=name_o32w, bounds=bounds)
        model += models.Gaussian1D(name=name_hbw, bounds=bounds)

        exec(str_tie_ampl_4959W.format(model_index))
        exec(str_tie_std_4959W.format(model_index))
        exec(str_tie_wave_4959W.format(model_index))
        exec(str_tie_ampl_hbNW.format(model_index))
        exec(str_tie_std_hbNW.format(model_index))
        exec(str_tie_wave_hbNW.format(model_index))

        exec('''model['[OIII]4959 W{0}'].amplitude.tied = tie_ampl_4959W{0}'''.format(model_index))
        exec('''model['[OIII]4959 W{0}'].stddev.tied = tie_std_4959W{0}'''.format(model_index))
        exec('''model['[OIII]4959 W{0}'].mean.tied = tie_wave_4959W{0}'''.format(model_index))
        exec('''model['Hbeta NW{0}'].amplitude.tied = tie_ampl_hbNW{0}'''.format(model_index))
        exec('''model['Hbeta NW{0}'].stddev.tied = tie_std_hbNW{0}'''.format(model_index))
        exec('''model['Hbeta NW{0}'].mean.tied = tie_wave_hbNW{0}'''.format(model_index))

    return model

def find_line_peak(model, x0):
    '''
    Find the peak wavelength and flux of the model line profile.

    Parameters
    ----------
    model : Astropy model
        The model of the line profile. It should be all positive.
    x0 : float
        The initial guess of the wavelength.

    Returns
    -------
    w_peak, f_peak : floats
        The wavelength and flux of the peak of the line profile.
    '''
    func = lambda x: -1 * model(x)
    res = minimize(func, x0=x0)
    w_peak = res.x[0]
    try:
        f_peak = model(w_peak)[0]
    except:
        f_peak = model(w_peak)
    return w_peak, f_peak

def line_fwhm(model, x0, x1, x0_limit=None, x1_limit=None, fwhm_disp=None):
    '''
    Calculate the FWHM of the line profile.

    Parameters
    ----------
    model : Astropy model
        The model of the line profile. It should be all positive.
    x0, x1 : float
        The initial guesses of the wavelengths on the left and right sides.
    x0_limit, x1_limit (optional) : floats
        The left and right boundaries of the search.
    fwhm_disp (optional) : float
        The instrumental dispersion that should be removed from the FWHM.

    Returns
    -------
    fwhm : float
        The FWHM of the line, units: km/s.
    w_l, w_r : floats
        The wavelength and flux of the peak of the line profile.
    w_peak : float
        The wavelength of the line peak.
    '''
    xc = (x0 + x1) / 2
    w_peak, f_peak = find_line_peak(model, xc)
    f_half = f_peak / 2

    func = lambda x: np.abs(model(x) - f_half)

    if x0_limit is not None:
        bounds = ((x0_limit, w_peak),)
    else:
        bounds = None
    res_l = minimize(func, x0=x0, bounds=bounds)

    if x1_limit is not None:
        bounds = ((w_peak, x1_limit),)
    else:
        bounds = None
    res_r = minimize(func, x0=x1, bounds=bounds)
    w_l = res_l.x[0]
    w_r = res_r.x[0]

    fwhm_w = (w_r - w_l)
    if fwhm_disp is not None:  # Correct for instrumental dispersion
        fwhm_w = np.sqrt(fwhm_w**2 - fwhm_disp**2)

    fwhm = fwhm_w / w_peak * ls_km
    return fwhm, w_l, w_r, w_peak

@custom_model
def extinction_ccm89(x, a_v=0, r_v=3.1):
    '''
    The extinction model of Cardelli et al. (1989).

    Parameters
    ----------
    x : array like
        Wavelength, units: Angstrom.
    a_v : float
        Scaling parameter, A_V: extinction in magnitudes at characteristic
        V-band wavelength.
    r_v : float
        Ratio of total to selective extinction, A_V / E(B-V).

    Returns
    -------
    f : array like
        The fraction of out emitting flux.
    '''
    f =10**(-0.4 * extinction.ccm89(x, a_v, r_v))
    return f

# Tie parameters
str_tie_ampl_4959W = '''
def tie_ampl_4959W{0}(model):
    return model['[OIII]5007 W{0}'].amplitude.value / r_OIII
'''

str_tie_std_4959W = '''
def tie_std_4959W{0}(model):
    return model['[OIII]5007 W{0}'].stddev.value / model['[OIII]5007 W{0}'].mean.value * model['[OIII]4959 W{0}'].mean.value
'''

str_tie_wave_4959W = '''
def tie_wave_4959W{0}(model):
    return OIII_4959 * (model['[OIII]5007 W{0}'].mean.value / OIII_5007)
'''

str_tie_ampl_hbNW = '''
def tie_ampl_hbNW{0}(model):
    r = model['[OIII]5007 W{0}'].amplitude.value / model['[OIII]5007 C'].amplitude.value
    return r * model['Hbeta NC'].amplitude.value
'''

str_tie_std_hbNW = '''
def tie_std_hbNW{0}(model):
    return model['[OIII]5007 W{0}'].stddev.value / model['[OIII]5007 W{0}'].mean.value * model['Hbeta NW{0}'].mean.value
'''

str_tie_wave_hbNW = '''
def tie_wave_hbNW{0}(model):
    return Hbeta * (model['[OIII]5007 W{0}'].mean.value / OIII_5007)
'''

def tie_ampl_4959C(model):
    return model['[OIII]5007 C'].amplitude.value / r_OIII

def tie_std_4959C(model):
    return model['[OIII]5007 C'].stddev.value / model['[OIII]5007 C'].mean.value * model['[OIII]4959 C'].mean.value

def tie_wave_4959C(model):
    return OIII_4959 * (model['[OIII]5007 C'].mean.value / OIII_5007)

def tie_std_hbNC(model):
    return model['[OIII]5007 C'].stddev.value / model['[OIII]5007 C'].mean.value * model['Hbeta NC'].mean.value

def tie_wave_hbNC(model):
    return Hbeta * (model['[OIII]5007 C'].mean.value / OIII_5007)
