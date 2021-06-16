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
    m_init[name_o32].amplitude.tied = tier_line_ratio(name_o32, name_o31, r_OIII)
    m_init[name_o32].stddev.tied = tier_line_width(name_o32, name_o31)
    m_init[name_o32].mean.tied = tier_line_center(name_o32, name_o31, OIII_4959, OIII_5007)
    m_init[name_hb].stddev.tied = tier_line_width(name_hb, name_o31)
    m_init[name_hb].mean.tied = tier_line_center(name_hb, name_o31, Hbeta, OIII_5007)


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

        model[name_o32w].amplitude.tied = tier_line_ratio(name_o32w, name_o31w, r_OIII)
        model[name_o32w].stddev.tied = tier_line_width(name_o32w, name_o31w)
        model[name_o32w].mean.tied = tier_line_center(name_o32w, name_o31w, OIII_4959, OIII_5007)
        model[name_hbw].amplitude.tied = tier_line_ratio(name_hbw, 'Hbeta NC', ratio_names=[name_o31w, '[OIII]5007 C'])
        model[name_hbw].stddev.tied = tier_line_width(name_hbw, name_o31w)
        model[name_hbw].mean.tied = tier_line_center(name_hbw, name_o31w, Hbeta, OIII_5007)

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
class tier_line_ratio(object):

    def __init__(self, name_fit, name_ref, ratio=None, ratio_names=None):

        self._name_fit = name_fit
        self._name_ref = name_ref
        self._ratio = ratio
        self._ratio_names = ratio_names

        if ((self._ratio is None) and (self._ratio_names is None)):
            raise keyError('Need to provide ratio or _ratio_names!')
        elif ((self._ratio is not None) and (self._ratio_names is not None)):
            raise keyError('Cannot set both ratio and _ratio_names!')

    def __repr__(self):

        if self._ratio is not None:
            return "<Set the amplitude of '{0}' to 1/{1} that of '{2}'>".format(self._name_fit, self._ratio, self._name_ref)
        else:
            return "<Set the amplitude of '{0}' according to '{1}' x '{2[0]}'/'{2[1]}'>".format(self._name_fit, self._name_ref, self._ratio_names)

    def __call__(self, model):

        if self._ratio is not None:
            r = 1 / self._ratio
        else:
            r = model[self._ratio_names[0]].amplitude.value / model[self._ratio_names[1]].amplitude.value

        return model[self._name_ref].amplitude.value * r


class tier_line_width(object):

    def __init__(self, name_fit, name_ref):
        self._name_fit = name_fit
        self._name_ref = name_ref

    def __repr__(self):
        return "<Set the line width of '{0}' the same as that of '{1}'>".format(self._name_fit, self._name_ref)

    def __call__(self, model):
        return model[self._name_ref].stddev.value / model[self._name_ref].mean.value * model[self._name_fit].mean.value


class tier_line_center(object):

    def __init__(self, name_fit, name_ref, wavec_fit, wavec_ref):
        self._name_fit = name_fit
        self._name_ref = name_ref
        self._wavec_fit = wavec_fit
        self._wavec_ref = wavec_ref

    def __repr__(self):
        return "<Set the line center of '{0}' ({2}) according to that of '{1}' ({3})>".format(self._name_fit, self._name_ref, self._wavec_fit, self._wavec_ref)

    def __call__(self, model):
        return self._wavec_fit / self._wavec_ref * model[self._name_ref].mean.value
