import numpy as np
from astropy.modeling import models, fitting
from astropy.modeling.models import custom_model
from scipy.optimize import minimize
import extinction
ls_km = 2.99792458e5  # km/s

__all__ = ['gen_hbo3', 'add_hbo3', 'find_line_peak', 'line_fwhm',
           'extinction_ccm89', 'Line_Gaussian', 'get_line_multigaussian',
           'tier_line_ratio', 'tier_line_width', 'tier_line_mean',
           'tier_line_center', 'tier_line_sigma', 'fix_line_profile',
           'duplicate_line']

Hbeta = 4862.721
OIII_4959 = 4960.295
OIII_5007 = 5008.239
r_OIII = 2.98  # Storey & Zeippen (2000) from Ruancun's paper
TOLERANCE = 1e-6


def fix_line_profile(model, name_ref, name_fix):
    '''
    Fix the one line profile to the other.

    Parameters
    ----------
    model : astropy.modeling.CompoundModel
        The model that consists two sets of line profiles.
    name_ref : str
        The name of the reference line.
    name_fix : str
        The name of the line to be fixed the profile.
    '''
    assert model.n_submodels > 1, 'There are not additional components to fix!'

    ncomp_ref = 0
    ncomp_fix = 0
    for n in model.submodel_names:
        if name_ref in n.split(': '):
            ncomp_ref += 1
        elif name_fix in n.split(': '):
            ncomp_fix += 1

    #print('Find {0} for {1} and {2} for {3}'.format(ncomp_ref, name_ref, ncomp_fix, name_fix))

    if ncomp_ref == 0:
        raise KeyError('The model does not consist {0}'.format(name_ref))
    elif ncomp_fix == 0:
        raise KeyError('The model does not consist {0}'.format(name_fix))
    elif ncomp_ref != ncomp_fix:
        raise KeyError('The model components does not match!')

    name_ref_0 = '{0}: 0'.format(name_ref)
    name_fix_0 = '{0}: 0'.format(name_fix)

    # Fix amplitude -- all respect to the first component
    if ncomp_ref > 1:
        for n in range(ncomp_ref - 1):
            # Set the tier
            name_ref_n = '{0}: {1}'.format(name_ref, n+1)
            name_fix_n = '{0}: {1}'.format(name_fix, n+1)
            model[name_fix_n].amplitude.tied = tier_line_ratio(name_fix_n, name_ref_n, ratio_names=[name_fix_0, name_ref_0])
            # Run it
            model[name_fix_n].amplitude.value = model[name_fix_n].amplitude.tied(model)

    # Fix center -- all respect to the first component
    if ncomp_ref > 1:
        wavec_ref = model[name_ref_0].center.value
        wavec_fix = model[name_fix_0].center.value
        for n in range(ncomp_ref - 1):
            # Set the tier
            name_ref_n = '{0}: {1}'.format(name_ref, n+1)
            name_fix_n = '{0}: {1}'.format(name_fix, n+1)
            model[name_fix_n].center.tied = tier_line_center(name_fix_n, name_ref_n, wavec_fix, wavec_ref)
            # Run it
            model[name_fix_n].center.value = model[name_fix_n].center.tied(model)

    # Fix sigma
    if ncomp_ref == 1:
        model[name_fix].sigma.tied = tier_line_sigma(name_fix, name_ref)
    else:
        for n in range(ncomp_ref):
            # Set the tier
            name_ref_n = '{0}: {1}'.format(name_ref, n)
            name_fix_n = '{0}: {1}'.format(name_fix, n)
            model[name_fix_n].sigma.tied = tier_line_sigma(name_fix_n, name_ref_n)
            # Run it
            model[name_fix_n].sigma.value = model[name_fix_n].sigma.tied(model)

    return model


def gen_hbo3_rest(amp_hb=1, amp_o5007=1, stddev_o5007=1, wave_o5007=None,
                  amp_o5007_list=None, std_o5007_list=None, wav_o5007_list=None,
                  nwind=0, stddev_bounds=(TOLERANCE, 500), delta_z=0.001,
                  scale_down=10):
    '''
    Get the fitting model of the narrow H-beta and [OIII] components.
    '''

    if wave_o5007 is None:
        mean_hb = Hbeta
        mean_o4959 = OIII_4959
        mean_o5007 = OIII_5007
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
    m_init[name_o32].mean.tied = tier_line_mean(name_o32, name_o31, OIII_4959, OIII_5007)
    m_init[name_hb].stddev.tied = tier_line_width(name_hb, name_o31)
    m_init[name_hb].mean.tied = tier_line_mean(name_hb, name_o31, Hbeta, OIII_5007)


    m_init = add_hbo3(m_init, amp_o5007_list, std_o5007_list, wav_o5007_list,
                 nwind, stddev_bounds, scale_down)

    return m_init


def add_hbo3_rest(model, amp_o5007_list=None, std_o5007_list=None, wav_o5007_list=None,
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
        model[name_o32w].mean.tied = tier_line_mean(name_o32w, name_o31w, OIII_4959, OIII_5007)
        model[name_hbw].amplitude.tied = tier_line_ratio(name_hbw, 'Hbeta NC', ratio_names=[name_o31w, '[OIII]5007 C'])
        model[name_hbw].stddev.tied = tier_line_width(name_hbw, name_o31w)
        model[name_hbw].mean.tied = tier_line_mean(name_hbw, name_o31w, Hbeta, OIII_5007)

    return model


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
    m_init[name_o32].mean.tied = tier_line_mean(name_o32, name_o31, OIII_4959, OIII_5007)
    m_init[name_hb].stddev.tied = tier_line_width(name_hb, name_o31)
    m_init[name_hb].mean.tied = tier_line_mean(name_hb, name_o31, Hbeta, OIII_5007)


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
        model[name_o32w].mean.tied = tier_line_mean(name_o32w, name_o31w, OIII_4959, OIII_5007)
        model[name_hbw].amplitude.tied = tier_line_ratio(name_hbw, 'Hbeta NC', ratio_names=[name_o31w, '[OIII]5007 C'])
        model[name_hbw].stddev.tied = tier_line_width(name_hbw, name_o31w)
        model[name_hbw].mean.tied = tier_line_mean(name_hbw, name_o31w, Hbeta, OIII_5007)

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


def duplicate_line(model, line_name='Line'):
    '''
    Duplicate the line with a different model name.
    '''
    model_new = model.copy()

    model_new.name = '{0}'.format(line_name)
    if model.n_submodels > 1:
        for loop in range(model_new.n_submodels):
            model_new[loop].name = '{0}: {1}'.format(line_name, loop)

    return model_new


def get_line_multigaussian(n=1, line_name='Line', **kwargs):
    '''
    Get a multigaussian line model.

    Parameters
    ----------
    n : int
        Number of Gaussian components.
    line_name : str
        The name of the line. Each component has an additional index,
        starting from 0, e.g., "Line 0".
    amplitude (optional) : list
        The value of the line amplitude.
    center (optional) : list
        The value of the line center.
    sigma (optional) : list
        The value of the line sigma.
    amplitude_bound (optional) : tuple or list
        The bound(s) of the line amplitude.
    center_bound (optional) : tuple or list
        The bound(s) of the line center.
    sigma_bound (optional) : tuple or list
        The bound(s) of the line sigma.

    Returns
    -------
    model : The sum of Line_Gaussian.
    '''
    assert isinstance(n, int) & (n > 0), 'We only accept n as >0 integer!'
    parList = ['amplitude', 'center', 'sigma']
    bouList = ['amplitude_bound', 'center_bound', 'sigma_bound']

    # Check the parameters
    for kw in kwargs:
        if kw not in parList + bouList:
            raise KeyError('{0} is not recognized!'.format(kw))

    # Generate the model
    if n > 1:
        model = Line_Gaussian(name='{0}: 0'.format(line_name))
        for loop in range(n-1):
            model += Line_Gaussian(name='{0}: {1}'.format(line_name, loop+1))
        model.name = '{0}'.format(line_name)
    else:
        model = Line_Gaussian(name='{0}'.format(line_name))

    # Set the parameters
    for kw in parList:
        kv = kwargs.get(kw, None)

        if kv is not None:
            assert isinstance(kv, list), 'We only accept {0} as a list!'.format(kw)
            assert len(kv) <= n, 'The length of {0} is larger than n!'.format(kw)

            for loop, v in enumerate(kv):
                model[loop].__setattr__(kw, v)

    # Set the bounds of the parameters
    for kw in bouList:
        kv = kwargs.get(kw, None)
        pn, pa = kw.split('_')

        if isinstance(kv, tuple):
            assert len(kv) == 2, 'The {0} should contain 2 elements!'.format(kw)

            for loop in range(n):
                p = model[loop].__getattribute__(pn)
                p.__setattr__(pa, kv)

        elif isinstance(kv, list):
            assert len(kv) <= n, 'The length of {0} is larger than n!'.format(kw)

            for loop, bou in enumerate(kv):
                p = model[loop].__getattribute__(pn)
                p.__setattr__(pa, bou)

        elif kv is not None:
            raise ValueError('Cannot recognize {0} ({1})'.format(kw, kv))

    return model


@custom_model
def Line_Gaussian(x, amplitude=1, center=5000., sigma=200.):
    '''
    The Gaussian line profile with the sigma as the velocity.

    Parameters
    ----------
    x : array like
        Wavelength, units: arbitrary.
    amplitude : float
        The amplitude of the line profile.
    center : float
        The central wavelength of the line profile, units: same as x.
    sigma : float
        The velocity dispersion of the line profile, units: km/s.
    '''
    v = (x - center) / center * ls_km  # convert to velocity (km/s)
    fl = amplitude * np.exp(-0.5 * (v / sigma)**2)
    return fl


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


class tier_line_center(object):

    def __init__(self, name_fit, name_ref, wavec_fit, wavec_ref):
        self._name_fit = name_fit
        self._name_ref = name_ref
        self._wavec_fit = wavec_fit
        self._wavec_ref = wavec_ref

    def __repr__(self):
        return "<Set the line center of '{0}' ({2}) according to that of '{1}' ({3})>".format(self._name_fit, self._name_ref, self._wavec_fit, self._wavec_ref)

    def __call__(self, model):
        return self._wavec_fit / self._wavec_ref * model[self._name_ref].center.value


class tier_line_sigma(object):

    def __init__(self, name_fit, name_ref):
        self._name_fit = name_fit
        self._name_ref = name_ref

    def __repr__(self):
        return "<Set the sigma of '{0}' the same as that of '{1}'>".format(self._name_fit, self._name_ref)

    def __call__(self, model):
        return model[self._name_ref].sigma.value


class tier_line_width(object):

    def __init__(self, name_fit, name_ref):
        self._name_fit = name_fit
        self._name_ref = name_ref

    def __repr__(self):
        return "<Set the line width of '{0}' the same as that of '{1}'>".format(self._name_fit, self._name_ref)

    def __call__(self, model):
        return model[self._name_ref].stddev.value / model[self._name_ref].mean.value * model[self._name_fit].mean.value


class tier_line_mean(object):

    def __init__(self, name_fit, name_ref, wavec_fit, wavec_ref):
        self._name_fit = name_fit
        self._name_ref = name_ref
        self._wavec_fit = wavec_fit
        self._wavec_ref = wavec_ref

    def __repr__(self):
        return "<Set the line center of '{0}' ({2}) according to that of '{1}' ({3})>".format(self._name_fit, self._name_ref, self._wavec_fit, self._wavec_ref)

    def __call__(self, model):
        return self._wavec_fit / self._wavec_ref * model[self._name_ref].mean.value
