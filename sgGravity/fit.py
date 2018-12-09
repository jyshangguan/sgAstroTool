import numpy as np
from scipy.integrate import quad
import scipy.special as special
from scipy.optimize import curve_fit
from .plot import flagDA

__all__ = ["Gaussian", "vis_gauss1d", "img2d_gauss", "vis2d_gauss", "vis_Inc_Gauss",
           "vis_gauss2d", "fit_VIS_ruv", "fit_VIS_uv", "visfit"]

pi = np.pi

def Gaussian(x, A, sigma):
    """
    The Gaussian function

    Parameters
    ----------
    x : array like
        The independent variable.
    A : float
        The amplitude.
    sigma : float
        The sigma of a Gaussian profile.

    Returns
    -------
    y : array like
        The dependent variable.
    """
    x = np.atleast_1d(x)
    y = A * np.exp(-0.5 * x**2 / sigma**2)
    return y


def vis_gauss1d(ruv, sigma_lm, A=1., offset=0):
    """
    A Gaussian model for the 1D visibility data.

    Parameters
    ----------
    ruv : array like
        The uv distance.
    sigma_lm : float
        The sigma of the Gaussian profile in real space.
    A : float, default: 1
        The amplitude of the visibility.
    offset : float, default: 0
        The offset of the visibility.
    """
    sigma_uv = 0.5 / pi / sigma_lm
    y = Gaussian(ruv, A, sigma_uv) - offset
    return y


def img2d_gauss(l, m, A, sigma_x, sigma_y, l0, m0, pa, offset=0):
    """
    A 2D Gaussian model on the image plane.

    Parameters
    ----------
    l : array like
        The l axis coordinate.
    m : array like
        The m axis coordinate.
    A : float
        The amplitude of the source emission.
    sigma_x : float
        The sigma of the x direction.
    sigma_y : float
        The sigma of the y direction.
    l0 : float
        The offset of the source center on l axis from the origin of
        the xy coordinates.
    m0 : float
        The offset of the source center on m axis from the origin of
        the xy coordinates.
    pa : float
        The phase angle rotating from lm coordinates to xy coordinates
        where the major and minor axes of the source are on the x and y
        axes, units: radian.
    offset : float, default: 0
        The offset of the model.

    Returns
    -------
    I : array like
        The intensity of the source emission.
    """
    l = np.atleast_1d(l)
    m = np.atleast_1d(m)
    a = 0.5 * ((np.cos(pa) / sigma_x)**2. + (np.sin(pa) / sigma_y)**2.)
    b = 0.5 * np.sin(2. * pa) * (sigma_x**-2. - sigma_y**-2.)
    c = 0.5 * ((np.sin(pa) / sigma_x)**2. + (np.cos(pa) / sigma_y)**2.)
    p = a * (l - l0)**2. + b * (l - l0) * (m - m0) + c * (m - m0)**2.
    I = A * np.exp(-p)
    return I


def vis2d_gauss(x, A, sigma_x, sigma_y, l0, m0, pa, offset=0):
    """
    A 2D Gaussian model on the uv plane. The product image and visibility are
    squares with the coordinates determined by the input axis information x.

    Parameters
    ----------
    x : 1D array
        The coordinates of one aspect of the squared image.  The array should
        be uniformly increasing.
    A : float
        The amplitude of the visibility.
    sigma_x : float
        The sigma of the x direction.
    sigma_y : float
        The sigma of the y direction.
    l0 : float
        The offset of the source center on l axis from the origin of
        the xy coordinates.
    m0 : float
        The offset of the source center on m axis from the origin of
        the xy coordinates.
    pa : float
        The phase angle rotating from lm coordinates to xy coordinates
        where the major and minor axes of the source are on the x and y
        axes, units: radian.
    offset : float, default: 0
        The offset of the model.

    Returns
    -------
    res : dict
        The dict of the results.
        vis : array_like
            The complex visibility.
        img : array_like
            The image.
        uu : array_like
            The u coordinates of the visibility.
        vv : array_like
            The v coordinates of the visivility.

    Notes
    -----
    This function is mainly for demonstration or test.
    """
    ll, mm = np.meshgrid(x, x)
    ii = img2d_gauss(ll, mm, A, sigma_x, sigma_y, l0, m0, pa, offset)
    i_l, i_m = ii.shape
    fsmp = 1/(x[1] - x[0])
    fii = np.fft.fft2(ii) / sigma_x / sigma_y / 2 / pi / fsmp**2
    u  = np.fft.fftfreq(len(x), 1./fsmp)
    uu, vv = np.meshgrid(u, u)
    idx = np.argsort(u)
    uu = uu[:, idx]
    vv = vv[idx, :]
    fii = fii[:, idx]
    fii = fii[idx, :]
    res = {
        "vis": fii,
        "img": ii,
        "uu": uu,
        "vv": vv
    }
    return res


def vis_inc_gauss(uv, sigma, A=1, i=0, pa=0, offset=0):
    """
    The model of inclined Gaussion profile in real space.

    Parameters
    ----------
    uv : list of [u, v]
        The uv coordinates.
    sigma : float
        The sigma of a Gaussian profile of the source image.
    A : float, default: 1
        The amplitude of the visibility.
    i : float, default: 0
        The inclination angle, units: radian.
    pa : float, default: 0
        The position angle, units: radian.
    offset : float, default: 0
        The offset of the model.

    Returns
    -------
    F_rho : float
        The visibility at given uv.

    Notes
    -----
    The position angle is the same as e.g., 2018ApJ...865...37M, but we are
    the same if u and v are switched.  There is freedom to choose the axes.
    """
    up = uv[0] * np.cos(pa) + uv[1] * np.sin(pa)
    vp = uv[1] * np.cos(pa) - uv[0] * np.sin(pa)
    rho = np.sqrt(up**2. + (vp * np.cos(i))**2.)
    A = A / 2. / pi / sigma**2.
    Ifunc  = lambda r: Gaussian(r, A, sigma)
    j0func = lambda r: special.jv(0, 2*pi*rho * r)
    integral = lambda r: Ifunc(r) * j0func(r) * r
    F_rho = 2 * pi * quad(integral, 0, np.inf)[0]
    return F_rho


def vis_Inc_Gauss(uv, sigma, A=1, i=0, pa=0, offset=0):
    """
    The model of inclined Gaussion profile in visibility space.
    Wrap up vis_inc_gauss() to enable calculating a list of uv coordinates.

    Parameters
    ----------
    uv : array_like
        The uv coordinates with shape (*, 2).
    sigma : float
        The sigma of a Gaussian profile of the source image.
    A : float, default: 1
        The amplitude of the visibility.
    i : float, default: 0
        The inclination angle, units: radian.
    pa : float, default: 0
        The position angle, units: radian.
    offset : float, default: 0
        The offset of the model.

    Returns
    -------
    F_rho : numpy 1D array
        The visibility at given uv coordinates.
    """
    uv = np.atleast_2d(uv)
    F_rho = []
    for loop in range(uv.shape[0]):
        F_rho.append(vis_inc_gauss(uv[loop, :], sigma, A, i, pa, offset))
    F_rho = np.array(F_rho)
    return F_rho


def vis_gauss2d(uv, sigma_l, sigma_m=None, A=1, pa=0, offset=0):
    """
    A Gaussian model for the 2D visibility data.

    Parameters
    ----------
    uv : array_like
        The uv coordinates with shape (*, 2).
    sigma_l : float
        The sigma of a Gaussian profile of the source image along l axis.
    sigma_m : float (optional)
        The sigma of a Gaussian profile of the source image along m axis.
        It equals to sigma_l if not given.
    A : float, default: 1
        The amplitude of the visibility.
    pa : float, default: 0
        The position angle, units: radian.
    offset : float, default: 0
        The offset of the model.

    Returns
    -------
    y : array_like
        The visibility at given uv.
    """
    uv = np.atleast_2d(uv)
    up = uv[:, 0] * np.cos(pa) + uv[:, 1] * np.sin(pa)
    vp = uv[:, 1] * np.cos(pa) - uv[:, 0] * np.sin(pa)
    if sigma_m is None:
        sigma_m = sigma_l
    sigma_u = 1. / (2. * pi * sigma_l)
    sigma_v = 1. / (2. * pi * sigma_m)
    y = A * np.exp(-0.5 * ((up / sigma_u)**2. + (vp / sigma_v)**2.) ) - offset
    return y


def fit_VIS_ruv(ruv, vis, viserr, p0, time=None, baseline=None, channel=None,
           mask=None, filter_ruv=False):
    """
    Fit the visibility with a radial Gaussian function.

    Parameters
    ----------
    ruv : numpy 3-dim array
        The ruv data which should have the dimensions [time, baseline, channel].
    vis : numpy 3-dim array
        The visibility data which should have the dimensions [time, baseline, channel].
    viserr : numpy 3-dim array
        The visibility error which should have the dimensions [time, baseline, channel].
    p0 : list
        The initial guess of "sigmax" and "A" of vis_gauss1d().
    time : list (optional)
        List the column number of the time dimension to be fitted.
    baseline : list (optional)
        List the column number of the baseline dimension to be fitted.
    channel : list (optional)
        List the column number of the channel to be fitted.
    mask : numpy 3-dim bool array (optional)
        Additional mask to the data array with dimensions the same as da.

    Returns
    -------
    (ruv_fit, vis_fit) : tuple
        The flattened and sorted ruv data used in the fitting and the best-fit vis data
        flattened according to ruv.
        Units of ruv_fit is mas.
    (A, A_err) : tupple
        The best-fit parameter "A" and its uncertainty
    (sigmax, sigmax_err) : tupple
        The best-fit parameter "sigmax" and its uncertainty
    """
    nt, nb, nc = ruv.shape
    if time is None:
        tFlag = []
    else:
        tFlag = list(set(np.arange(nt)) - set(time))
    if baseline is None:
        bFlag = []
    else:
        bFlag = list(set(np.arange(nb)) - set(baseline))
    if channel is None:
        cFlag = []
    else:
        cFlag = list(set(np.arange(nc)) - set(channel))
    ruv_f  = flagDA(ruv, tFlag, bFlag, cFlag, mask)
    vis_f  = flagDA(vis, tFlag, bFlag, cFlag, mask)
    vise_f = flagDA(viserr, tFlag, bFlag, cFlag, mask)
    fltr   = np.logical_not(ruv_f.mask | vis_f.mask | vise_f.mask)
    if filter_ruv:
        fltr_ruv = ruv_f > 1e-10
        fltr = fltr & fltr_ruv
    if np.sum(fltr) > 2:
        pass
    else:
        raise RuntimeError("There are not enough data!")
    x  = ruv_f[fltr]
    y  = vis_f[fltr]
    ye = vise_f[fltr]
    coeff, var_matrix = curve_fit(vis_gauss1d, x, y, p0=p0, sigma=ye)
    ce = np.sqrt(np.diag(var_matrix))
    ruv_fit = np.sort(x)
    vis_fit = vis_gauss1d(ruv_fit, *coeff)
    sigmax, A = coeff
    sigmax_err, A_err = ce
    return (ruv_fit, vis_fit), (sigmax, sigmax_err), (A, A_err)


def fit_VIS_uv(u, v, vis, viserr, p0, time=None, baseline=None, channel=None,
           mask=None, filter_ruv=False):
    """
    Fit the visibility with a two dimensional Gaussian model.

    Parameters
    ----------
    u : numpy 3-dim array
        The u coordinate which should have the dimensions [time, baseline, channel].
    v : numpy 3-dim array
        The v coordinate which should have the dimensions [time, baseline, channel].
    vis : numpy 3-dim array
        The visibility data which should have the dimensions [time, baseline, channel].
    viserr : numpy 3-dim array
        The visibility error which should have the dimensions [time, baseline, channel].
    p0 : list
        The initial guess of "sigma_l", "sigma_m", "A", "pa", and "offset" of vis_gauss2d().
    time : list (optional)
        List the column number of the time dimension to be fitted.
    baseline : list (optional)
        List the column number of the baseline dimension to be fitted.
    channel : list (optional)
        List the column number of the channel to be fitted.
    mask : numpy 3-dim bool array (optional)
        Additional mask to the data array with dimensions the same as da.

    Returns
    -------
    (u_fit, v_fit, vis_fit) : tuple
        The flattened uv coordinates used in the fitting and the best-fit
        vis data flattened consistently with ruv.  Units of (u, v) are mas.
    c : list
        The best-fit parameters.
    ce : list
        The uncertainties of the best-fit parameters.

    Notes
    -----
    The offset of the 2D Gaussian model is not allowed at present!
    """
    nt, nb, nc = u.shape
    if time is None:
        tFlag = []
    else:
        tFlag = list(set(np.arange(nt)) - set(time))
    if baseline is None:
        bFlag = []
    else:
        bFlag = list(set(np.arange(nb)) - set(baseline))
    if channel is None:
        cFlag = []
    else:
        cFlag = list(set(np.arange(nc)) - set(channel))
    u_f    = flagDA(u, tFlag, bFlag, cFlag, mask)
    v_f    = flagDA(v, tFlag, bFlag, cFlag, mask)
    vis_f  = flagDA(vis, tFlag, bFlag, cFlag, mask)
    vise_f = flagDA(viserr, tFlag, bFlag, cFlag, mask)
    fltr   = np.logical_not(u_f.mask | v_f.mask | vis_f.mask | vise_f.mask)
    if filter_ruv:
        fltr_ruv = np.sqrt(u_f**2. + v_f**2.) > 1e-10
        fltr = fltr & fltr_ruv
    if np.sum(fltr) > len(p0):
        pass
    else:
        raise RuntimeError("There are not enough degree of freedom!")
    u_fit = u_f[fltr]
    v_fit = v_f[fltr]
    x  = np.transpose([u_fit, v_fit])
    y  = vis_f[fltr]
    ye = vise_f[fltr]
    c, var_matrix = curve_fit(vis_gauss2d, x, y, p0=p0, sigma=ye,
                              bounds=(0, [np.inf, np.inf, 1., pi]))
    ce = np.sqrt(np.diag(var_matrix))
    vis_fit = vis_gauss2d(x, *c)
    return (u_fit, v_fit, vis_fit), c, ce


def fit_VIS_IG(u, v, vis, viserr, p0, time=None, baseline=None, channel=None,
           mask=None, filter_ruv=False):
    """
    Fit the visibility with a two dimensional axisymmetric model.  The only model
    currently available is an inclined Gaussian disk.

    Parameters
    ----------
    u : numpy 3-dim array
        The u coordinate which should have the dimensions [time, baseline, channel].
    v : numpy 3-dim array
        The v coordinate which should have the dimensions [time, baseline, channel].
    vis : numpy 3-dim array
        The visibility data which should have the dimensions [time, baseline, channel].
    viserr : numpy 3-dim array
        The visibility error which should have the dimensions [time, baseline, channel].
    p0 : list
        The initial guess of "sigma", "A", "i", "pa", and "offset" of vis_Inc_Gauss().
    time : list (optional)
        List the column number of the time dimension to be fitted.
    baseline : list (optional)
        List the column number of the baseline dimension to be fitted.
    channel : list (optional)
        List the column number of the channel to be fitted.
    mask : numpy 3-dim bool array (optional)
        Additional mask to the data array with dimensions the same as da.

    Returns
    -------
    (u_fit, v_fit, vis_fit) : tuple
        The flattened uv coordinates used in the fitting and the best-fit
        vis data flattened consistently with ruv.  Units of (u, v) are mas.
    c : list
        The best-fit parameters.
    ce : list
        The uncertainties of the best-fit parameters.
    """
    nt, nb, nc = u.shape
    if time is None:
        tFlag = []
    else:
        tFlag = list(set(np.arange(nt)) - set(time))
    if baseline is None:
        bFlag = []
    else:
        bFlag = list(set(np.arange(nb)) - set(baseline))
    if channel is None:
        cFlag = []
    else:
        cFlag = list(set(np.arange(nc)) - set(channel))
    u_f    = flagDA(u, tFlag, bFlag, cFlag, mask)
    v_f    = flagDA(v, tFlag, bFlag, cFlag, mask)
    vis_f  = flagDA(vis, tFlag, bFlag, cFlag, mask)
    vise_f = flagDA(viserr, tFlag, bFlag, cFlag, mask)
    fltr   = np.logical_not(u_f.mask | v_f.mask | vis_f.mask | vise_f.mask)
    if filter_ruv:
        fltr_ruv = np.sqrt(u_f**2. + v_f**2.) > 1e-10
        fltr = fltr & fltr_ruv
    if np.sum(fltr) > len(p0):
        pass
    else:
        raise RuntimeError("There are not enough degree of freedom!")
    u_fit = u_f[fltr]
    v_fit = v_f[fltr]
    x  = np.transpose([u_fit, v_fit])
    y  = vis_f[fltr]
    ye = vise_f[fltr]
    if len(p0) < 3:
        pbounds = (-np.inf, np.inf)
    elif len(p0) == 3:
        pbounds = (0, [np.inf, np.inf, pi/2.])
    elif len(p0) == 4:
        pbounds = (0, [np.inf, np.inf, pi/2., pi])
    else:
        raise RuntimeError("The initial guess is incorrect!")
    c, var_matrix = curve_fit(vis_Inc_Gauss, x, y, p0=p0, sigma=ye, bounds=pbounds)
    ce = np.sqrt(np.diag(var_matrix))
    vis_fit = vis_Inc_Gauss(x, *c)
    return (u_fit, v_fit, vis_fit), c, ce


from .jad import gaussfit
def visfit(ruv, vis, viserr, p0, time=None, baseline=None, channel=None,
           mask=None, filter_ruv=False):
    """
    Fit the visibility with gaussfit.fit() by JAD.

    Parameters
    ----------
    ruv : numpy 3-dim array
        The ruv data which should have the dimensions [time, baseline, channel].
    vis : numpy 3-dim array
        The visibility data which should have the dimensions [time, baseline, channel].
    viserr : numpy 3-dim array
        The visibility error which should have the dimensions [time, baseline, channel].
    p0 : list
        The initial guess of "A" and "sigmax" of gaussfit.fit().
    time : list (optional)
        List the column number of the time dimension to be fitted.
    baseline : list (optional)
        List the column number of the baseline dimension to be fitted.
    channel : list (optional)
        List the column number of the channel to be fitted.
    mask : numpy 3-dim bool array (optional)
        Additional mask to the data array with dimensions the same as da.

    Returns
    -------
    (ruv_fit, vis_fit) : tuple
        The flattened and sorted ruv data used in the fitting and the best-fit vis data
        flattened according to ruv.
        Units of ruv_fit is mas.
    (A, A_err) : tupple
        The best-fit parameter "A" and its uncertainty
    (sigmax, sigmax_err) : tupple
        The best-fit parameter "sigmax" and its uncertainty
    """
    nt, nb, nc = ruv.shape
    if time is None:
        tFlag = []
    else:
        tFlag = list(set(np.arange(nt)) - set(time))
    if baseline is None:
        bFlag = []
    else:
        bFlag = list(set(np.arange(nb)) - set(baseline))
    if channel is None:
        cFlag = []
    else:
        cFlag = list(set(np.arange(nc)) - set(channel))
    ruv_f  = flagDA(ruv, tFlag, bFlag, cFlag, mask)
    vis_f  = flagDA(vis, tFlag, bFlag, cFlag, mask)
    vise_f = flagDA(viserr, tFlag, bFlag, cFlag, mask)
    fltr   = np.logical_not(ruv_f.mask | vis_f.mask | vise_f.mask)
    if filter_ruv:
        fltr_ruv = ruv_f > 1e-10
        fltr = fltr & fltr_ruv
    if np.sum(fltr) > 2:
        pass
    else:
        raise RuntimeError("There are not enough data!")
    x  = ruv_f[fltr] / 1e6
    y  = vis_f[fltr]
    ye = vise_f[fltr]
    yfit, c, ce = gaussfit.fit(x, y, ye, p0)
    inx = np.argsort(x)
    ruv_fit = x[inx] * 4.85e-3 # Units: mas
    vis_fit = yfit[inx]
    A, sigmax = c
    A_err, sigmax_err = ce
    return (ruv_fit, vis_fit), (A, A_err), (sigmax, sigmax_err)
