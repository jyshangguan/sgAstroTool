import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from astropy.visualization import simple_norm
from tqdm import tqdm
from scipy.optimize import minimize


N_TELESCOPE = 4
N_BASELINE = 6
N_TRIANGLE = 4


telescope_names = {
    'UT': ['U4', 'U3', 'U2', 'U1'],
    'AT': ['A4', 'A3', 'A2', 'A1'],
    'GV': ['G1', 'G2', 'G3', 'G4']
}


baseline_names = {
    'UT': ['U43', 'U42', 'U41', 'U32', 'U31', 'U21'],
    'AT': ['A43', 'A42', 'A41', 'A32', 'A31', 'A21'],
    'GV': ['G12', 'G13', 'G14', 'G23', 'G24', 'G34'],
}


triangle_names = {
    'UT': ['U432', 'U431', 'U421', 'U321'],
    'AT': ['A432', 'A431', 'A421', 'A321'],
    'GV': ['G123', 'G124', 'G134', 'G234'],
}


t2b_matrix = np.array([[1, -1, 0, 0],
                       [1, 0, -1, 0],
                       [1, 0, 0, -1],
                       [0, 1, -1, 0],
                       [0, 1, 0, -1],
                       [0, 0, 1, -1]])

lambda_met = 1.908  # micron


def phase_model(ra, dec, u, v):
    '''
    Calculate the phase model.

    Parameters
    ----------
    ra : float
        Right ascension offset in milliarcsec.
    dec : float
        Declination offset in milliarcsec.
    u : float or array
        U coordinate in Mlambda; (NDIT, NBASELINE, NCHANEL).
    v : float or array
        V coordinate in Mlambda; (NDIT, NBASELINE, NCHANEL).
    '''
    phase = 2 * np.pi * (np.pi / 3.6 / 180) * (u * ra + v * dec)
    return phase


def compute_metzp(oi1List, oi2List, ra, dec, pol=2, opd_lim=3000, step=0.1, zoom=30, plot=False, axs=None, verbose=True):
    '''
    Calculate the metrology zero point
    '''
    visref1 = []
    for oi in oi1List:
        assert oi._swap == False
        u, v = oi.get_vis_uvcoord(polarization=pol, units='Mlambda')
        phase = phase_model(ra, dec, u, v)
        visref = oi.get_visref(polarization=pol, per_dit=True)
        visref1.append(visref * np.exp(1j * phase))
    visref1 = np.mean(np.concatenate(visref1), axis=0)
    
    visref2 = []
    for oi in oi2List:
        assert oi._swap == True
        u, v = oi.get_vis_uvcoord(polarization=pol, units='Mlambda')
        phase = phase_model(-ra, -dec, u, v)
        visref = oi.get_visref(polarization=pol, per_dit=True)
        visref2.append(visref * np.exp(1j * phase))
    visref2 = np.mean(np.concatenate(visref2), axis=0)

    phi0 = np.angle(0.5 * (visref1 + visref2))
    
    # Prepare deriving the metrology zeropoint
    wave = oi.get_wavelength(polarization=pol)

    opdzp = fit_opd_closure(phi0, wave, opd_lim=opd_lim, step=step, zoom=zoom, 
                            plot=False, verbose=verbose)
    fczp = np.array([opdzp[2], opdzp[4], opdzp[5], 0])

    if plot:
        if axs is None:
            fig, axs = plt.subplots(6, 1, figsize=(8, 8), sharex=True)
            fig.subplots_adjust(hspace=0.02)
            axo = fig.add_subplot(111, frameon=False) # The out axis
            axo.tick_params(axis='y', which='both', left=False, labelleft=False)
            axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
            axo.set_xlabel(r'Wavelength ($\mu$m)', fontsize=18, labelpad=20)
            axo.set_ylabel(r'$\phi_0$ (rad)', fontsize=18, labelpad=50)
            axo.set_title(f'Metrology zero point in phase', fontsize=12, loc='left')

        else:
            assert len(axs) == 6, 'The number of axes must be 6!'

        for bsl, ax in enumerate(axs):
            ax.plot(wave, phi0[bsl, :], color=f'C{bsl}')
            ax.plot(wave, np.angle(matrix_opd(opdzp[bsl], wave)), color='gray', alpha=0.5, 
                    label='OPD model')
            ax.text(0.95, 0.9, f'{opdzp[bsl]:.2f} $\mu$m', fontsize=14, color='k', 
                    transform=ax.transAxes, va='top', ha='right', 
                    bbox=dict(facecolor='w', edgecolor='w', alpha=0.7))

            ax.minorticks_on()
            ax.text(0.02, 0.9, oi._baseline[bsl], transform=ax.transAxes, fontsize=14, 
                    va='top', ha='left', color=f'C{bsl}', fontweight='bold',
                    bbox=dict(facecolor='w', edgecolor='w', alpha=0.7))
        axs[0].legend(fontsize=14, loc='lower right', handlelength=1, 
                      bbox_to_anchor=(1, 1))
        
    return phi0, opdzp, fczp


def matrix_opd(opd, wave):
    '''
    Calculate the phase 
    '''
    opd = np.atleast_1d(opd)
    return np.exp(2j * np.pi * opd[None, :] / wave[:, None])


def opd_model(l1, l2, l3, l4, wave):
    '''
    Compute the OPD model for the given baseline configuration and wavelength.

    Parameters
    ----------
    l1, l2, l3, l4 : float
        The opd for each telescope in micron.
        Note that we adopt l1 to l4 corresponds to UT4 to UT1.
    wave : array
        The wavelength in micron.

    Returns
    -------
    v : array
        The complex array with the phase corresponds to the OPD model.
    '''
    v = np.exp(2j * np.pi / wave[None, :] * np.dot(t2b_matrix, np.array([l1, l2, l3, l4]))[:, None])
    return v


def lossfunc(l, phi0, wave): 
    '''
    Compute the loss function for the given baseline configuration and wavelength.
    '''
    l1, l2, l3 = l
    model = opd_model(l1, l2, l3, 0, wave)
    return np.sum(np.angle(np.exp(1j * phi0) * np.conj(model))**2)


def fit_zerofc(phi0, opd0, wave, opd_lim=1):
    '''
    Fit the zero frequency of the metrology zero point in phase.
    '''
    # l1, l2, l3 with the assumption that l4=0
    l_init = [opd0[2], opd0[4], opd0[5]]
    bounds = [(l-opd_lim, l+opd_lim) for l in l_init]
    res = minimize(lossfunc, l_init, args=(phi0, wave), bounds=bounds)

    if res.success:
        zerofc = np.array([res.x[0], res.x[1], res.x[2], 0])
    else:
        raise ValueError('The optimization is not successful!')

    return zerofc


def solve_offset(opd, uvcoord):
    '''
    Solve the astrometry offsets from the OPD and UV coordinates.

    Parameters
    ----------
    opd : np.ndarray
        The OPD values in micron, [NDIT, NBASELINE].
    uvcoord : np.ndarray
        The UV coordinates in meter, [NDIT, 2, NBASELINE].
    '''
    offset = []
    for dit in range(opd.shape[0]):
        uvcoord_pseudo_inverse = np.linalg.pinv(uvcoord[dit, :, :] * 1e6)
        offset.append(np.dot(opd[dit, :], uvcoord_pseudo_inverse)) 
    offset = np.array(offset) / np.pi * 180 * 3600 * 1000
    return offset


def grid_search(
            chi2_func : callable, 
            chi2_func_args : dict = {},
            ra_lim : float = 30, 
            dec_lim : float = 30, 
            nra : int = 100, 
            ndec : int = 100, 
            zoom : int = 5, 
            plot : bool = True, 
            axs : plt.axes = None, 
            percent : float = 99.5):
        '''
        Perform a grid search to find the best RA and Dec offsets.
        '''
        ra_grid = np.linspace(-ra_lim, ra_lim, nra)
        dec_grid = np.linspace(-dec_lim, dec_lim, ndec)
        chi2_grid = np.zeros((nra, ndec))

        chi2_best = np.inf
        # Add a description to the progress bar
        for i, ra in tqdm(list(enumerate(ra_grid)), desc='Initial grid search'):
            for j, dec in tqdm(list(enumerate(dec_grid)), leave=False):
                chi2_grid[i, j] = chi2_func(ra, dec, **chi2_func_args)[0]
    
                if chi2_grid[i, j] < chi2_best:
                    chi2_best = chi2_grid[i, j]
                    ra_best, dec_best = ra, dec

        ra_grid_zoom = ra_best + np.linspace(-ra_lim, ra_lim, nra) / zoom
        dec_grid_zoom = dec_best + np.linspace(-dec_lim, dec_lim, ndec) / zoom
        chi2_grid_zoom = np.zeros((len(dec_grid), len(ra_grid)))
        chi2_grid_bsl_zoom = np.zeros((len(dec_grid), len(ra_grid), 6))

        chi2_best_zoom = np.inf
        for i, ra in tqdm(list(enumerate(ra_grid_zoom)), desc='Zoomed grid search'):
            for j, dec in tqdm(list(enumerate(dec_grid_zoom)), leave=False):
                chi2_grid_zoom[i, j], chi2_grid_bsl_zoom[i, j] = chi2_func(ra, dec, **chi2_func_args)

                if chi2_grid_zoom[i, j] < chi2_best_zoom:
                    chi2_best_zoom = chi2_grid_zoom[i, j]
                    ra_best_zoom, dec_best_zoom = ra, dec
    
        if plot:
            # Plot both the full grid and the zoomed grid
            if axs is None:
                fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            else:
                assert len(axs) == 2, 'The number of axes must be 2!'

            ax = axs[0]
            norm = simple_norm(chi2_grid, stretch='linear', percent=percent)
            dra, ddec = np.diff(ra_grid)[0], np.diff(dec_grid)[0]
            extent = [ra_grid[0]-dra/2, ra_grid[-1]+dra/2, dec_grid[0]-ddec/2, dec_grid[-1]+ddec/2]
            im = ax.imshow(chi2_grid.T, origin='lower', norm=norm, extent=extent)
            rect = patches.Rectangle((ra_best-ra_lim/zoom, dec_best-dec_lim/zoom), 
                                     ra_lim/zoom*2, dec_lim/zoom*2, linewidth=1, 
                                     edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.set_xlabel(r'$\Delta$RA (mas)', fontsize=18)
            ax.set_ylabel(r'$\Delta$Dec (mas)', fontsize=18)
            ax.set_title(f'Full grid ({ra_best:.2f}, {dec_best:.2f})', fontsize=16)
            ax.plot(0, 0, marker='+', ls='none', color='C1', ms=15, label='Initial')
            ax.plot(ra_best, dec_best, marker='x', ls='none', color='C3', ms=15, label='Best-fit')
            ax.legend(loc='upper right', fontsize=14, frameon=True, framealpha=0.8, handlelength=1)
            ax.minorticks_on()

            ax = axs[1]
            norm = simple_norm(chi2_grid_zoom, stretch='linear', percent=percent)
            dra, ddec = np.diff(ra_grid_zoom)[0], np.diff(dec_grid_zoom)[0]
            extent = [ra_grid_zoom[0]-dra/2, ra_grid_zoom[-1]+dra/2, dec_grid_zoom[0]-ddec/2, dec_grid_zoom[-1]+ddec/2]
            im = ax.imshow(chi2_grid_zoom.T, origin='lower', norm=norm, extent=extent)
            ax.set_xlabel(r'$\Delta$RA (mas)', fontsize=18)
            ax.set_title(f'Zoomed grid ({ra_best_zoom:.2f}, {dec_best_zoom:.2f})', fontsize=16)
            ax.plot(0, 0, marker='+', ls='none', color='C1', ms=15, label='Initial')
            ax.plot(ra_best_zoom, dec_best_zoom, marker='x', ls='none', color='C3', ms=15, label='Best-fit')
            ax.minorticks_on()

        results = dict(ra_best=ra_best,
                       dec_best=dec_best,
                       chi2_best=chi2_best,
                       ra_best_zoom=ra_best_zoom, 
                       dec_best_zoom=dec_best_zoom, 
                       chi2_best_zoom=chi2_best_zoom, 
                       chi2_grid=chi2_grid,
                       chi2_grid_zoom=chi2_grid_zoom, 
                       chi2_grid_bsl_zoom=chi2_grid_bsl_zoom,
                       axs=axs,
                       ra_lim=ra_lim,
                       dec_lim=dec_lim,
                       nra=nra,
                       ndec=ndec,
                       zoom=zoom)

        return results


def compute_gdelay(
        visdata : np.array, 
        wave : np.array, 
        max_width : float = 2000,
        closure : bool = True,
        closure_lim : float = 1.2,
        verbose : bool = True,
        logger : logging.Logger = None):
    '''
    Compute the group delay from the visdata. 
    Same method as GRAVITY pipeline, except that I maximize the real part of 
    the visibility in the last pass.

    Parameters
    ----------
    visdata : np.array
        The visibility data, [NDIT, NBASELINE, NCHANNEL].
    wave : np.array
        The wavelength in micron, [NCHANNEL].
    max_width : float
        The maximum width of the OPD in micron.

    Returns
    -------
    gd : np.array
        The group delay in micron, [NDIT, NBASELINE].
    '''
    lbd = wave.mean()
    sigma = 1 / wave
    coherence = 0.5 * len(sigma) / np.abs(sigma[0] - sigma[-1])

    # First pass; less than max_width   
    width1 = np.min([coherence, max_width])
    step1 = 1 * lbd
    nstep1 = int(width1 / step1)
    opd1 = np.linspace(-width1 / 2, width1 / 2, nstep1)
    waveform1 = np.exp(-2j * np.pi * sigma[:, np.newaxis] * opd1[np.newaxis, :])

    # Second pass; less than 6 * nstep1
    width2 = 6 * step1
    step2 = 0.1 * lbd
    nstep2 = int(width2 / step2)
    opd2 = np.linspace(-width2 / 2, width2 / 2, nstep2)
    waveform2 = np.exp(-2j * np.pi * sigma[:, np.newaxis] * opd2[np.newaxis, :])

    # Third pass; less than 6 * nstep2
    width3 = 6 * step2
    step3 = 0.01 * lbd
    nstep3 = int(width3 / step3)
    opd3 = np.linspace(-width3 / 2, width3 / 2, nstep3)
    waveform3 = np.exp(-2j * np.pi * sigma[:, np.newaxis] * opd3[np.newaxis, :])

    # Compute the group delay for 0th order
    ds = np.mean(np.diff(sigma))
    dphi = np.ma.angle(visdata[:, :, 1:] * np.conj(visdata[:, :, :-1]))
    gd0 = np.ma.mean(dphi, axis=-1) / (2 * np.pi * ds)
    visdata_zero = visdata * np.exp(-2j * np.pi * sigma[np.newaxis, np.newaxis, :] * gd0[:, :, np.newaxis])

    # Find the higher order group delays
    opdList = [opd1, opd2]
    wfList = [waveform1, waveform2]
    gdList = [gd0]

    for opd, wf in zip(opdList, wfList):
        amp = np.abs(np.ma.dot(visdata_zero, wf))
        gd = opd[np.ma.argmax(amp, axis=-1)]
        gdList.append(gd)
        visdata_zero = visdata_zero * np.exp(-2j * np.pi * sigma[np.newaxis, np.newaxis, :] * gd[:, :, np.newaxis])
    
    # Last pass; maximize the real part
    amp = np.real(np.ma.dot(visdata_zero, waveform3))
    gd = opd3[np.ma.argmax(amp, axis=-1)]
    gdList.append(gd)

    gd = np.ma.sum(gdList, axis=0)

    if closure:
        if logger is not None:
            logger.info('Fitting for the closure...')
        elif verbose:
            print('Fitting for the closure...')

        visphi = np.angle(visdata)
        for dit in range(gd.shape[0]):
            try:
                zerofc = fit_zerofc(visphi[dit, :, :], gd[dit, :], wave, opd_lim=closure_lim)
                gd[dit, :] = np.dot(t2b_matrix, zerofc)
            except ValueError:
                if logger is not None:
                    logger.warning(f'Cannot find a closed solution within {closure_lim} fringe. Return the initial OPD results!')
                elif verbose:
                    print(f'Cannot find a closed solution within {closure_lim} fringe. Return the initial OPD results!')

    return gd


def gdelay_astrometry(
        oi1, 
        oi2, 
        polarization : int = None,
        average : bool = True,
        max_width : float = 2000,
        closure : bool = True,
        closure_lim : float = 1.2,
        plot : bool = False,
        ax : plt.axes = None):
    '''
    Compute the astrometry with a pair of swap data using the group delay method.
    '''
    wave = oi1._wave
    visphi = oi1.diff_visphi(oi2, polarization=polarization, average=average)
    uvcoord = (np.array(oi1._uvcoord_m)[:, :, :, 0] + 
               np.array(oi2._uvcoord_m)[:, :, :, 0]).swapaxes(0, 1)

    if average:
        uvcoord = uvcoord.mean(axis=0, keepdims=True)

    gd = compute_gdelay(np.exp(1j*visphi), wave, max_width=max_width, 
                        closure=closure, closure_lim=closure_lim, 
                        verbose=False)
    offset = solve_offset(gd, uvcoord)

    if plot:
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            plain = False
        else:
            plain = True

        for dit in range(visphi.shape[0]):
            ruv = oi1.get_uvdist(units='permas')[dit, :, :]

            s_B = np.pi / 180 / 3.6 * np.dot(offset[dit, :], uvcoord[dit, :, :])[:, np.newaxis]
            model = np.angle(np.exp(1j * 2*np.pi * s_B / wave[np.newaxis, :]))
            for bsl in range(visphi.shape[1]):
                if dit == 0:
                    l1 = oi1._baseline[bsl]
                else:
                    l1 = None
                ax.plot(ruv[bsl, :], visphi[dit, bsl, :], ls='-', color=f'C{bsl}', label=l1)

            for bsl in range(visphi.shape[1]):
                if (dit == 0) & (bsl == 0):
                    l2 = 'Gdelay'
                    l3 = r'$\vec{s} \cdot \vec{B}$'
                else:
                    l2 = None
                    l3 = None
                ax.plot(ruv[bsl, :], np.angle(np.exp(2j * np.pi * gd[dit, bsl] / wave)), 
                        ls='-', color=f'gray', label=l2)
                ax.plot(ruv[bsl, :], model[bsl, :], ls='-', color=f'k', label=l3)
        
        if not plain:
            ax.legend(loc='best', fontsize=14, handlelength=1, columnspacing=1, ncols=3)
            ax.set_ylim([-np.pi, np.pi])
            ax.set_xlabel(r'UV distance (mas$^{-1}$)', fontsize=18)
            ax.set_ylabel(r'VISPHI (rad)', fontsize=18)
            ax.minorticks_on()

    if average & (len(offset.shape) > 1):
        offset = offset[0, :]

    return offset


class GraviFits(object):
    '''
    The general class to read and plot GRAVITY OIFITS data.
    '''
    def __init__(self, filename, ignore_flag=False):
        '''
        Parameters
        ----------
        filename : str
            The name of the OIFITS file.
        ignore_flag : bool, optional
            If True, the FLAG and REJECTION_FLAG will be ignored. Default is False.
        '''
        self._filename = filename
        header = fits.getheader(filename, 0)
        self._header = header
        self._ignore_flag = ignore_flag

        self._arcfile = header.get('ARCFILE', None)
        if self._arcfile is not None:
            self._arctime = self._arcfile.split('.')[1]
        else:
            self._arctime = None

        self._object = header.get('OBJECT', None)

        # Instrument mode
        self._res = header.get('ESO INS SPEC RES', None)
        self._pol = header.get('ESO INS POLA MODE', None)
        if self._pol == 'SPLIT':
            self._pol_list = [1, 2]
        elif self._pol == 'COMBINED':
            self._pol_list = [0]
        else:
            self._pol_list = None
        self._npol = len(self._pol_list)

        # Science target
        self._sobj_x = header.get('ESO INS SOBJ X', None)
        self._sobj_y = header.get('ESO INS SOBJ Y', None)
        self._sobj_offx = header.get('ESO INS SOBJ OFFX', None)
        self._sobj_offy = header.get('ESO INS SOBJ OFFY', None)
        self._swap = header.get('ESO INS SOBJ SWAP', None) == 'YES'
        self._dit = header.get('ESO DET2 SEQ1 DIT', None)
        self._ndit = header.get('ESO DET2 NDIT', None)

        telescop = header.get('TELESCOP', None)
        if 'U1234' in telescop:
            self.set_telescope('UT')
        elif 'A1234' in telescop:
            self.set_telescope('AT')
        else:
            self.set_telescope('GV')

    
    def copy(self):
        '''
        Copy the current AstroFits object.
        '''
        return deepcopy(self)
    

    def get_extver(self, fiber='SC', polarization=None):
        '''
        Get the extver of the OIFITS HDU. The first digit is the fiber type 
        (1 or 2), and the second digit is the polarization (0, 1, or 2).

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for 
            COMBINED and SPLIT, respectively.
        
        Returns
        -------
        extver : int
            The extver of the OIFITS HDU.
        '''
        assert fiber in ['SC', 'FT'], 'fiber must be SC or FT.'

        if polarization is None:
            if self._pol == 'SPLIT':
                polarization = 1
            else:
                polarization = 0
        assert polarization in [0, 1, 2], 'polarization must be 0, 1, or 2.'

        fiber_code = {'SC': 1, 'FT': 2}
        extver = int(f'{fiber_code[fiber]}{polarization}')
        return extver


    def get_vis(self, field, fiber='SC', polarization=None):
        '''
        Get data from the OI_VIS extension of the OIFITS HDU.

        Parameters
        ----------
        fiber : str
            The fiber type. Either SC or FT.
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for 
            COMBINED and SPLIT, respectively.
        
        Returns
        -------
        data : masked array
            The data of the OI_VIS HDU. The shape is (NDIT, N_BASELINE, N_CHANNEL).
        '''
        extver = self.get_extver(fiber, polarization)
        data = fits.getdata(self._filename, 'OI_VIS', extver=extver)

        # Get the data
        field_data = data[field]
        if len(field_data.shape) == 2:
            nchn = field_data.shape[1]

            # Get the flag
            if self._ignore_flag:
                flag = None
            else:
                try:
                    flag = data['FLAG']
                    try:
                        rejflag  = data['REJECTION_FLAG'][:, np.newaxis]
                    except KeyError:
                        rejflag = np.zeros_like(flag)
                    flag = flag | ((rejflag & 19) > 0)
                except KeyError:
                    flag = None

        elif len(field_data.shape) < 3:
            # Supporting data, so flag is not needed
            flag = None  #np.zeros_like(field_data, dtype=bool)
            nchn = 1

        else:
            raise ValueError(f'The data has an incorrect shape (data[field].shape)!')

        field_data = np.ma.array(field_data, mask=flag).reshape(-1, N_BASELINE, nchn)
        return field_data
    

    def get_wavelength(self, fiber='SC', units='micron'):
        '''
        Get the wavelength of the OIFITS HDU.
        '''
        assert units in ['micron', 'm'], 'units must be micron or m.'

        extver = self.get_extver(fiber)
        wave = fits.getdata(self._filename, 'OI_WAVELENGTH', extver=extver)['EFF_WAVE']

        if units == 'micron':
            wave = wave * 1e6

        return wave
    

    def set_telescope(self, telescope):
        '''
        Set the telescope input of GRAVITY. The telescope, baseline, and 
        triangle names are adjusted accordingly.

        Parameters
        ----------
        telescope : str
            The telescope. Either UT, AT, or GV.
        '''
        assert telescope in ['UT', 'AT', 'GV'], 'telescope must be UT, AT, or GV.'
        self._telescope = telescope_names[telescope]
        self._baseline = baseline_names[telescope]
        self._triangle = triangle_names[telescope]


class SciVisFits(GraviFits):
    '''
    '''
    # Revised
    def __init__(self, filename, ignore_flag=False, normalize=True):
        super().__init__(filename, ignore_flag)

        self._wave = self.get_wavelength(units='micron')

        # Set the uv coordinates
        self._uvcoord_m = self.get_uvcoord_vis(units='m')
        self._uvcoord_permas = self.get_uvcoord_vis(units='permas')
        self._uvcoord_Mlambda = self.get_uvcoord_vis(units='Mlambda')

        # Visibility data
        for p in self._pol_list:
            extver = self.get_extver(fiber='SC', polarization=p)

            visdata = self.get_visdata(fiber='SC', polarization=p, 
                                       normalize=normalize)
            setattr(self, f'_visdata_{extver}', visdata)

            visamp = self.get_vis('VISAMP', fiber='SC', polarization=p)
            setattr(self, f'_visamp_{extver}', visamp)

            visphi = self.get_vis('VISPHI', fiber='SC', polarization=p)
            setattr(self, f'_visphi_{extver}', visphi)
            
            extver = self.get_extver(fiber='FT', polarization=p)

            visdata = self.get_visdata(fiber='FT', polarization=p, 
                                       normalize=normalize)
            setattr(self, f'_visdata_{extver}', visdata)

            visamp = self.get_vis('VISAMP', fiber='FT', polarization=p)
            setattr(self, f'_visamp_{extver}', visdata)

            visphi = self.get_vis('VISPHI', fiber='FT', polarization=p)
            setattr(self, f'_visphi_{extver}', visphi)


    def chi2_phase(
            self, 
            ra : float, 
            dec : float, 
            polarization=None):
        '''
        Calculate the chi2 to search for source offset with the phase method.
        '''
        gamma = []
        gooddata = []

        if polarization is None:
            pols = self._pol_list
        else:
            pols = [polarization]

        for p in pols:
            u, v = self._uvcoord_Mlambda
            phase = phase_model(ra, dec, u, v)
            model = np.exp(1j * phase)
            visdata = getattr(self, f'_visdata_1{p}')
            gamma.append(np.conj(model) * visdata)
            gooddata.append(visdata.mask == False)

        gamma = np.ma.sum(np.concatenate(gamma), axis=0) / np.sum(np.concatenate(gooddata), axis=0)
        chi2 = np.ma.sum(gamma.imag**2)
        chi2_baseline = np.ma.sum(gamma.imag**2, axis=1)
        return chi2, chi2_baseline


    def correct_visdata(
            self, 
            polarization : int = None,
            met_jump : list = None, 
            opdzp : list = None):
        '''
        Correct the metrology phase jump.

        Parameters
        ----------
        met_jump : list
            Number of fringe jumps to be added, telescope quantity, [UT4, UT3, UT2, UT1].
        opdzp : list
            OPD zeropoint, baseline quantity, [UT43, UT42, UT41, UT32, UT31, UT21].
        '''
        assert ~((met_jump is None) & (opdzp is None)), 'met_jump and opdzp cannot be None at the same time!'
        extver = self.get_extver(fiber='SC', polarization=polarization)
        visdata = getattr(self, f'_visdata_{extver}')

        if met_jump is not None:
            corr_tel = np.array(met_jump)[:, np.newaxis] * 2*np.pi * (1 - lambda_met / self._wave)[np.newaxis, :]
            opdDisp_corr = np.dot(t2b_matrix, corr_tel)
            visdata *= np.exp(1j * opdDisp_corr)
            setattr(self, f'_visphi_{extver}', np.angle(visdata, deg=True))

        if opdzp is not None:
            v_opd = np.exp(2j * np.pi * np.array(opdzp)[:, np.newaxis] / self._wave[np.newaxis, :])
            visdata *= np.conj(v_opd)[np.newaxis, :, :]
            setattr(self, f'_visphi_{extver}', np.angle(visdata, deg=True))


    def diff_visphi(self, 
                    oi : 'AstroFits', 
                    polarization : int = None,
                    average : bool = False):
        '''
        Calculate the difference of the VISPHI between the VISDATA of 
        the current and input AstroFits file.
        '''
        extver = self.get_extver(fiber="SC", polarization=polarization)
        vd1 = getattr(self, f'_visdata_{extver}')
        vd2 = getattr(oi, f'_visdata_{extver}')

        if average:
            vd1 = np.mean(vd1, axis=0, keepdims=True)
            vd2 = np.mean(vd2, axis=0, keepdims=True)

        visphi = np.angle(vd1 * np.conj(vd2))
        return visphi


    # Revised
    def get_visdata(
            self, 
            fiber : str = 'SC', 
            polarization : int = None, 
            per_exp : bool = False, 
            normalize : bool = True):
        '''
        Get the VISDATA of the SCIVIS data.

        Parameters
        ----------
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for
            COMBINED and SPLIT, respectively.
        
        Returns
        -------
        visdata : masked arrays
            The VISDATA of the OIFITS HDU. The shape is (NDIT, NBASELINE, NCHANNEL).
        '''
        assert ~(per_exp & normalize), 'per_exp and normalize cannot be True at the same time.'

        if normalize:
            visamp = self.get_vis('VISAMP', fiber=fiber, polarization=polarization)
            visphi = self.get_vis('VISPHI', fiber=fiber, polarization=polarization)
            visdata = visamp * np.exp(1j * np.deg2rad(visphi))
            return visdata

        visdata = self.get_vis('VISDATA', fiber=fiber, polarization=polarization)

        if per_exp:
            if visdata.shape[0] == 1:
                visdata /= self._dit * self._ndit
            elif visdata.shape[0] == self._ndit:
                visdata /= self._dit
            else:
                raise ValueError('Unclear how to calculate the exposure time!')

        return visdata


    def get_uvcoord_vis(
            self, 
            fiber : str = 'SC', 
            polarization : int = None, 
            units : str = 'm'):
        '''
        Get the u and v coordinates of the baselines.

        Parameters
        ----------
        polarization : int, optional
            The polarization. If None, the polarization 0 and 1 will be used for 
            COMBINED and SPLIT, respectively.
        units : str, optional
            The units of the uv coordinates. Either Mlambda, permas, or m.
            Mlambda is million lambda, permas is per milliarcsecond, and m is meter.
        
        Returns
        -------
        ucoord, vcoord : arrays
            The uv coordinate of the baselines, (NDIT, NBASELINE, NCHANNEL).
        '''
        assert units in ['Mlambda', 'permas', 'm'], 'units must be Mlambda, per mass, or m.'

        ucoord = self.get_vis('UCOORD', fiber=fiber, polarization=polarization)
        vcoord = self.get_vis('VCOORD', fiber=fiber, polarization=polarization)
        
        if units != 'm':
            wave = self.get_wavelength(units='micron')
            ucoord = ucoord / wave[np.newaxis, np.newaxis, :]
            vcoord = vcoord / wave[np.newaxis, np.newaxis, :]
            
            if units == 'permas':
                ucoord = ucoord * np.pi / 180. / 3600 * 1e3
                vcoord = vcoord * np.pi / 180. / 3600 * 1e3

        return ucoord, vcoord


    def get_uvdist(
            self, 
            units : str = 'Mlambda'):
        '''
        Get the uv distance of the baselines.

        Parameters
        ----------
        units : str, optional
            The units of the uv distance. Either Mlambda, permas, or m.
            Mlambda is million lambda, permas is per milliarcsecond, and m is meter.
        
        Returns
        -------
        uvdist : array
            The uv distance of the baselines, (NDIT, NBASELINE, NCHANNEL).
        '''
        assert units in ['Mlambda', 'permas', 'm'], 'units must be Mlambda, per mass, or m.'

        ucoord, vcoord = getattr(self, f'_uvcoord_{units}')
        uvdist = np.sqrt(ucoord**2 + vcoord**2)

        return uvdist


    def grid_search_phase(
            self, 
            polarization=None, 
            plot=False, 
            **kwargs):
        '''
        Perform a grid search to find the best RA and Dec offsets.

        WARNING
        -------
        This method seems to be less robust than the cal_offset_opd() method for 
        individual files.
        '''
        res = grid_search(self.chi2_phase, 
                          chi2_func_args=dict(polarization=polarization), 
                          plot=plot, **kwargs)
        
        if plot:
            if self._swap:
                ra_total = self._sobj_x - res['ra_best']
                dec_total = self._sobj_y - res['dec_best']
            else:
                ra_total = self._sobj_x + res['ra_best']
                dec_total = self._sobj_y + res['dec_best']

            ax1, ax2 = res['axs']
            ax1.text(0.05, 0.95, f'Swap: {self._swap}', fontsize=12, 
                     transform=ax1.transAxes, va='top', ha='left',
                     bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
            text = '\n'.join([f'SOBJ_XY: ({self._sobj_x:.2f}, {self._sobj_y:.2f})', 
                    f'Measured: ({ra_total:.2f}, {dec_total:.2f})'])
            ax2.text(0.05, 0.95, text, fontsize=12, transform=ax2.transAxes, va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        return res


    # Revised
    def plot_visphi(self, fiber='SC', polarization=None, use_visdata=False, ax=None, plain=False):
        '''
        Plot the angle of the VISDATA.
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        extver = self.get_extver(fiber=fiber, polarization=polarization)

        if use_visdata:
            visdata = getattr(self, f'_visdata_{extver}')
            visphi = np.angle(visdata[dit, bsl, :], deg=True)
        else:
            visphi = getattr(self, f'_visphi_{extver}')

        ruv = self.get_uvdist(units='Mlambda')

        for dit in range(visphi.shape[0]):
            for bsl in range(visphi.shape[1]):
                ax.plot(ruv[dit, bsl, :], visphi[dit, bsl, :], color=f'C{bsl}')
        
        if not plain:
            ax.set_ylim([-180, 180])
            ax.set_xlabel('uv distance (Mlambda)', fontsize=18)
            ax.set_ylabel(r'VISPHI ($^\circ$)', fontsize=18)
            ax.minorticks_on()
        return ax
    

    # Revised
    def plot_visamp(self, fiber='SC', polarization=None, use_visdata=False, ax=None, plain=False):
        '''
        Plot the amplitude of the VISDATA
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))

        extver = self.get_extver(fiber=fiber, polarization=polarization)

        if use_visdata:
            visdata = getattr(self, f'_visdata_{extver}')
            visamp = np.absolute(visdata[dit, bsl, :])
        else:
            visamp = getattr(self, f'_visamp_{extver}')

        ruv = self.get_uvdist(units='Mlambda')
        
        for dit in range(visamp.shape[0]):
            for bsl in range(visamp.shape[1]):
                ax.plot(ruv[dit, bsl, :], visamp[dit, bsl, :], color=f'C{bsl}')
        
        if not plain:
            ax.set_ylim([0, None])
            ax.set_xlabel('uv distance (Mlambda)', fontsize=18)
            ax.set_ylabel(r'VISAMP', fontsize=18)
            ax.minorticks_on()
        return ax


class GraviList(object):
    '''
    A list of GRAVITY OIFITS object.
    '''
    def __init__(self, name='GraviList', log_name=None, verbose=True):

        self._name = name

        self._logger = logging.getLogger(self._name)
        self._logger.setLevel(logging.INFO)  # Set the logging level
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Avoid adding handlers multiple times
        if not self._logger.hasHandlers():
            # Create a file handler
            file_handler = logging.FileHandler(f'{log_name}.log', mode='w')
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self._logger.addHandler(file_handler)

            # Create a console handler
            if verbose:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(logging.INFO)
                console_handler.setFormatter(formatter)
                self._logger.addHandler(console_handler)


    def __add__(self, other : 'GraviList'):
        '''
        Add two AstroList objects.
        '''
        self._datalist = self._datalist + other._datalist
    

    def __getitem__(self, index):
        '''
        Parameters
        ----------
        index : int or list
        '''
        if isinstance(index, list):
            return [self._datalist[i] for i in index]
        elif isinstance(index, int):
            return self._datalist[index]


    def __setitem__(self, index, value):
        self._datalist[index] = value


    def __len__(self):
        return len(self._datalist)


    def __iter__(self):
        return iter(self._datalist)
    

    def __repr__(self):
        return f'GraviList with {len(self._datalist)} files'
    

    def __str__(self):
        return f'GraviList with {len(self._datalist)} files'


    def append(self, oi):
        '''
        Append an AstroFits object to the list.
        '''
        self._datalist.append(oi)

    
    def copy(self):
        '''
        Copy the current AstroList object.
        '''
        return deepcopy(self)


class SciVisList(GraviList):
    '''
    A list of SciVisFits objects.
    '''
    def __init__(
            self, 
            files : list, 
            ignore_flag : bool = False, 
            normalize : bool = False,
            verbose=True,
            log_name=None) -> None:
        '''
        Parameters
        ----------

        '''
        super().__init__(name='SciVisList', log_name=log_name, verbose=verbose)
        
        self._datalist = []
        self._index_unswap = []
        self._index_swap = []
        for i, f in enumerate(files):
            self._logger.info(f'Processing {f}')

            oi = SciVisFits(f, ignore_flag=ignore_flag, normalize=normalize)
            self._datalist.append(oi)

            if oi._swap:
                self._index_swap.append(i)
            else:
                self._index_unswap.append(i)

        self._pol_list = self._datalist[0]._pol_list
        self._sobj_x = self._datalist[0]._sobj_x
        self._sobj_y = self._datalist[0]._sobj_y


    def chi2_phase(self, ra, dec):
        '''
        Parameters
        ----------
        ra : float
            Right ascension offset in milliarcsec.
        dec : float
            Declination offset in milliarcsec.
        '''
        gamma1 = []
        gooddata1 = []
        for oi in self[self._index_unswap]:
            for p in self._pol_list:
                u, v = oi._uvcoord_Mlambda
                phase = phase_model(ra, dec, u, v)
                model = np.exp(1j * phase)
                visdata = getattr(oi, f'_visdata_1{p}')
                gamma1.append(np.conj(model) * visdata)
                gooddata1.append(visdata.mask == False)
        
        gamma2 = []
        gooddata2 = []
        for oi in self[self._index_swap]:
            for p in self._pol_list:
                u, v = oi._uvcoord_Mlambda
                phase = phase_model(ra, dec, u, v)
                model = np.exp(1j * phase)
                visdata = getattr(oi, f'_visdata_1{p}')
                gamma2.append(model * visdata)
                gooddata2.append(visdata.mask == False)

        gamma1 = np.ma.sum(np.concatenate(gamma1), axis=0) / np.sum(np.concatenate(gooddata1), axis=0)
        gamma2 = np.ma.sum(np.concatenate(gamma2), axis=0) / np.sum(np.concatenate(gooddata2), axis=0)
        gamma_swap = (np.conj(gamma1) * gamma2)**0.5  # Important not to use np.sqrt() here!
        chi2 = np.ma.sum(gamma_swap.imag**2)
        chi2_baseline = np.ma.sum(gamma_swap.imag**2, axis=1)
    
        return chi2, chi2_baseline


    def compute_metzp(
            self, 
            ra : float,
            dec : float,
            closure=True,
            closure_lim : float = 1.2,
            max_width : float = 2000,
            plot=False, 
            axs=None, 
            verbose=True,
            pdf=None):
        '''
        Calculate the metrology zero point
        '''
        visdata1 = []
        visdata2 = []
        for p in self._pol_list:
            # Unswapped data
            vd = []
            for oi in self[self._index_unswap]:
                u, v = oi._uvcoord_Mlambda
                phase = phase_model(-ra, -dec, u, v)
                vd.append(getattr(oi, f'_visdata_1{p}') * np.exp(1j * phase))
            visdata1.append(vd)
    
            vd = []
            for oi in self[self._index_swap]:
                u, v = oi._uvcoord_Mlambda
                phase = phase_model(ra, dec, u, v)
                vd.append(getattr(oi, f'_visdata_1{p}') * np.exp(1j * phase))
            visdata2.append(vd)

        visdata = 0.5 * (np.mean(visdata1, axis=(1, 2)) + np.mean(visdata2, axis=(1, 2)))
        phi0 = np.angle(visdata)
    
        npol = len(self._pol_list)
        wave = oi._wave
        baseline_name = oi._baseline

        # Prepare deriving the metrology zeropoint; [NPOL, NBASELINE]
        self._opdzp = compute_gdelay(visdata, wave, max_width=max_width, 
                                     closure=closure, closure_lim=closure_lim, 
                                     verbose=verbose, logger=self._logger)
        self._fczp = np.array([self._opdzp[:, 2], 
                               self._opdzp[:, 4], 
                               self._opdzp[:, 5], 
                               np.zeros(npol)])

        if plot:
            if axs is None:
                fig, axs = plt.subplots(6, npol, figsize=(8*npol, 12), sharex=True)
                fig.suptitle(f'Metrology zero point in phase', fontsize=14)
                fig.subplots_adjust(hspace=0.02)
                axo = fig.add_subplot(111, frameon=False) # The out axis
                axo.tick_params(axis='y', which='both', left=False, labelleft=False)
                axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                axo.set_xlabel(r'Wavelength ($\mu$m)', fontsize=18, labelpad=20)
                axo.set_ylabel(r'$\phi_0$ (rad)', fontsize=18, labelpad=50)

            else:
                assert len(axs) == 6, 'The number of axes must be 6!'

            for i, p in enumerate(self._pol_list):
                for bsl, ax in enumerate(axs[:, i]):
                    ax.plot(wave, phi0[i, bsl, :], color=f'C{bsl}')
                    ax.plot(wave, np.angle(matrix_opd(self._opdzp[i, bsl], wave)), color='gray', alpha=0.5, 
                            label='OPD model')
                    ax.text(0.95, 0.9, f'{self._opdzp[i, bsl]:.2f} $\mu$m', fontsize=14, color='k', 
                            transform=ax.transAxes, va='top', ha='right', 
                            bbox=dict(facecolor='w', edgecolor='w', alpha=0.7))

                    ax.minorticks_on()
                    ax.text(0.02, 0.9, baseline_name[bsl], transform=ax.transAxes, fontsize=14, 
                            va='top', ha='left', color=f'C{bsl}', fontweight='bold',
                            bbox=dict(facecolor='w', edgecolor='w', alpha=0.7))
            ax.legend(loc='lower left', fontsize=14)

            if pdf is not None:
                pdf.savefig(fig)
                plt.close(fig)


    def correct_metzp(self):
        '''
        Correct the metrology zero points.
        '''
        for oi in self:
            for i, p in enumerate(self._pol_list):
                oi.correct_visdata(polarization=p, opdzp=self._opdzp[i, :])


    def correct_met_jump(
            self, 
            index : int, 
            met_jump : list):
        '''
        Correct the metrology phase jump.
        '''
        for p in self._pol_list:
            self._datalist[index].correct_visdata(polarization=p, met_jump=met_jump)


    def grid_search_phase(self, plot=True, **kwargs):
        '''
        Perform a grid search to find the best RA and Dec offsets.
        '''
        res= grid_search(self.chi2_phase, plot=plot, **kwargs)

        if plot:
            ra_total = self._sobj_x + res['ra_best']
            dec_total = self._sobj_y + res['dec_best']
            ax = res['axs'][1]
            text = '\n'.join([f'SOBJ_XY: ({self._sobj_x:.2f}, {self._sobj_y:.2f})', 
                    f'Total: ({ra_total:.2f}, {dec_total:.2f})'])
            ax.text(0.05, 0.95, text, fontsize=12, transform=ax.transAxes, va='top', ha='left',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        return res


    def gdelay_astrometry(self, max_width=2000, plot=False, axs=None, pdf=None):
        '''
        Perform the group delay astrometry.
        '''
        pair_index = np.array(np.meshgrid(self._index_unswap, self._index_swap)).T.reshape(-1, 2)

        offsetList = []
        for loop, (i, j) in enumerate(pair_index):
            offset = []

            if plot:
                npol = len(self._pol_list)
                if axs is None:
                    fig, axs_use = plt.subplots(1, npol, figsize=(6*npol, 6), sharex=True, sharey=True)
                    fig.suptitle(f'GD astrometry: {self[int(i)]._arcfile} and {self[int(j)]._arcfile}', fontsize=14)
                    fig.subplots_adjust(wspace=0.02)
                    axo = fig.add_subplot(111, frameon=False)
                    axo.tick_params(axis='y', which='both', left=False, labelleft=False)
                    axo.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
                    axo.set_xlabel(r'$uv$ distance (mas$^{-1}$)', fontsize=24, labelpad=20)
                    axo.set_ylabel(r'VISPHI ($^\circ$)', fontsize=24, labelpad=32)
                else:
                    axs_use = axs[loop, :]
            else:
                axs_use = None

            for loop, p in enumerate(self._pol_list):
                if axs_use is None:
                    ax = None
                else:
                    ax = np.atleast_1d(axs_use)[loop]
                    ax.axhline(y=0, ls='--', color='k')

                offset.append(gdelay_astrometry(self[int(i)], self[int(j)], polarization=p, 
                                                average=True, max_width=max_width, plot=plot, 
                                                ax=ax))

            if ax is not None:
                ax.minorticks_on()
                ax.set_ylim([-np.pi, np.pi])
                ax.legend(loc='best', ncols=3, handlelength=1, columnspacing=1, fontsize=14)
            
            if pdf is not None:
                pdf.savefig(fig)
                plt.close(fig)

            offsetList.append(offset)

        offsetList = np.array(offsetList).swapaxes(1, 2)  #[NDIT, XY, POLARIZATION]

        return offsetList


    def run_swap_astrometry(self, met_jump_dict=None, plot=True, report_name=None, verbose=True):
        '''
        The main function to measure astrometry and metrology zero point from 
        the swap data.
        '''
        assert len(self._index_unswap) > 0, 'There is no unswap data in the list!'
        assert len(self._index_swap) > 0, 'There is no swap data in the list!'

        if report_name is not None:
            from matplotlib.backends.backend_pdf import PdfPages

            pdf = PdfPages(report_name)
        else:
            pdf = None


        # Correct the metrology phase jump
        if met_jump_dict is not None:
            self._logger.info('Correct the metrology phase jump')
            for i, met_jump in met_jump_dict.items():
                self.correct_met_jump(i, met_jump)


        # Plot data
        if plot:
            self._logger.info('Plotting the data')
            for oi in self:
                fig, axs = plt.subplots(oi._npol, 2, figsize=(14, 7*oi._npol))
                axs = np.array(axs).reshape(-1, 2)
                fig.suptitle(f'Original data: {oi._arcfile}', fontsize=18)

                for i, p in enumerate(self._pol_list):
                    oi.plot_visamp(polarization=p, ax=axs[i, 0])
                    oi.plot_visphi(polarization=p, ax=axs[i, 1])

                if pdf is not None:
                    pdf.savefig(fig)
                    plt.close(fig)
        

        # Search for astrometry using the phase method
        self._logger.info('Grid search for astrometry solution')
        res = self.grid_search_phase(plot=False)


        # Measure metrology zero point
        self._logger.info('Measure metrology zero point')
        self.compute_metzp(res['ra_best'], res['dec_best'], plot=plot, pdf=pdf)


        # Correct the metrology zero point
        self._logger.info('Correct the metrology zeropoint')
        self.correct_metzp()


        # Plot the phase to evaluate the data correction
        self._logger.info('Plotting the metrology zeropoint corrected data')
        for oi in self:
            fig, axs = plt.subplots(1, oi._npol, figsize=(7*oi._npol, 7))
            axs = np.atleast_1d(axs)
            fig.suptitle(f'METZP corrected: {oi._arcfile}', fontsize=18)

            for i, p in enumerate(self._pol_list):
                oi.plot_visphi(polarization=p, ax=axs[i])
                axs[i].text(0.05, 0.95, f'P{p}', fontsize=16, 
                            transform=axs[i].transAxes, va='top', ha='left',
                            bbox=dict(facecolor='w', edgecolor='none', alpha=0.5))

            if pdf is not None:
                pdf.savefig(fig)
                plt.close(fig)


        # Plot the chi2 map, grid search solution, and group delay astrometry solutions
        self._logger.info('Final grid search for astrometry solution')

        if plot:
            fig, axs = plt.subplots(1, 3, figsize=(21, 7))
            fig.suptitle('Astrometry results', fontsize=18)

            axs_grid = axs[:2]
        else:
            axs_grid = None

        res = self.grid_search_phase(plot=plot, axs=axs_grid)
        self._ra_best = res['ra_best']
        self._dec_best = res['dec_best']
        self._sobj_x_fit = self._sobj_x + res['ra_best']
        self._sobj_y_fit = self._sobj_y + res['dec_best']

        
        # Calculate the group delay astrometry
        self._logger.info('Calculate the group delay astrometry')
        offsets_gd = self.gdelay_astrometry(plot=plot, pdf=pdf)

        if plot:
            ax = axs[2]
            colors = ['C0', 'C2']
            markers = ['+', 'x']
            for p in range(offsets_gd.shape[-1]):
                for i in range(offsets_gd.shape[0]):
                    if i == 0:
                        label = f'GD P{self._pol_list[p]}'
                    else:
                        label = None

                    ax.plot(offsets_gd[i, 0, p], offsets_gd[i, 1, p], color=colors[p], ls='none', 
                            marker=markers[p], ms=6, lw=2, label=label)

            ax.plot(res['ra_best_zoom'], res['dec_best_zoom'], marker='x', ms=15, color='C3')
            ax.legend(fontsize=16, loc='upper right', handlelength=1)
        
            # Expand the panel by 2 times for the current axis and make the aspect ratio equal
            ax = axs[2]
            xlim =  ax.get_xlim()
            ylim =  ax.get_ylim()
        
            alim = np.max([np.abs(np.diff(xlim)), np.abs(np.diff(ylim))]) * 0.8
            xcent = np.mean(xlim)
            ycent = np.mean(ylim)
            ax.set_xlim([xcent - alim, xcent + alim])
            ax.set_ylim([ycent - alim, ycent + alim])
            ax.set_aspect('equal')
            ax.set_title('Zoom in astrometry', fontsize=16)
            ax.minorticks_on()
            ax.set_xlabel(r'$\Delta$RA (mas)', fontsize=18)

            if pdf is not None:
                pdf.savefig(fig)
                plt.close(fig)

        if pdf is not None:
            pdf.close()
        
        self._logger.info('OPD_MET_ZERO_FC: ')
        for i, p in enumerate(self._pol_list):
            self._logger.info(', '.join([f'{v:.2f}' for v in self._fczp[:, i]]))

        self._logger.info("Pipeline completed!")


if __name__ == '__main__':
    print('The script is not finished!')
