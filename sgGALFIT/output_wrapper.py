import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.io import fits
from astropy.nddata import CCDData
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from .utils import *


__all__ = ['imgblock_standard', 'imgblock_subcomp']

class imgblock_standard(object):
    '''
    GALFIT standard output.  E.g., output from
        `galfit -o2 <file> (standard img. block)`
    '''
    def __init__(self, filename, zeromag=None, pixelscale=None, exptime=None, unit=None):
        '''
        Parameters
        ----------
        filename : string
            Path of the GALFIT output file.
        '''
        f = fits.open(filename)
        header = f[1].header
        self.model_info = get_model_from_header(header)
        self.zeromag = zeromag

        if unit is None:
            unit = convert_unit_hst2astropy(header.get('BUNIT'))
        self.unit = unit

        self.extensions = {
            'data': [CCDData.read(filename, hdu=1, unit=unit), None],
            'model': [CCDData.read(filename, hdu=2, unit=unit), None],
            'residual': [CCDData.read(filename, hdu=3, unit=unit), None]
        }
        if pixelscale is None:
            pixelscale = self.get_pixelscale_wcs(units='arcsec')
        self.pixelscale = pixelscale
        if exptime is None:
            exptime = header.get('EXPTIME', None)
        self.exptime = exptime
        self.isophotes = {}

    def cut_fov(self, position, size):
        '''
        Cut the field of view.

        Parameters
        ----------
        position : (x, y)
            Position of the image center, units: pixel.
        size : (dx, dy)
            Size of the new image, units: pixel.
        '''
        for loop, ext_name in enumerate(self.extensions):
            ext, tag = self.extensions[ext_name]
            img_cut = Cutout2D(ext.data, position=position, size=size)
            if ext.wcs is not None:
                header = ext.wcs.to_header()
                crpix_org = [header['CRPIX1'], header['CRPIX2']]
                crpix_new = img_cut.to_cutout_position(crpix_org)
                header['CRPIX1'] = crpix_new[0]
                header['CRPIX2'] = crpix_new[1]
                wcs = WCS(header)
            else:
                wcs = None

            if ext.mask is not None:
                mask = Cutout2D(ext.mask, position=position, size=size).data
            else:
                mask = None
            ccd_new = CCDData(img_cut.data, wcs=wcs, mask=mask, unit=ext.unit)
            self.extensions[ext_name] = [ccd_new, tag]

    def fit_ellipse(self, ext_name, x0=None, y0=None, sma=None, eps=0, pa=0,
                    expand=False, **kwargs):
        '''
        Fit isophotes of an extension.

        Parameters
        ----------
        ext_name : string
            Name of the extenstion to be fitted.
        x0, y0 : float
            The center pixel coordinate of the ellipse.
        sma : float
            The semimajor axis of the ellipse in pixels.
        eps : ellipticity
            The ellipticity of the ellipse.
        pa : float
            The position angle (in radians) of the semimajor axis in
            relation to the postive x axis of the image array (rotating
            towards the positive y axis). Position angles are defined in the
            range :math:`0 < PA <= \\pi`. Avoid using as starting position
            angle of 0., since the fit algorithm may not work properly. When
            the ellipses are such that position angles are near either
            extreme of the range, noise can make the solution jump back and
            forth between successive isophotes, by amounts close to 180
            degrees.
        **kwargs : Additional parameters feed to ellipse.fit_image().

        Returns
        -------
        isolist : IsophoteList instance
            A list-like object of Isophote instances, sorted by increasing
            semimajor axis length.
        '''
        ext = self.get_extension(ext_name)
        if ext.mask is None:
            image = ext.data
        else:
            image = np.ma.array(ext.data, mask=ext.mask)

        if x0 is None:
            x0 = image.shape[1] / 2
        if y0 is None:
            y0 = image.shape[0] / 2
        if sma is None:
            ## Estimate the size of the source
            #xx, yy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
            #xx = xx.astype(np.float64) - x0
            #yy = yy.astype(np.float64) - y0
            #rr = np.sqrt(xx**2 + yy**2)
            #sma = np.sum(rr * image) / np.sum(image)
            sma = 10

        isolist = fit_ellipse(image, x0, y0, sma, eps, pa, **kwargs)

        if expand is True:
            step = kwargs.get('step', 0.1)
            fflag = kwargs.get('fflag', 0.7)
            maxsma = kwargs.get('maxsma', None)
            isolist_exp = grow_isophote(image, isolist[-1], step=step, fflag=fflag,
                                        maxsma=maxsma, maxsteps=np.inf)
            isolist = isolist + isolist_exp

        self.isophotes[ext_name] = isolist
        return isolist

    def fit_isophote(self, ext_name, isolist):
        '''
        Fit isophotes according to the input isolist.

        Parameters
        ----------
        ext_name : string
            Name of the extenstion to be fitted.
        isolist : IsophoteList
            Input isophotes.

        Returns
        -------
        isolist_out : IsophoteList
            New measurements.
        '''
        ext = self.get_extension(ext_name)
        if ext.mask is None:
            image = ext.data
        else:
            image = np.ma.array(ext.data, mask=ext.mask)
        isolist_out = fit_isophote(image, isolist)
        self.isophotes[ext_name] = isolist_out
        return isolist_out

    def get_CRPIX(self):
        '''
        Get the reference pixel.
        '''
        header = self.extensions['data'][0].wcs.to_header()
        return (header['CRPIX1'], header['CRPIX2'])

    def get_extension(self, ext_name):
        '''
        Get the extension data.
        '''
        return self.extensions[ext_name][0]

    def get_extent(self, units='arcsec'):
        '''
        Get the extent for imshow based on the data wcs.

        Parameters
        ----------
        units : string (default: 'arcsec')
            Units of the pixel scale.
        '''
        data = self.get_extension('data')
        nrow, ncol = data.shape
        x_len = ncol * self.get_pixelscale(units)
        y_len = nrow * self.get_pixelscale(units)
        extent = (-x_len/2, x_len/2, -y_len/2, y_len/2)
        return extent

    def get_ImageCenter(self):
        '''
        Get the central pixel of the image.

        BE CAREFUL:
            For values exactly halfway between rounded decimal values, NumPy
            rounds to the nearest even value. Thus 1.5 and 2.5 round to 2.0,
            -0.5 and 0.5 round to 0.0, etc.
        '''
        image = self.get_extension('data').data
        xcent = np.around(image.shape[1]/2.0, decimals=0)
        ycent = np.around(image.shape[0]/2.0, decimals=0)
        return (xcent, ycent)

    def get_isolist(self, ext_name):
        '''
        Get isolist of an extension.
        '''
        return self.isophotes[ext_name]

    def get_mu(self, ext_name, pixelscale=None, exptime=None, zeromag=None):
        '''
        Get the surface brightness.
        '''
        isolist = self.isophotes.get(ext_name, None)
        if isolist is None:
            raise KeyError('Cannot find isolist for {0}!'.format(ext_name))
        if pixelscale is None:
            assert self.pixelscale is not None
            pixelscale = self.pixelscale
        if exptime is None:
            assert self.exptime is not None
            exptime = self.exptime
        if zeromag is None:
            assert self.zeromag is not None
            zeromag = self.zeromag

        x = isolist.sma * pixelscale
        y = isolist.intens / exptime / pixelscale**2
        e = isolist.int_err / exptime / pixelscale**2
        y, e = flux2mag(y, e, zeromag)
        return x, y, e

    def get_tag(self, ext_name):
        '''
        Get the extension tag.
        '''
        return self.extensions[ext_name][1]

    def get_pixelscale(self, units='arcsec'):
        '''
        Get pixel scale of the data image.
        '''
        ps_arcsec = self.pixelscale * u.arcsec
        ps_out = ps_arcsec.to(units).value
        return ps_out

    def get_pixelscale_wcs(self, units='arcsec'):
        '''
        Get pixel scale of the data image from WCS.
        '''
        cdelt1, cdelt2 = wcs_pixel_scale(self.get_extension('data').wcs)
        cdelt1 = cdelt1.to(units).value
        cdelt2 = cdelt2.to(units).value
        if not np.isclose(cdelt1, cdelt2, rtol=0.001):
            raise ValueError('The pixel scale of two axes are not the same!')
        ps = (cdelt1 + cdelt2) / 2
        return ps

    def grow_isophote(self, ext_name, isophote, step=0.1, fflag=0.7, maxsma=None,
                      maxsteps=1000):
        '''
        Fit the isophote of the extension starting from the input isophote and
        use its shape fixed.

        Parameters
        ----------
        ext_name : string
            The name of the extension.
        isophote : photutils.isophote.Isophote
            The isophote to be expanded.
        step : float (default: 0.1)
            The step value for growing/shrinking the semimajor axis.
        fflag : float (default: 0.7)
            The acceptable fraction of flagged data points in the
            sample.  If the actual fraction of valid data points is
            smaller than this, the iterations will stop.  Flagged
            data points are points that either lie outside the image
            frame, are masked, or were rejected by sigma-clipping.
        maxsma (optional) : float
            Maximum semimajor axis length, units: pixel.
        maxsteps : int (default: 1000)
            Maximum steps to grow the isophote.
        '''
        ext = self.get_extension(ext_name)
        if ext.mask is None:
            image = ext.data
        else:
            image = np.ma.array(ext.data, mask=ext.mask)
        isolist = grow_isophote(image, isophote, step=step, fflag=fflag,
                                maxsma=maxsma, maxsteps=maxsteps)
        return isolist

    def plot_direction(self, ax, xy, len_E=None, len_N=None, color='k', fontsize=20,
                       linewidth=2, frac_len=0.15, units='arcsec', backextend=0.05):
        '''
        Plot the direction arrow. Only applied to plots using WCS.

        Parameters
        ----------
        ax : Axis
            Axis to plot the direction.
        xy : (x, y)
            Coordinate of the origin of the arrows.
        length : float
            Length of the arrows, units: pixel.
        units: string (default: arcsec)
            Units of xy.
        '''
        xlim = ax.get_xlim()
        len_total = np.abs(xlim[1] - xlim[0])
        pixelscale = self.get_pixelscale(units)
        if len_E is None:
            len_E = len_total * frac_len / pixelscale
        if len_N is None:
            len_N = len_total * frac_len / pixelscale

        wcs = self.extensions['data'][0].wcs
        header = wcs.to_header()
        d_ra = len_E * self.get_pixelscale('degree')
        d_dec = len_N * self.get_pixelscale('degree')
        ra = [header['CRVAL1'], header['CRVAL1']+d_ra, header['CRVAL1']]
        dec = [header['CRVAL2'], header['CRVAL2'], header['CRVAL2']+d_dec]
        ra_pix, dec_pix = wcs.all_world2pix(ra, dec, 1)
        d_arrow1 = [ra_pix[1]-ra_pix[0], dec_pix[1]-dec_pix[0]]
        d_arrow2 = [ra_pix[2]-ra_pix[0], dec_pix[2]-dec_pix[0]]
        l_arrow1 = np.sqrt(d_arrow1[0]**2 + d_arrow1[1]**2)
        l_arrow2 = np.sqrt(d_arrow2[0]**2 + d_arrow2[1]**2)
        d_arrow1 = np.array(d_arrow1) / l_arrow1 * len_E * pixelscale
        d_arrow2 = np.array(d_arrow2) / l_arrow2 * len_N * pixelscale

        def sign_2_align(sign):
            '''
            Determine the alignment of the text.
            '''
            if sign[0] < 0:
                ha = 'right'
            else:
                ha = 'left'
            if sign[1] < 0:
                va = 'top'
            else:
                va = 'bottom'
            return ha, va
        ha1, va1 = sign_2_align(np.sign(d_arrow1))
        ha2, va2 = sign_2_align(np.sign(d_arrow2))

        xy_e = (xy[0] - d_arrow1[0] * backextend, xy[1] - d_arrow1[1] * backextend)
        ax.annotate('E', xy=xy_e, xycoords='data', fontsize=fontsize,
                    xytext=(d_arrow1[0]+xy[0], d_arrow1[1]+xy[1]), color=color,
                    arrowprops=dict(color=color, arrowstyle="<-", lw=linewidth),
                    ha=ha1, va=va1)
        xy_n = (xy[0] - d_arrow2[0] * backextend, xy[1] - d_arrow2[1] * backextend)
        ax.annotate('N', xy=xy_n, xycoords='data', fontsize=fontsize,
                    xytext=(d_arrow2[0]+xy[0], d_arrow2[1]+xy[1]), color=color,
                    arrowprops=dict(color=color, arrowstyle="<-", lw=linewidth),
                    ha=ha2, va=va2)

    def plot_extension(self, ext_name, stretch='asinh', units='arcsec',
                       vmin=None, vmax=None, a=None, ax=None, plain=False,
                       **kwargs):
        '''
        Plot one extenstion.

        Parameters
        ----------
        ext_name : string
            Extension name.
        stretch : string (default: 'asinh')
            Choice of stretch: asinh, linear, sqrt, log.
        units : string (default: 'arcsec')
            Units of pixel scale.
        vmin (optional) : float
            Minimal value of imshow.
        vmax (optional) : float
            Maximal value of imshow.
        a (optional) : float
            Scale factor of some stretch function.
        ax (optional) : matplotlib Axis
            Axis to plot the image.
        plain : bool (default: False)
            If False, tune the image.
        **kwargs : Additional parameters goes into plt.imshow()

        Returns
        -------
        ax : matplotlib Axis
            Axis to plot the image.
        '''
        ext = self.get_extension(ext_name)
        ax = plot_image(ext, ext.wcs, stretch, units, vmin, vmax, a, ax, plain, **kwargs)
        return ax

    def plot_ellipse(self, ext_name, ax=None, thin=1, **kwargs):
        '''
        Plot the ellipse isophotes.
        '''
        if ax is None:
            plt.figure(figsize=(7, 7))
            ax = plt.gca()

        isolist = self.isophotes.get(ext_name, None)
        if isolist is None:
            raise KeyError('Cannot find isolist for {0}!'.format(ext_name))

        if 'color' not in kwargs:
            kwargs['color'] = 'r'
        if 'lw' not in kwargs:
            kwargs['lw'] = 0.5
        for iso in isolist[::thin]:
            x, y, = iso.sampled_coordinates()
            ax.plot(x, y, **kwargs)
        return ax

    def plot_mu(self, ext_name, xscale='log', yscale='mag', pixelscale=None,
                zeromag=None, exptime=None, ax=None, plain=False, show_error=True,
                error_type='int_err', **kwargs):
        '''
        Plot surface brightness profile.
        '''
        if yscale == 'mag':
            x, y, e = self.get_mu(ext_name, pixelscale=pixelscale, exptime=exptime,
                                  zeromag=zeromag)
        else:
            isolist = self.isophotes.get(ext_name, None)
            if isolist is None:
                raise KeyError('Cannot find isolist for {0}!'.format(ext_name))
            x = isolist.sma
            y = isolist.intens
            e = isolist.int_err

        # Plot
        if ax is None:
            plt.figure(figsize=(7, 7))
            ax = plt.gca()
        if show_error is True:
            ax.errorbar(x, y, yerr=e, **kwargs)
        else:
            ax.plot(x, y, **kwargs)
        if plain is False:
            if yscale == 'mag':
                ax.set_xlabel(r'Radius (arcsec)', fontsize=24)
                ax.set_ylabel(r'$\mu\,(\mathrm{mag\,arcsec^{-2}})$', fontsize=24)
            else:
                ax.set_xlabel(r'Radius (pixel)', fontsize=24)
                ax.set_ylabel(r'Flux ({0} per pixel)'.format(self.unit), fontsize=24)
            ax.minorticks_on()
            ax.set_xscale(xscale)
            if yscale != 'mag':
                ax.set_yscale(yscale)
            else:
                ax.invert_yaxis()
        return ax

    def remove_mask(self, ext_name):
        '''
        Remove the mask of the extension.
        '''
        self.extensions[ext_name][0].mask = None

    def set_mask(self, ext_name, mask):
        '''
        Set mask for the extension.

        Parameters
        ----------
        ext_name : string
            The name of the extentsion.
        mask : 2D array
            The mask.
        '''
        if ext_name not in self.extensions:
            raise ValueError('Cannot find {0} in the extensions!'.format(ext_name))

        assert self.extensions[ext_name][0].shape == mask.shape, 'Mask shape incorrect!'
        self.extensions[ext_name][0].mask = mask

    def __getitem__(self, key):
        '''
        Get the extension and tag.

        Parameters
        ----------
        key : string
            Name of extention.
        '''
        return self.extensions[key]

    def __repr__(self):
        ext_names = list(self.extensions.keys())
        image = self.get_extension(self.get_ext_names()[0])
        info1 = 'Extensions: {0}'.format(', '.join(ext_names))
        info2 = 'Image size: {0}x{1}'.format(image.shape[1], image.shape[0])
        return '\n'.join([info1, info2])


class imgblock_subcomp(object):
    '''
    GALFIT subcomponent output.  E.g., output from
        `galfit -o3 <file> (standard img. block)`
    '''
    def __init__(self, filename, zeromag=None, pixelscale=None, exptime=None, unit='adu'):
        '''
        Parameters
        ----------
        filename : string
            Path of the GALFIT output file.
        '''
        f = fits.open(filename)
        header = f[1].header
        self.model_info = get_model_from_header(header)
        self.unit = unit
        self.zeromag = zeromag
        self.pixelscale = pixelscale
        self.exptime = exptime

        self.extensions = {}
        for loop in range(self.model_info['N_components']):
            ext = CCDData.read(filename, hdu=loop+1, unit=unit)
            ext_obj = ext.header['OBJECT']
            counter = 0
            comp_name = '{0}_{1}'.format(ext_obj, counter)
            while True:
                if comp_name in self.extensions:
                    counter += 1
                    comp_name = '{0}_{1}'.format(ext_obj, counter)
                else:
                    break
            self.extensions[comp_name] = [ext, None]
        self.isophotes = {}

    def combine_components(self, ext_list, ext_name=None, tag=None):
        '''
        Combine the components.

        Parameters
        ----------
        ext_list : list
            List of extension names.

        Returns
        -------
        ext : 2D array
            Image combining different components.
        '''
        extList = []
        for en in ext_list:
            extList.append(self.get_extension(en).data)
        ext = CCDData(np.sum(extList, axis=0), unit=self.unit)

        if ext_name is not None:
            assert ext_name not in self.extensions, 'The name has been used!'
            self.extensions[ext_name] = (ext, tag)
        return ext

    def cut_fov(self, position, size):
        '''
        Cut the field of view.

        Parameters
        ----------
        position : (x, y)
            Position of the image center, units: pixel.
        size : (dx, dy)
            Size of the new image, units: pixel.
        '''
        for loop, ext_name in enumerate(self.extensions):
            ext, tag = self.extensions[ext_name]
            img_cut = Cutout2D(ext.data, position=position, size=size).data
            if ext.mask is not None:
                mask = Cutout2D(ext.mask, position=position, size=size).data
            else:
                mask = None
            ccd_new = CCDData(img_cut, wcs=ext.wcs, mask=mask, unit=ext.unit)
            self.extensions[ext_name] = [ccd_new, tag]

    def fit_ellipse(self, ext_name, x0=None, y0=None, sma=None, eps=0, pa=0,
                    expand=False, **kwargs):
        '''
        Fit isophotes of an extension.

        Parameters
        ----------
        ext_name : string
            Name of the extenstion to be fitted.
        x0, y0 : float
            The center pixel coordinate of the ellipse.
        sma : float
            The semimajor axis of the ellipse in pixels.
        eps : ellipticity
            The ellipticity of the ellipse.
        pa : float
            The position angle (in radians) of the semimajor axis in
            relation to the postive x axis of the image array (rotating
            towards the positive y axis). Position angles are defined in the
            range :math:`0 < PA <= \\pi`. Avoid using as starting position
            angle of 0., since the fit algorithm may not work properly. When
            the ellipses are such that position angles are near either
            extreme of the range, noise can make the solution jump back and
            forth between successive isophotes, by amounts close to 180
            degrees.
        **kwargs : Additional parameters feed to ellipse.fit_image().

        Returns
        -------
        isolist : IsophoteList instance
            A list-like object of Isophote instances, sorted by increasing
            semimajor axis length.
        '''
        ext = self.get_extension(ext_name)
        if ext.mask is None:
            image = ext.data
        else:
            image = np.ma.array(ext.data, mask=ext.mask)

        if x0 is None:
            x0 = image.shape[1] / 2
        if y0 is None:
            y0 = image.shape[0] / 2
        if sma is None:
            ## Estimate the size of the source
            #xx, yy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
            #xx = xx.astype(np.float64) - x0
            #yy = yy.astype(np.float64) - y0
            #rr = np.sqrt(xx**2 + yy**2)
            #sma = np.sum(rr * image) / np.sum(image)
            sma = 10

        isolist = fit_ellipse(image, x0, y0, sma, eps, pa, **kwargs)

        if expand is True:
            step = kwargs.get('step', 0.1)
            fflag = kwargs.get('fflag', 0.7)
            maxsma = kwargs.get('maxsma', None)
            isolist_exp = grow_isophote(image, isolist[-1], step=step, fflag=fflag,
                                        maxsma=maxsma, maxsteps=np.inf)
            isolist = isolist + isolist_exp

        self.isophotes[ext_name] = isolist
        return isolist

    def fit_isophote(self, ext_name, isolist):
        '''
        Fit isophotes according to the input isolist.

        Parameters
        ----------
        ext_name : string
            Name of the extenstion to be fitted.
        isolist : IsophoteList
            Input isophotes.

        Returns
        -------
        isolist_out : IsophoteList
            New measurements.
        '''
        ext = self.get_extension(ext_name)
        if ext.mask is None:
            image = ext.data
        else:
            image = np.ma.array(ext.data, mask=ext.mask)
        isolist_out = fit_isophote(image, isolist)
        self.isophotes[ext_name] = isolist_out
        return isolist_out

    def get_ext_names(self):
        '''
        Get list of extension names.
        '''
        return list(self.extensions.keys())

    def get_extension(self, ext_name):
        '''
        Get the extension data.
        '''
        return self.extensions[ext_name][0]

    def get_model_param(self, ext_name, parname):
        '''
        Get model parameter value.
        '''
        idx = list(self.extensions.keys()).index(ext_name) + 1
        info = self.model_info['COMP_{0}'.format(idx)]
        return info.get('{0}_{1}'.format(idx, parname))

    def get_mu(self, ext_name, pixelscale=None, exptime=None, zeromag=None):
        '''
        Get the surface brightness.
        '''
        isolist = self.isophotes.get(ext_name, None)
        if isolist is None:
            raise KeyError('Cannot find isolist for {0}!'.format(ext_name))
        if pixelscale is None:
            assert self.pixelscale is not None
            pixelscale = self.pixelscale
        if exptime is None:
            assert self.exptime is not None
            exptime = self.exptime
        if zeromag is None:
            assert self.zeromag is not None
            zeromag = self.zeromag

        x = isolist.sma * pixelscale
        y = isolist.intens / exptime / pixelscale**2
        e = isolist.int_err / exptime / pixelscale**2
        y, e = flux2mag(y, e, zeromag)
        return x, y, e

    def get_isolist(self, ext_name):
        '''
        Get isolist of an extension.
        '''
        return self.isophotes[ext_name]

    def get_tag(self, ext_name):
        '''
        Get the extension tag.
        '''
        return self.extensions[ext_name][1]

    def plot_extension(self, ext_name, stretch='asinh', units='arcsec',
                       vmin=None, vmax=None, a=None, ax=None, plain=False,
                       **kwargs):
        '''
        Plot one extenstion.

        Parameters
        ----------
        ext_name : string
            Extension name.
        stretch : string (default: 'asinh')
            Choice of stretch: asinh, linear, sqrt, log.
        units : string (default: 'arcsec')
            Units of pixel scale.
        vmin (optional) : float
            Minimal value of imshow.
        vmax (optional) : float
            Maximal value of imshow.
        a (optional) : float
            Scale factor of some stretch function.
        ax (optional) : matplotlib Axis
            Axis to plot the image.
        plain : bool (default: False)
            If False, tune the image.
        **kwargs : Additional parameters goes into plt.imshow()

        Returns
        -------
        ax : matplotlib Axis
            Axis to plot the image.
        '''
        ext = self.get_extension(ext_name)
        ax = plot_image(ext, ext.wcs, stretch, units, vmin, vmax, a, ax, plain, **kwargs)
        return ax

    def plot_extension_all(self, stretch='asinh', units='arcsec', vmin=None,
                           vmax=None, a=None, axs=None, plain=False, **kwargs):
        '''
        Plot all of the extensions for quick look.
        '''
        next = len(self.extensions)
        if axs is None:
            fig, axs = plt.subplots(1, next, sharey=True, figsize=(5*next, 5))
        for loop, ext in enumerate(self.extensions):
            ax = axs[loop]
            self.plot_extension(ext, stretch, units, vmin, vmax, a, ax,
                                True, **kwargs)
            if plain is False:
                fig.subplots_adjust(wspace=0)
                ax.text(0.05, 0.95, ext, fontsize=20, transform=ax.transAxes,
                        ha='left', va='top')
        return axs

    def plot_ellipse(self, ext_name, ax=None, thin=1, **kwargs):
        '''
        Plot the ellipse isophotes.
        '''
        if ax is None:
            plt.figure(figsize=(7, 7))
            ax = plt.gca()

        isolist = self.isophotes.get(ext_name, None)
        if isolist is None:
            raise KeyError('Cannot find isolist for {0}!'.format(ext_name))

        if 'color' not in kwargs:
            kwargs['color'] = 'r'
        if 'lw' not in kwargs:
            kwargs['lw'] = 0.5
        for iso in isolist[::thin]:
            x, y, = iso.sampled_coordinates()
            ax.plot(x, y, **kwargs)
        return ax

    def plot_mu(self, ext_name, xscale='log', yscale='mag', pixelscale=None,
                zeromag=None, exptime=None, ax=None, plain=False, **kwargs):
        '''
        Plot surface brightness profile.
        '''
        if yscale == 'mag':
            x, y, e = self.get_mu(ext_name, pixelscale=pixelscale, exptime=exptime,
                                  zeromag=zeromag)
        else:
            isolist = self.isophotes.get(ext_name, None)
            if isolist is None:
                raise KeyError('Cannot find isolist for {0}!'.format(ext_name))
            x = isolist.sma
            y = isolist.intens
            e = isolist.int_err

        # Plot
        if ax is None:
            plt.figure(figsize=(7, 7))
            ax = plt.gca()
        ax.plot(x, y, **kwargs)
        if plain is False:
            if yscale == 'mag':
                ax.set_xlabel(r'Radius (arcsec)', fontsize=24)
                ax.set_ylabel(r'$\mu\,(\mathrm{mag\,arcsec^{-2}})$', fontsize=24)
            else:
                ax.set_xlabel(r'Radius (pixel)', fontsize=24)
                ax.set_ylabel(r'Flux ({0} per pixel)'.format(self.unit), fontsize=24)
            ax.minorticks_on()
            ax.set_xscale(xscale)
            if yscale != 'mag':
                ax.set_yscale(yscale)
            else:
                ax.invert_yaxis()
        return ax

    def remove_mask(self, ext_name):
        '''
        Remove the mask of the extension.
        '''
        self.extensions[ext_name][0].mask = None

    def set_mask(self, ext_name, mask):
        '''
        Set mask for the extension.

        Parameters
        ----------
        ext_name : string
            The name of the extentsion.
        mask : 2D array
            The mask.
        '''
        if ext_name not in self.extensions:
            raise ValueError('Cannot find {0} in the extensions!'.format(ext_name))

        assert self.extensions[ext_name][0].shape == mask.shape, 'Mask shape incorrect!'
        self.extensions[ext_name][0].mask = mask

    def set_tag(self, ext_name, tag):
        '''
        Set the tag.
        '''
        self.extensions[ext_name][1] = tag

    def __getitem__(self, key):
        '''
        Get the extension and tag.

        Parameters
        ----------
        key : string
            Name of extention.
        '''
        return self.extensions[key]

    def __repr__(self):
        ext_names = list(self.extensions.keys())
        image = self.get_extension(self.get_ext_names()[0])
        info1 = 'Extensions: {0}'.format(', '.join(ext_names))
        info2 = 'Image size: {0}x{1}'.format(image.shape[1], image.shape[0])
        return '\n'.join([info1, info2])
