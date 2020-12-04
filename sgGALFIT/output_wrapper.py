import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.nddata import CCDData
from .utils import *


__all__ = ['imgblock_standard', 'imgblock_subcomp']

class imgblock_standard(object):
    '''
    GALFIT standard output.  E.g., output from
        `galfit -o2 <file> (standard img. block)`
    '''
    def __init__(self, filename, zeromag=None, pixelscale=None, unit=None):
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
            pixelscale = self.get_pixelscale(units='arcsec')
        self.pixelscale = pixelscale
        self.isophotes = {}

    def fit_ellipse(self, ext_name, x0=None, y0=None, sma=None, eps=0, pa=0,
                    **kwargs):
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
        image = self.get_extension(ext_name).data
        if x0 is None:
            x0 = image.shape[1] / 2
        if y0 is None:
            y0 = image.shape[0] / 2
        if sma is None:
            # Estimate the size of the source
            xx, yy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
            xx = xx.astype(np.float64) - x0
            yy = yy.astype(np.float64) - y0
            rr = np.sqrt(xx**2 + yy**2)
            sma = np.sum(rr * image) / np.sum(image)

        isolist = fit_ellipse(image, x0, y0, sma, eps, pa, **kwargs)
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
        image = self.get_extension(ext_name).data
        isolist_out = fit_isophote(image, isolist)
        self.isophotes[ext_name] = isolist_out
        return isolist_out

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
        cdelt1, cdelt2 = wcs_pixel_scale(data.wcs)
        nrow, ncol = data.shape
        x_len = ncol * cdelt1.to(units).value
        y_len = nrow * cdelt2.to(units).value
        extent = (-x_len/2, x_len/2, -y_len/2, y_len/2)
        return extent

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

    def get_pixelscale(self, units='arcsec'):
        '''
        Get pixel scale of the data image.
        '''
        cdelt1, cdelt2 = wcs_pixel_scale(self.get_extension('data').wcs)
        cdelt1 = cdelt1.to(units).value
        cdelt2 = cdelt2.to(units).value
        if not np.isclose(cdelt1, cdelt2, rtol=0.001):
            raise ValueError('The pixel scale of two axes are not the same!')
        ps = (cdelt1 + cdelt2) / 2
        return ps

    def plot_direction(self, ax, xy, length, color='k', fontsize=20, linewidth=2,
                       units='arcsec', backextend=0.05):
        '''
        Plot the direction arrow. Only applied to plots using WCS.

        Parameters
        ----------
        ax : Axis
            Axis to plot the direction.
        xy : (x, y)
            Coordinate of the origin of the arrows.
        length : float
            Length of the arrows.
        units: string (default: arcsec)
            Units of xy and length.
        '''
        wcs = self.extensions['data'][0].wcs
        header = wcs.to_header()
        pixelscale = self.get_pixelscale(units)
        d_ra = length / pixelscale * self.get_pixelscale('degree')
        d_dec = length / pixelscale * self.get_pixelscale('degree')
        ra = [header['CRVAL1'], header['CRVAL1']+d_ra, header['CRVAL1']]
        dec = [header['CRVAL2'], header['CRVAL2'], header['CRVAL2']+d_dec]
        ra_pix, dec_pix = wcs.all_world2pix(ra, dec, 1)
        d_arrow1 = [ra_pix[1]-ra_pix[0], dec_pix[1]-dec_pix[0]]
        d_arrow2 = [ra_pix[2]-ra_pix[0], dec_pix[2]-dec_pix[0]]
        d_arrow1 = np.array(d_arrow1) * pixelscale
        d_arrow2 = np.array(d_arrow2) * pixelscale

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
        print(xy_e)
        ax.annotate('E', xy=xy_e, xycoords='data', fontsize=fontsize,
                    xytext=(d_arrow1[0]+xy[0], d_arrow1[1]+xy[1]), color=color,
                    arrowprops=dict(color=color, arrowstyle="<-", lw=linewidth),
                    ha=ha1, va=va1,
                    )
        xy_n = (xy[0] - d_arrow2[0] * backextend, xy[1] - d_arrow2[1] * backextend)
        ax.annotate('N', xy=xy_n, xycoords='data', fontsize=fontsize,
                    xytext=(d_arrow2[0]+xy[0], d_arrow2[1]+xy[1]), color=color,
                    arrowprops=dict(color=color, arrowstyle="<-", lw=linewidth),
                    ha=ha2, va=va2,
                    )

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
        extension = self.get_extension(ext_name)
        ax = plot_image(extension.data, extension.wcs, stretch, units, vmin,
                        vmax, a, ax, plain, **kwargs)
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
                zeromag=None, ax=None, plain=False, show_error=True, **kwargs):
        '''
        Plot surface brightness profile.
        '''
        # Get the isophotal list
        isolist = self.isophotes.get(ext_name, None)
        if isolist is None:
            raise KeyError('Cannot find isolist for {0}!'.format(ext_name))

        # Plot
        if ax is None:
            plt.figure(figsize=(7, 7))
            ax = plt.gca()
        if pixelscale is None:
            pixelscale = self.pixelscale

        x = isolist.sma * pixelscale
        if yscale == 'mag':
            if zeromag is None:
                zeromag = self.zeromag
            y, e = flux2mag(isolist.intens, isolist.int_err, zeromag)
        else:
            y = isolist.intens
            e = isolist.int_err
        if show_error is True:
            ax.errorbar(x, y, yerr=e, **kwargs)
        else:
            ax.plot(x, y, **kwargs)
        if plain is False:
            ax.set_xlabel(r'Radius (arcsec)', fontsize=24)
            if yscale == 'mag':
                ax.set_ylabel(r'$\mu\,(\mathrm{mag\,arcsec^{-2}})$', fontsize=24)
            else:
                ax.set_ylabel(r'Flux ({0} per pixel)'.format(self.unit), fontsize=24)
            ax.minorticks_on()
            ax.set_xscale(xscale)
            if yscale != 'mag':
                ax.set_yscale(yscale)
            else:
                ax.invert_yaxis()
        return ax

    def __getitem__(self, key):
        '''
        Get the extension and tag.

        Parameters
        ----------
        key : string
            Name of extention.
        '''
        return self.extensions[key]


class imgblock_subcomp(object):
    '''
    GALFIT subcomponent output.  E.g., output from
        `galfit -o3 <file> (standard img. block)`
    '''
    def __init__(self, filename, unit='adu'):
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

    def fit_ellipse(self, ext_name, x0=None, y0=None, sma=None, eps=0, pa=0,
                    **kwargs):
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
        image = self.get_extension(ext_name).data
        if x0 is None:
            x0 = image.shape[1] / 2
        if y0 is None:
            y0 = image.shape[0] / 2
        if sma is None:
            # Estimate the size of the source
            xx, yy = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
            xx = xx.astype(np.float64) - x0
            yy = yy.astype(np.float64) - y0
            rr = np.sqrt(xx**2 + yy**2)
            sma = np.sum(rr * image) / np.sum(image)

        isolist = fit_ellipse(image, x0, y0, sma, eps, pa, **kwargs)
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
        image = self.get_extension(ext_name).data
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
        extenstion = self.get_extension(ext_name)
        ax = plot_image(extenstion.data, extenstion.wcs, stretch, units, vmin,
                        vmax, a, ax, plain, **kwargs)
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
                zeromag=None, ax=None, plain=False, **kwargs):
        '''
        Plot surface brightness profile.
        '''
        # Get the isophotal list
        isolist = self.isophotes.get(ext_name, None)
        if isolist is None:
            raise KeyError('Cannot find isolist for {0}!'.format(ext_name))

        # Plot
        if ax is None:
            plt.figure(figsize=(7, 7))
            ax = plt.gca()
        if pixelscale is None:
            pixelscale = self.pixelscale

        x = isolist.sma * pixelscale
        if yscale == 'mag':
            if zeromag is None:
                zeromag = self.zeromag
            y, e = flux2mag(isolist.intens, isolist.int_err, zeromag)
        else:
            y = isolist.intens
            e = isolist.int_err
        ax.plot(x, y, **kwargs)
        if plain is False:
            ax.set_xlabel(r'Radius (arcsec)', fontsize=24)
            if yscale == 'mag':
                ax.set_ylabel(r'$\mu\,(\mathrm{mag\,arcsec^{-2}})$', fontsize=24)
            else:
                ax.set_ylabel(r'Flux ({0} per pixel)'.format(self.units), fontsize=24)
            ax.minorticks_on()
            ax.set_xscale(xscale)
            if yscale != 'mag':
                ax.set_yscale(yscale)
            else:
                ax.invert_yaxis()
        return ax

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
