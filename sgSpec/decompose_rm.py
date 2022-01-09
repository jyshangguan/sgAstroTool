from operator import add
from functools import reduce
import numpy as np
import dynesty
import emcee
import corner
from dynesty import utils as dyfunc
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
#from numpy.polynomial.legendre import legval
#from astropy.modeling.core import Fittable1DModel
#from astropy.modeling.parameters import Parameter
from astropy.modeling.polynomial import Legendre1D

ls_km = 2.99792458e5  # km/s

__all__ = ['spectrum_fitter', 'model_spectrum', 'gaussian_filter1d']


class model_spectrum(object):
    '''
    The full model of the observed spectrum.
    '''
    
    def __init__(self, model_physical_private, model_physical_shared=None, 
                 ins_add=None, ins_mul=None, ins_dsp_sigma=0, verbose=True):
        '''
        Input the model.
        '''
        self._mphy_pri = model_physical_private
        self._mphy_sha = model_physical_shared
        
        # Instrumental dispersion
        if ins_dsp_sigma > 0:
            self._ins_dsp = True  
            self._ins_dsp_sigma = ins_dsp_sigma  # In km/s
        else:
            self._ins_dsp = False
            self._ins_dsp_sigma = None
        
        if ins_add is not None:
            assert isinstance(ins_add, dict), 'ins_add should be a dict!'
            degree = ins_add.get('degree', 2)
            domain = ins_add.get('domain', None)
            bounds_input = ins_add.get('bounds', None)
            
            if domain is None:
                raise ValueError('Need to set the domain of the ins_add!!')
            
            if bounds_input is None:
                raise ValueError('Need to set the bounds of the ins_add!!')
            
            c_default = {}
            bounds = {}
            for loop in range(degree+1):
                c_default['c{0}'.format(loop)] = 0
                bounds['c{0}'.format(loop)] = bounds_input
            self._mins_add = Legendre1D(degree, domain=domain, bounds=bounds, name='ins_add', **c_default)
        else:
            self._min_add = None
        
        if ins_mul is not None:
            assert isinstance(ins_mul, dict), 'ins_mul should be a dict!'
            degree = ins_add.get('degree', 2)
            domain = ins_add.get('domain', None)
            
            if domain is None:
                raise ValueError('Need to set the domain of the ins_mul!!')
            
            c_default = {'c0': 1}
            bounds = {}
            for loop in range(degree):
                c_default['c{0}'.format(loop+1)] = 0
                bounds['c{0}'.format(loop+1)] = [-1, 1]
            self._mins_mul = Legendre1D(degree, domain=domain, bounds=bounds, name='ins_mul',
                                        fixed={'c0': True}, **c_default)
        else:
            self._mins_mul = None
        
        self._verbose = verbose
        
        # Physical, private
        self._mphy_pri_name = []  # Model name
        self._pphy_pri_map = {}  # Map the index to the parameter
        self._npar_phy_pri = 0  # Count the number of phy_pri parameters
        for loop, m in enumerate(self._mphy_pri):
            self._mphy_pri_name.append(m.name)
            for idx, k in enumerate(m.fixed.keys()):
                if (m.fixed[k] is False) and (m.tied[k] is False):
                    self._pphy_pri_map['{}'.format(self._npar_phy_pri)] = (loop, k)
                    self._npar_phy_pri += 1
        
        # Physical, shared
        self._mphy_sha_name = []  # Model name
        self._pphy_sha_map = {}  # Map the index to the parameter
        self._npar_phy_sha = 0  # Count the number of phy_pri parameters
        if self._mphy_sha:
            for loop, m in enumerate(self._mphy_sha):
                self._mphy_sha_name.append(m.name)
                for idx, k in enumerate(m.fixed.keys()):
                    if (m.fixed[k] is False) and (m.tied[k] is False):
                        self._pphy_sha_map['{}'.format(self._npar_phy_sha)] = (loop, k)
                        self._npar_phy_sha += 1
                    
        # Instrumental, additive
        self._pins_add_name = []   # Parameter name
        self._pins_add_index = []  # Parameter index
        if self._mins_add:
            for idx, k in enumerate(self._mins_add.fixed.keys()):
                if (self._mins_add.fixed[k] is False) and (self._mins_add.tied[k] is False):
                    self._pins_add_name.append(k)
                    self._pins_add_index.append(idx)
                        
        # Instrumental, multiplicative
        self._pins_mul_name = []   # Parameter name
        self._pins_mul_index = []  # Parameter index
        if self._mins_mul:
            for idx, k in enumerate(self._mins_mul.fixed.keys()):
                if (self._mins_mul.fixed[k] is False) and (self._mins_mul.tied[k] is False):
                    self._pins_mul_name.append(k)
                    self._pins_mul_index.append(idx)
       
        # Number of model components
        self._nmod_phy_pri = len(self._mphy_pri_name)
        self._nmod_phy_sha = len(self._mphy_sha_name)
        self._nmod_all = self._nmod_phy_pri + self._nmod_phy_sha
        if self._mins_add:
            self._nmod_all += 1
        if self._mins_mul:
            self._nmod_all += 1
        if self._ins_dsp is True:
            self._nmod_all += 1
        
        # Number of parameters
        self._npar_ins_add = len(self._pins_add_name)
        self._npar_ins_mul = len(self._pins_mul_name)
        self._npar_all = self._npar_phy_pri + self._npar_phy_sha + self._npar_ins_add + self._npar_ins_mul
        if self._ins_dsp is True:
            self._npar_all += 1
    
    def cal_private(self, x, component_index=None):
        '''
        Calculate the private model.
        '''
        if component_index is None:
            model = reduce(add, self._mphy_pri)
        else:
            model = self._mphy_pri[component_index]
        
        return model(x)
    
    def cal_shared(self, x, component_index=None):
        '''
        Calculate the shared model.
        '''
        if component_index is None:
            model = reduce(add, self._mphy_sha)
        else:
            model = self._mphy_sha[component_index]
        
        return model(x)
    
    def cal_additive(self, x):
        '''
        Calculate the instrumental additive component.
        '''
        if self._mins_add:
            return self._mins_add(x)
        else:
            return 0
    
    def cal_multiplicative(self, x):
        '''
        Calculate the instrumental cal_multiplicative component.
        '''
        if self._mins_mul:
            return self._mins_mul(x)
        else:
            return 1

    def get_parameters(self):
        '''
        Get the free parameters for fit.
        '''
        params = []
        
        for idx in range(self._npar_phy_pri):
            nm, pn = self._pphy_pri_map['{}'.format(idx)]
            params.append(self._mphy_pri[nm].__getattribute__(pn).value)
        
        if self._mphy_sha:
            for idx in range(self._npar_phy_sha):
                nm, pn = self._pphy_sha_map['{}'.format(idx)]
                params.append(self._mphy_sha[nm].__getattribute__(pn).value)
                
        if self._mins_add:
            for pn in self._pins_add_name:
                params.append(self._mins_add.__getattribute__(pn).value)
                
        if self._mins_mul:
            for pn in self._pins_mul_name:
                params.append(self._mins_mul.__getattribute__(pn).value)
        
        if self._ins_dsp is True:
            params.append(self._ins_dsp_sigma)
        
        return np.array(params)
    
    def set_par(self, pindex, pvalue):
        '''
        Set the free parameters of the models.
        '''
        # Physical, private
        if pindex < self._npar_phy_pri:
            idx = pindex
            nm, pn = self._pphy_pri_map['{}'.format(idx)]
            self._mphy_pri[nm].__setattr__(pn, pvalue)
        
        # Physical, shared
        elif (pindex < (self._npar_phy_pri + self._npar_phy_sha)) & (self._mphy_sha is not None):
            idx = pindex - self._npar_phy_pri
            nm, pn = self._pphy_sha_map['{}'.format(idx)]
            self._mphy_sha[nm].__setattr__(pn, pvalue)
        
        # Instrumental, additive
        elif (pindex < (self._npar_phy_pri + self._npar_phy_sha + self._npar_ins_add)) & (self._mins_add is not None):
            idx = pindex - (self._npar_phy_pri + self._npar_phy_sha)
            pn = self._pins_add_name[idx]
            self._mins_add.__setattr__(pn, pvalue)
            
        # Instrumental, additive
        elif (pindex < (self._npar_phy_pri + self._npar_phy_sha + self._npar_ins_add + self._npar_ins_mul)) & (self._mins_mul is not None):
            idx = pindex - (self._npar_phy_pri + self._npar_phy_sha + self._npar_ins_add)
            pn = self._pins_mul_name[idx]
            self._mins_mul.__setattr__(pn, pvalue)
        
        # Instrumental dispersion
        elif self._ins_dsp is True:
            self._ins_dsp_sigma = pvalue
        
        else:
            raise ValueError('The index ({0}) is out of the range of the free parameters ({1})!'.format(pindex, self._npar_all))
    
    def set_params(self, params):
        '''
        Set the free parameters of the models.
        '''
        for loop, pv in enumerate(params):
            self.set_par(loop, pv)
    
    def _convolve(self, x, y):
        '''
        Convolve the spectrum with instrumental dispersion.
        '''
        assert self._ins_dsp_sigma is not None
        
        dlam = np.diff(x)[0]  # Assume constant dlam
        sigma_pix = self._ins_dsp_sigma * x / ls_km / dlam
        
        y_conv = gaussian_filter1d(y, sigma_pix)
        return y_conv
    
    def __call__(self, x, phy_pri=None, phy_sha=None, ins_add=None, ins_mul=None):
        '''
        Call the model to calculate the results.
        '''
        if phy_pri is None:
            phy_pri = self.cal_private(x)
            
        if phy_sha is None:
            phy_sha = self.cal_shared(x)
        
        if ins_add is None:
            ins_add = self.cal_additive(x)
        
        if ins_mul is None:
            ins_mul = self.cal_multiplicative(x)
            
        phy_all = phy_pri + phy_sha
        
        if self._ins_dsp is True:
            phy_all = self._convolve(x, phy_all)
        
        return (phy_all + ins_add) * ins_mul
    
    
def gaussian_filter1d(spec, sig):
    '''
    Convolve a spectrum by a Gaussian with different sigma for every pixel.
    If all sigma are the same this routine produces the same output as
    scipy.ndimage.gaussian_filter1d, except for the border treatment.
    Here the first/last p pixels are filled with zeros.
    When creating a template library for SDSS data, this implementation
    is 60x faster than a naive for-loop over pixels.

    :param spec: vector with the spectrum to convolve
    :param sig: vector of sigma values (in pixels) for every pixel
    :return: spec convolved with a Gaussian with dispersion sig

    '''
    sig = sig.clip(0.01)  # forces zero sigmas to have 0.01 pixels
    p = int(np.ceil(np.max(3*sig)))
    m = 2*p + 1  # kernel size
    x2 = np.linspace(-p, p, m)**2

    n = spec.size
    a = np.zeros((m, n))
    for j in range(m):   # Loop over the small size of the kernel
        a[j, p:-p] = spec[j:n-m+j+1]

    gau = np.exp(-x2[:, None]/(2*sig**2))
    gau /= np.sum(gau, 0)[None, :]  # Normalize kernel

    conv_spectrum = (a*gau).sum(0)

    return conv_spectrum
    

#class Legendre1D_2nd(Fittable1DModel):
#    '''
#    The Legendre1D.
#
#    Parameters
#    ----------
#    x : array like
#        Wavelength, units: arbitrary.
#    c_i : float
#        The coefficients.
#    '''
#    
#    c0 = Parameter(default=0, bounds=(-1, 1))
#    c1 = Parameter(default=0, bounds=(-1, 1))
#    c2 = Parameter(default=0, bounds=(-1, 1))
#
#    @staticmethod
#    def evaluate(x, c0, c1, c2):
#        """
#        Gaussian model function.
#        """
#        x = np.linspace(-1, 1, len(x))
#        f = legval(x, [c0, c1, c2])
#
#        return f


def ptform_uniform(u, bounds):
    '''
    Transform the Uniform(0, 1) sample to the model parameter value with the
    uniform prior.

    Parameters
    ----------
    u : float
        The random number ~Uniform(0, 1).
    prior : dict
        The prior of the parameter, prior=dict(bound=[a, b]).
    '''
    pmin, pmax = bounds
    return (pmax - pmin) * (u - 0.5) + (pmax + pmin) * 0.5


class spectrum_fitter(object):
    '''
    A test to use dynesty and Astropy model.
    '''
    
    def __init__(self, model, data, verbose=True):
        '''
        Input the model.
        '''
        self._model = model
        self._data = data
        self._wave = data['wave']
        self._flux = data['flux']
        self._ferr = data['ferr']
        self._inv2 = data['ferr']**(-2)
        
        self._parameter_index_list = []
        self._parameter_name_list = []
        
        for idx, k in enumerate(model.fixed.keys()):
            if (model.fixed[k] is False) and (model.__getattr__(k).tied is False):
                self._parameter_index_list.append(idx)
                self._parameter_name_list.append(k)
        
        # Number of free parameters
        self._nparams = len(self._parameter_index_list)
        
        if verbose:
            print('There are {0} free parameters!'.format(self._nparams))

    def prior_transform(self, u):
        '''
        Transforms the uniform random variables `u ~ Unif[0., 1.)` to the parameters of interest.
        '''
        params = np.array(u)
        for loop, p in enumerate(params):
            pn = self._parameter_name_list[loop]
            params[loop] = ptform_uniform(p, self._model.bounds[pn])
        return params
    
    def get_model(self, params):
        '''
        Get a model with the input parameters.
        '''
        model = self._model.copy()
        for loop, pn in enumerate(self._parameter_name_list):
            model.__setattr__(pn, params[loop])
        return model
    
    def get_model_parameters(self):
        '''
        Get the initial parameters of the input model.
        '''
        params = []
        for loop, idx in enumerate(self._parameter_index_list):
            params.append(self._model.parameters[idx])
        return np.array(params)
    
    def get_samples(self):
        '''
        Get the equal weighted samples.
        '''
        results = self._results
        weights = np.exp(results.logwt - results.logz[-1])  # normalized weights
        # Resample weighted samples. This is similar to the typical MCMC samples.
        samples_equal = dyfunc.resample_equal(results.samples, weights)
        return samples_equal
    
    def loglike(self, params):
        '''
        Calculate the likelihood.
        '''
        self.set_parameters(params)
        fmod = self._model(self._wave)
        ll = -0.5 * np.sum((self._flux - fmod)**2 * self._inv2 - np.log(2 * np.pi * self._inv2))
        return ll
    
    def logprob(self, params):
        '''
        Calculate the log probability.
        '''
        for loop, pn in enumerate(self._parameter_name_list):
            pmin, pmax = self._model.bounds[pn]
            if (params[loop] < pmin) | (params[loop] > pmax):
                return -1e25
        return self.loglike(params)
    
    def plot_corner(self, par_range=[None, None]):
        '''
        Plot the corner
        '''
        p1, p2 =  par_range
        c = self._sampler.get_chain(flat=True, discard=self._nburn)
        labels = self._parameter_name_list[p1:p2]
        corner.corner(c[:, p1:p2], labels=labels)
        
    def plot_chain(self):
        """
        Generate the chain plot.
        """
        chain = self._sampler.get_chain()
        ndim = chain.shape[2]
        
        fig, axs = plt.subplots(ndim, 1, sharex=True, figsize=(8, 3*ndim))
        fig.subplots_adjust(hspace=0)
        for loop in range(ndim):
            axs[loop].plot(chain[:, :, loop].T, color="k", alpha=0.4)
            axs[loop].yaxis.set_major_locator(MaxNLocator(5))
            axs[loop].set_ylabel(self._parameter_name_list[loop], fontsize=24)
        return (fig, axs)
        
    def pos_init(self):
        """
        Initialize the position of walkers.

        Parameters
        ----------
        nwalkers : int
            The number of walkers.

        Returns
        -------
        posList : array
            The position of the walkers with the shape (nwalkers, ndim).
        """
        posList = []
        for loop, pn in enumerate(self._parameter_name_list):
            pmin, pmax = self._model.bounds[pn]
            pos = (pmax - pmin) * np.random.rand(self._nwalkers) + pmin #Uniform distribution
            posList.append(pos)
        posList = np.array(posList).T
        return posList
        
    def pos_ball(self, radius=1e-4):
        '''
        Generate the position of the walkers as a hyperball.
        '''
        pos_center = self.get_model_parameters()
        pos = pos_center + radius * np.random.randn(self._nwalkers, self._nparams)
        return pos
    
    def pos_median(self):
        """
        Get the the median of the posterior probability based on the equal
        weight samples.

        Returns
        -------
        p : 1D array
            The list of parameters with median posterior probability.
        """
        p = np.median(self.get_samples(), axis=0)
        return p
        
    def run_dynesty(self, ncpu=1, sampler_name='DynamicNestedSampler',
                    sampler_kws={}, run_nested_kws={}, verbose=True):
        '''
        Run the fit with dynesty.

        Parameters
        ----------
        ncpu : int (default: 1)
            The number of the cpu.
        sampler_kws : dict
            The keywords passed to the sampler object.
        run_nested_kws : dict
            The keywords passed to the run_nested function.
        '''
        if not 'bound' in sampler_kws:
            sampler_kws['bound'] = 'multi'
        if not 'sample' in sampler_kws:
            sampler_kws['sample'] = 'unif'
        if verbose is True:
            print('#-----------------------------------------------#')
            print(' Fit with {0}'.format(sampler_name))
            print(' Number of CPUs: {0}'.format(ncpu))
            print(' Number of parameters: {0}'.format(self._nparams))
            optList = ['bound', 'sample']
            for opt in optList:
                print(' {0}: {1}'.format(opt, sampler_kws.get(opt, 'default')))
            print('#-----------------------------------------------#')
        if not 'pool' in sampler_kws:
            if ncpu > 1:
                from multiprocessing import Pool
                sampler_kws['pool'] = Pool(processes=ncpu)
            else:
                sampler_kws['pool'] = None
            sampler_kws['queue_size'] = ncpu
        if sampler_name == 'DynamicNestedSampler':
            sampler = dynesty.DynamicNestedSampler(self.loglike, self.prior_transform, ndim=self._nparams,
                                                   **sampler_kws)
        elif sampler_name == 'NestedSampler':
            sampler = dynesty.NestedSampler(self.loglike, self.prior_transform, ndim=self._nparams,
                                            **sampler_kws)
        else:
            raise ValueError('The sampler name ({0}) is not recognized!'.format(sampler_name))
        sampler.run_nested(**run_nested_kws)

        if sampler_kws.get('pool', None) is not None:
            sampler.pool.close()

        self._sampler = sampler
        self._results = sampler.results
        
    def run_emcee(self, nwalkers, nsteps=100, nburn=100, ncpu=1, progress=True, verbose=True):
        '''
        Run the fit with emcee.
        '''
        self._nwalkers = nwalkers
        self._nsteps = nsteps
        self._nburn = nburn
        
        if ncpu > 1:
            from multiprocessing import Pool
            pool = Pool(processes=ncpu)
        else:
            pool = None
        #pos = self.pos_init()
        pos = self.pos_ball()
        sampler = emcee.EnsembleSampler(nwalkers, self._nparams, self.logprob, pool=pool)
        sampler.run_mcmc(pos, nsteps+nburn, progress=progress)

        if pool is not None:
            sampler.pool.close()

        self._sampler = sampler
        
    def set_parameters(self, params):
        '''
        Set the model parameters.
        '''
        for loop, pn in enumerate(self._parameter_name_list):
            self._model.__setattr__(pn, params[loop])
