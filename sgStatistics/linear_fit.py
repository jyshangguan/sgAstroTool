import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

__all__ = ['linearfit']

class linearfit(object):
    '''
    A fitter of a straight line.
    '''
    def __init__(self, x, y, xerr=None, yerr=None, fix_a=None, fix_b=None, fix_s=None,
                 prior_a=[-5, 5], prior_b=[-5, 5], prior_s=[0, 5], x_ref=None, y_ref=None):
        '''
        '''
        parname_all = ['a', 'b', 's']
        parname_fit = parname_all.copy()
        
        self._pfixed = {}
        if fix_a is not None:
            assert (fix_a > prior_a[0]) & (fix_a < prior_a[1])
            self._pfixed['a'] = fix_a
            parname_fit.remove('a')
        if fix_b is not None:
            assert (fix_b > prior_b[0]) & (fix_b < prior_b[1])
            self._pfixed['b'] = fix_b
            parname_fit.remove('b')
        if fix_s is not None:
            assert (fix_s > prior_s[0]) & (fix_s < prior_s[1])
            self._pfixed['s'] = fix_s
            parname_fit.remove('s')
        
        self.ndim = len(parname_fit)
        if self.ndim == 0:
            raise ValueError('Please do not fix all parameters!')
        
        self._parname_all = parname_all
        self._parname_fit = parname_fit
            
        if x_ref is None:
            self._xref = np.mean(x)
        else:
            self._xref = x_ref
            
        if y_ref is None:
            self._yref = np.mean(y)
        else:
            self._yref = y_ref
            
        self.x = x
        self.y = y
        self._x_fit = x - self._xref
        self._y_fit = y - self._yref
        
        if xerr is None:
            self.xerr = np.zeros_like(x)
        else:
            self.xerr = xerr
        if yerr is None:
            self.yerr = np.zeros_like(y)
        else:
            self.yerr = yerr
            
        self.prior_a = prior_a
        self.prior_b = prior_b
        self.prior_s = prior_s
        
        self.sampler = None
        self.flat_samples = None
        
    def get_parlist(self, theta):
        '''
        Get the list of parameters.
        '''
        pdict = self._pfixed.copy()
        for k, p in zip(self._parname_fit, theta):
            pdict[k] = p
            
        return [pdict['a'], pdict['b'], pdict['s']]
        
    def log_likelihood(self, theta):
        '''
        Log likelihood.
        '''
        a, b, s = self.get_parlist(theta)
        
        m = a + b * self._x_fit
        sigma2 = self.yerr**2 + (b * self.xerr)**2 + s**2
        return -0.5 * np.sum((self._y_fit - m)**2 / sigma2 + np.log(sigma2))
    
    def log_prior(self, theta):
        '''
        Prior.
        '''
        a, b, s = self.get_parlist(theta)
        
        if (self.prior_a[0] < a < self.prior_a[1] and \
            self.prior_b[0] < b < self.prior_b[1] and \
            self.prior_s[0] < s < self.prior_s[1]):
            return 0.0
        return -np.inf
    
    def log_probability(self, theta):
        '''
        Probability.
        '''
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta)
    
    def fit(self, p0=None, nwalkers=32, nsteps=5000, ncpus=1, progress=True):
        '''
        Fit the data.
        '''
        if p0 is None:
            self._p0 = np.ones(self.ndim)
        else:
            self._p0 = p0
            
        self._nwalkers = nwalkers
        self._nsteps = nsteps
        
        pos = self._p0 + 1e-4 * np.random.randn(nwalkers, self.ndim)
        
        if ncpus > 1:
            from multiprocessing import Pool
            pool = Pool(processes=ncpus)
        else:
            pool = None
            
        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self.log_probability, 
                                        pool=pool)
        sampler.run_mcmc(pos, nsteps, progress=progress)
        self.sampler = sampler
        
    def get_bestfit(self, discard=100, thin=10, q=[16, 50, 84]):
        '''
        Get the best fit parameters.
        '''
        flat_samples = self.sampler.get_chain(discard=discard, thin=thin, flat=True)
        self.flat_samples = flat_samples
        
        plow, pmed, phig = np.percentile(flat_samples, q, axis=0)
        
        rdict = {}
        for loop, k in enumerate(self._parname_fit):
            rdict[k] = {
                'center': pmed[loop],
                'range': [pmed[loop]-plow[loop], phig[loop]-pmed[loop]]
            }
        self._rdict = rdict
        return rdict
    
    def get_bestfit_intercept(self):
        '''
        Get the intercept of the best-fit model.
        '''
        pfit = [v['center'] for v in self._rdict.values()]
        a, b, s = self.get_parlist(pfit)
        intercept = a + self._yref - b * self._xref
        return intercept

    def cal_bestfit_model(self, x):
        '''
        Calculate the best-fit linear model. 
        '''
        pfit = [v['center'] for v in self._rdict.values()]
        a, b, s = self.get_parlist(pfit)
        y = a + self._yref + b * (x - self._xref)
        return y
    
    def plot_corner(self, **kwargs):
        '''
        Corner plot.
        '''
        if self.flat_samples is None:
            self.get_bestfit()
        
        if 'labels' not in kwargs:
            kwargs['labels'] = self._parname_fit
        if 'show_titles' not in kwargs:
            kwargs['show_titles'] = True
        if 'quantiles' not in kwargs:
            kwargs['quantiles'] = [0.16, 0.5, 0.84]
        fig = corner.corner(self.flat_samples, **kwargs)
    
    def plot_chain(self, alpha=0.3, fontsize=24):
        '''
        Plot the chains.
        '''
        fig, axes = plt.subplots(self.ndim, figsize=(self.ndim*3, 7), sharex=True)
        samples = self.sampler.get_chain()
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(samples[:, :, i], "k", alpha=alpha)
            ax.set_xlim(0, len(samples))
            ax.set_ylabel(self._parname_fit[i], fontsize=fontsize)
        
        axes[-1].set_xlabel("step number", fontsize=fontsize)
        return axes
        
    def get_autocorr_time(self):
        return self.sampler.get_autocorr_time()
    
    def plot_fit(self, ax=None, q=[16, 84], color_data='k', color_model='r', alpha=0.2):
        '''
        Plot the fitting.
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        
        ax.errorbar(self.x, self.y, xerr=self.xerr, yerr=self.yerr, ls='none', color=color_data)
        
        xlim = ax.get_xlim()
        self.plot_fit_model(xlim, ax=ax, q=q, color=color_model, alpha=alpha)
        return ax
    
    def plot_fit_model(self, xlim, ax=None, q=[16, 84], color='r', alpha=0.2, label=None):
        '''
        Plot the fitting.
        '''
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 7))
        
        if self.flat_samples is None:
            self.get_bestfit()
       
        pfit = [v['center'] for v in self._rdict.values()]
        a, b, s = self.get_parlist(pfit)
        
        xm = np.linspace(xlim[0], xlim[1], 100)
        ym = a + self._yref + b * (xm - self._xref)
        ax.plot(xm, ym, color=color, lw=2, label=label)
        
        ymList = []
        inds = np.random.randint(len(self.flat_samples), size=100)
        for ind in inds:
            a, b, s = self.get_parlist(self.flat_samples[ind])
            ymList.append(a + self._yref + b * (xm - self._xref))
        ym_low, ym_hig = np.percentile(ymList, q, axis=0)
        ax.fill_between(xm, y1=ym_low, y2=ym_hig, color=color, alpha=alpha)
        return ax