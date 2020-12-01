import numpy as np
from astropy.io import fits

__all__ = ['convert_output_string2float', 'get_model_from_header']

def convert_output_string2float(s):
    '''
    Convert the parameter of the header to float.

    Parameters
    ----------
    s : string
        Header information in string.
    
    Returns
    -------
    v : float
        Best-fit value.
    ve : float
        Fit uncertainty.
    vf : string
        Flag: free, contrained, or fixed.
    '''
    if s is None:
        v = None  
        ve = None
        f = None
    elif '{' in s:
        v = float(s[1:-1])
        ve = None
        f = 'contrained'
    elif '[' in s:
        v = float(s[1:-1])
        ve = None
        f = 'fixed'
    elif '+/-' in s:
        v_list = s.split('+/-')
        v = float(v_list[0])
        ve = float(v_list[1])
        f = 'free'
    else:
        raise ValueError('Input cannot be understood: {0}'.format(s))
    return v, ve, f

def get_model_from_header(header, parlist=None, ncomponents_max=100):
    '''
    Get model information from the header.
    
    Parameters
    ----------
    header : FITS header
        The header of model extensions, which contains best-fit parameters.
    parlist (optional) : list
        List of parameter names.
    ncomponents_max : int
        Maximum number of parameters.
    
    Returns
    -------
    info_dict : dict
        Model information. Number of components and details of each components.
          *_e is the uncertainty.
          *_f is the flag.
    '''
    if parlist is None:
        parlist = ['XC', 'YC', 'MAG', 'RE', 'N', 'AR', 'PA']
        
    counter = 1
    info_dict = {}
    while True:
        comp_name = header.get('COMP_{0}'.format(counter), None)
        if comp_name is None:
            break
        info_dict['COMP_{0}'.format(counter)] = {'{0}_MODEL'.format(counter): comp_name}

        for loop, par in enumerate(parlist):
            vc, vce, vcf = convert_output_string2float(header.get('{0}_{1}'.format(counter, par), None))
            info_dict['COMP_{0}'.format(counter)]['{0}_{1}'.format(counter, par)] = vc
            info_dict['COMP_{0}'.format(counter)]['{0}_{1}_e'.format(counter, par)] = vce
            info_dict['COMP_{0}'.format(counter)]['{0}_{1}_f'.format(counter, par)] = vcf
        counter += 1
        if counter > ncomponents_max:
            break
    info_dict['N_components'] = counter - 1
    return info_dict
