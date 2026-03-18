import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys,os,glob
# from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.interpolate import splrep, splev
from astropy.convolution import convolve_fft
from collections import OrderedDict
import logging
import inspect

def forward_func(
    lw,w2,ILS,vza,w_shift=0,
    T_layer=None,N2O=None,CH4=None,H2O=None,Ts=None,emissivity=None,
    T_layer_scale=1.,N2O_scale=1.,CH4_scale=1.,H2O_scale=1.,
    return_all_jacs=True
):
    '''
    lw:
        a Longwave object
    w2:
        wavelength grid of target radiance
    ILS,vza:
        instrument lineshape, viewing zenith angle
    w_shift:
        wavelength shift
    '''
    w1 = lw.w1
    lw.vza = vza
    profile_keys = []
    scalar_keys = []
    if T_layer is not None:
        lw.profiles['T_layer'] = T_layer*T_layer_scale
        lw.profiles['T_level'] = np.concatenate(
            (
                [T_layer[0]],(T_layer[0:-1]+T_layer[1:])/2,[T_layer[-1]]
            )
        )
        profile_keys.append('T_layer')
    else:
        lw.profiles['T_layer'] *= T_layer_scale
        lw.profiles['T_level'] = np.concatenate(
            (
                [lw.profiles['T_layer'][0]],
                (lw.profiles['T_layer'][0:-1]+lw.profiles['T_layer'][1:])/2,
                [lw.profiles['T_layer'][-1]]
            )
        )
    
    if N2O is not None:
        lw.profiles['N2O'] = N2O*N2O_scale
        profile_keys.append('N2O')
    else:
        lw.profiles['N2O'] *= N2O_scale
    
    if CH4 is not None:
        lw.profiles['CH4'] = CH4*CH4_scale
        profile_keys.append('CH4')
    else:
        lw.profiles['CH4'] *= CH4_scale
    
    if H2O is not None:
        lw.profiles['H2O'] = H2O*H2O_scale
        profile_keys.append('H2O')
    else:
        lw.profiles['H2O'] *= H2O_scale
    
    if Ts is not None:
        lw.Ts = Ts
        scalar_keys.append('Ts')
    if emissivity is not None:
        lw.emissivity = emissivity
        scalar_keys.append('emissivity')
    
    if return_all_jacs:
        profile_keys = ['T_layer','N2O','CH4','H2O']
        scalar_keys = ['Ts']
    
    scales_dict = {
        'T_layer':T_layer_scale,
        'N2O':N2O_scale,
        'CH4':CH4_scale,
        'H2O':H2O_scale
    }
    jacs = OrderedDict()
    for key in profile_keys:
        jacs[key] = np.zeros((lw.nlayer,len(w2)))
        jac_w1 = lw.get_jacobian(
            key,upper_B_jac=True,lower_B_jac=True
        )
        profile0 = lw.profiles[key]/scales_dict[key]
        jac_scale_w1 = np.sum(
            jac_w1.T*np.broadcast_to(profile0[:,np.newaxis],(lw.nlayer,len(w1))),
            axis=0,keepdims=False
        )
        jac_scale = convolve_fft(jac_scale_w1,ILS,normalize_kernel=True)
        bspline_coef = splrep(w1,jac_scale)
        jacs[key+'_scale'] = splev(w2+w_shift,bspline_coef).reshape(1,len(w2))
        for ilayer in range(lw.nlayer):
            jac = convolve_fft(
                jac_w1[:,ilayer],ILS,
                normalize_kernel=True
            )
            bspline_coef = splrep(w1,jac)
            jacs[key][ilayer,:] = splev(w2+w_shift,bspline_coef)/scales_dict[key]
        
    for key in scalar_keys:
        jacs[key] = np.zeros((1,len(w2)))
        jac_w1 = lw.get_jacobian(key)
        jac = convolve_fft(
            jac_w1[:,0],ILS,normalize_kernel=True
        )
        bspline_coef = splrep(w1,jac)
        jacs[key][0,:] = splev(w2+w_shift,bspline_coef)
    
    if len(profile_keys) == 0 and len(scalar_keys) == 0:
        lw.get_sigmas()
    r1_oversampled = convolve_fft(lw.get_radiance(),ILS,normalize_kernel=True)
    bspline_coef = splrep(w1,r1_oversampled)
    jacs['radiance'] = splev(w2+w_shift,bspline_coef)
    jacs['w_shift'] = splev(w2+w_shift,bspline_coef,der=1)
    jacs['wavelength'] = w2
    return jacs


class RetrievalResults(object):
    def __init__(self):
        pass


class Parameter:
    def __init__(self, name,prior=None,value=None,prior_error=None,
                 prior_error_matrix=None,prior_corr_matrix=None,
                 p_profile=None,correlation_scaleHeight=None,
                 vmin=-np.inf,vmax=np.inf,vary=True):
        self.name = name
        self.prior = prior
        self.vmin = vmin
        self.vmax = vmax
        self.vary = vary
        if value is None:
            value = prior
        self.value = value
        if np.isscalar(prior):
            self.prior_error_matrix = prior_error**2
            self.nstate = 1
            return
        self.nstate = len(prior)
        if prior_error_matrix is not None:
            self.prior_error_matrix = prior_error_matrix
            return
        # prevent zero profile prior error
        mask = prior_error == 0
        prior_error[mask] = np.min(prior_error[~mask])
        
        if prior_corr_matrix is not None:
            self.prior_error_matrix = prior_corr_matrix*np.outer(prior_error,prior_error)
            return
        self.prior_error_matrix = np.diag(prior_error**2)
        if correlation_scaleHeight is not None and p_profile is not None:
            log_p_profile = np.log(p_profile)
            for (i,logp1) in enumerate(log_p_profile):
                for (j,logp2) in enumerate(log_p_profile):
                    if i == j:
                        continue
                    self.prior_error_matrix[i,j] = prior_error[i]*prior_error[j]*np.exp(-np.abs(logp1-logp2)/correlation_scaleHeight)


class Parameters(OrderedDict):
    def __init__(self):
        pass
    
    def add(self,param):
        OrderedDict.__setitem__(self,param.name,param)
    
    def flatten_values(self,field_to_flatten):
        nstates = 0
        for (name,par) in self.items():
            if not par.vary:
                continue
            nstates = nstates+par.nstate
        Sa = np.zeros((nstates,nstates))
        count = 0
        beta0 = np.zeros(nstates)
        params_names = []
        for (name,par) in self.items():
            if not par.vary:
                continue
            params_names.append(name)
            Sa[count:count+par.nstate,count:count+par.nstate] = par.prior_error_matrix
            beta0[count:count+par.nstate] = getattr(par,field_to_flatten)
            count = count+par.nstate
        return beta0, Sa, nstates, params_names
    
    def update_vectors(self,vector_name,vector):
        count = 0
        for (name,par) in self.items():
            if not par.vary:
                continue
            new_values = vector[count:count+par.nstate]
            if vector_name == 'value':
                new_values[new_values<par.vmin] = par.vmin
                new_values[new_values>par.vmax] = par.vmax
            setattr(self[name],vector_name,new_values)
            count = count+par.nstate
    
    def update_matrices(self,matrix_name,matrix):
        count = 0
        for (name,par) in self.items():
            setattr(self[name],matrix_name,matrix[count:count+par.nstate,count:count+par.nstate])
            count = count+par.nstate


class LWOE(object):
    def __init__(self,func,independent_vars,param_names):
        '''
        func:
            callable, the forward function
        independent_vars:
            a list of independent variables with prescribed values
        param_names:
            a list of names for parameters to be fitted in OE
        '''
        self.logger = logging.getLogger(__name__)
        self.func = func
        self.independent_vars = independent_vars
        self._param_names = param_names
        self._func_allargs = []
        self._func_haskeywords = False
        self.param_hints = OrderedDict()
        self._parse_params()
    
    def param_names(self):
        """Return the parameter names of the Model."""
        return self._param_names
    
    def _parse_params(self):
        """Build parameters from function arguments."""
        pos_args = []
        kw_args = {}
        keywords_ = None
        sig = inspect.signature(self.func)
        for fnam, fpar in sig.parameters.items():
            if fpar.kind == fpar.VAR_KEYWORD:
                keywords_ = fnam
            elif fpar.kind == fpar.POSITIONAL_OR_KEYWORD:
                if fpar.default == fpar.empty:
                    pos_args.append(fnam)
                else:
                    kw_args[fnam] = fpar.default
            elif fpar.kind == fpar.VAR_POSITIONAL:
                raise ValueError("varargs '*%s' is not supported" % fnam)
        # inspection done

        self._func_haskeywords = keywords_ is not None
        self._func_allargs = pos_args + list(kw_args.keys())
        
    def make_funcargs(self, params=None, kwargs=None):
        """Convert parameter values and keywords to function arguments."""
        if params is None:
            params = {}
        if kwargs is None:
            kwargs = {}
        out = {}
        for name, par in params.items():
            if name in self._func_allargs or self._func_haskeywords:
                out[name] = par.value

        # kwargs handled slightly differently -- may set param value too!
        for name, val in kwargs.items():
            if name in self._func_allargs or self._func_haskeywords:
                out[name] = val
        return out
    
    def evaluate(self,params=None,**kwargs):
        return self.func(**self.make_funcargs(params, kwargs))
    
    def set_prior(self,name,**kwargs):
        if name not in self.param_hints:
            self.param_hints[name] = OrderedDict()

        for key, val in kwargs.items():
            self.param_hints[name][key] = val
    
    def make_params(self):
        params = Parameters()
        for name in self.param_names():
            par = Parameter(name,**self.param_hints[name])
            params.add(par)
        return params
    
    def retrieve(self,radiance,radiance_error,
                 params=None,max_iter=100,use_LM=False,
                 max_diverging_step=5,converge_criterion_scale=1,**kwargs):
        
        if params is None:
            params = self.make_params()
        nw2 = len(radiance)
        beta0, Sa, nstates, params_names = params.flatten_values(field_to_flatten='prior')
        beta = beta0.copy()
        
        Sa_inv = np.linalg.inv(Sa)
        yy = radiance
        Sy = np.diag(radiance_error**2)
        Sy_inv = np.diag(1/radiance_error**2)
        count = 0
        count_div = 0
        dsigma2 = np.inf
        result = RetrievalResults()
        result.if_success = True
        if use_LM:
            LM_gamma = 10.
        
        while(dsigma2 > nstates*converge_criterion_scale and count < max_iter):
            self.logger.info('Iteration {}'.format(count))
            if count != 0:
                params.update_vectors(vector_name='value',vector=beta)
            jacs = self.evaluate(params,**kwargs)
            yhat = jacs['radiance']
            all_jacobians = [jacs[name].T for name in params_names]
            K = np.column_stack(all_jacobians)
            if use_LM:
                self.logger.info('gamma = {}'.format(LM_gamma))
                dbeta = np.linalg.inv((1+LM_gamma)*Sa_inv+K.T@Sy_inv@K)@(K.T@Sy_inv@(yy-yhat)-Sa_inv@(beta-beta0))
                beta_try = beta+dbeta
                c_i = (yy-yhat).T@Sy_inv@(yy-yhat)+(beta-beta0).T@Sa_inv@(beta-beta0)
                params.update_vectors(vector_name='value',vector=beta_try)
                yhat_try = self.evaluate(params,**kwargs)['radiance']
                c_in1 = (yy-yhat_try).T@Sy_inv@(yy-yhat_try)+(beta_try-beta0).T@Sa_inv@(beta_try-beta0)
                yhat_linear = yhat+K@dbeta
                c_in1_FC = (yy-yhat_linear).T@Sy_inv@(yy-yhat_linear)+(beta_try-beta0).T@Sa_inv@(beta_try-beta0)
                LM_R = (c_i-c_in1)/(c_i-c_in1_FC)
                self.logger.info('R = {:.3f}'.format(LM_R))
                if LM_R <=0.0001:
                    LM_gamma = LM_gamma*10
                    self.logger.warning('R = {:.3f} and is diverging. Abandon step and increase gamma by 10'.format(LM_R))
                    count_div += 1
                    self.logger.info('{} diverging steps'.format(count_div))
                    if count_div >= max_diverging_step:
                        self.logger.warning('too many diverging steps, abandon retrieval')
                        result.if_success = False
                        break
                    continue
                elif LM_R < 0.25:
                    LM_gamma = LM_gamma*10
                elif LM_R < 0.75:
                    pass
                else:
                    LM_gamma = LM_gamma/2
            else:
                dbeta = np.linalg.inv(Sa_inv+K.T@Sy_inv@K)@(K.T@Sy_inv@(yy-yhat)-Sa_inv@(beta-beta0))
            dsigma2 = dbeta.T@(K.T@Sy_inv@(yy-yhat)+Sa_inv@(beta-beta0))
            self.logger.info('dsigma2: {}'.format(dsigma2))
            self.logger.debug(' '.join('{:2E}'.format(b) for b in beta))
            beta = beta+dbeta
            if count == 0:
                result.y0 = jacs['radiance']
                result.wavelength = jacs['wavelength']
                result.Sy = Sy
                result.Sa = Sa
                result.beta0 = beta0
            count = count+1
#        params.update_vectors(vector_name='value',vector=beta)
        result.nw2 = nw2
        result.yy = yy
        result.yhat = yhat
        result.niter = count
        beta = beta-dbeta
        result.beta = beta
        result.Jprior = (beta-beta0).T@Sa_inv@(beta-beta0)
        result.max_iter = max_iter
        if result.niter == max_iter:
            result.if_success = False
        result.chi2 = np.sum(np.power(yy-yhat,2))/np.trace(Sy)
        result.rmse = np.sqrt(np.mean(np.power(yy-yhat,2)))
        Shat = np.linalg.inv(K.T@Sy_inv@K+Sa_inv)
        result.Shat = Shat
        AVK = Shat@K.T@Sy_inv@K
        result.AVK = AVK
        G = Sa@K.T@np.linalg.inv(K@Sa@K.T+Sy)
        Sm = G@Sy@G.T
        result.Sm = Sm
        params.update_matrices(matrix_name='posterior_error_matrix',matrix=Shat)
        params.update_vectors(vector_name='posterior_error',vector=np.sqrt(np.diag(Shat)))
        params.update_matrices(matrix_name='measurement_error_matrix',matrix=Sm)
        params.update_vectors(vector_name='measurement_error',vector=np.sqrt(np.diag(Sm)))
        params.update_matrices(matrix_name='averaging_kernel',matrix=AVK)
        params.update_vectors(vector_name='dofs',vector=np.diag(AVK))
        result.params = params
        return result