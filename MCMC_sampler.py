import Icarus
import matplotlib.pyplot as plt
import numpy as np
import emcee
from multiprocessing import Pool
import time
import os 
import pickle as pk


os.environ["OMP_NUM_THREADS"] = "1"

phot = Icarus.Photometry.Photometry(atmo_fln='atmo.txt', data_fln='data_mjd.txt', ndiv=8) #change ndiv to 16 later on

'''
#parameters we want to sample
theta = [MP_mcmc, MC_mcmc, i_mcmc, k1_mcmc, filling_mcmc, tbase_mcmc, tirr_mcmc, dm_mcmc, av_mcmc]
'''

#define tophat priors
def log_prior (theta):
    """
    This function defines tophat priors used in our sampling. We use the convention:
    In range => returns 0.0 
    Out of range => returns -np.inf
    """
    start = time.time()
    #unpack
    MP_mcmc, MC_mcmc, sin_i, k1_mcmc, filling_mcmc, tbase_mcmc, tirr_mcmc, dm_mcmc, av_mcmc = theta

    #define tophat prior
    if (0.8< MP_mcmc <2.5 and 0.1< MC_mcmc <0.5 and 0.5< sin_i <1 and 0< filling_mcmc <1 and 3000< tbase_mcmc <8000 and
     3000< tirr_mcmc <8000 and 0<dm_mcmc<15 and 0<av_mcmc<0.5):

        #define gaussian prior for k1 
        k1_val = 325000
        k1_sigma = 50000
        log_prior_k1 = -0.5 * (((k1_mcmc - k1_val) / k1_sigma)**2)

        # print(f'Prior eval: {time.time() -  start}')

        return log_prior_k1

    return -np.inf


#define likelihood of each step 
def log_likelihood (theta):
    """
    This function calculates the log likelihood for our parameters that is later used in def log_prob
    """

    #unpack
    MP_mcmc, MC_mcmc, sin_i, k1_mcmc, filling_mcmc, tbase_mcmc, tirr_mcmc, dm_mcmc, av_mcmc = theta

    # input parameters
    ip = {'q' : MP_mcmc / MC_mcmc, # mass ratio (M_P / M_C)
        'porb' : 36000, # orbital period in seconds
        'incl' : np.arcsin((sin_i)), # inclination in radians
        'k1' : k1_mcmc, # companion's radial velocity in m/s
        'omega': 1., # companion co-rotation rate (keep @ 1)
        'filling' : filling_mcmc, # Roche lobe filling factor (0->1)
        'temp' : {'tbase':tbase_mcmc, # Base temperature
                'tirr' :tirr_mcmc}, # Temperature applied to irradiated face
        'tempgrav' : .25, # gravity darkening coefficient
        'dm' : dm_mcmc, # distance modulus
        'av' : av_mcmc} # visual extinction

    # generate the model 
    phases = phot.data['phase']

    startll = time.time()
    model = phot.Get_flux(ip)
    endll = time.time()
    # print (f"Time for log_likelihood: {endll - startll}")

    model_r, model_g, model_u = model
    # fluxes = np.array(phot.data['flux'])
    fluxes_r, fluxes_g, fluxes_u = phot.data['mag'] 
    # dfluxes = phot.data['flux_err']
    dfluxes_r, dfluxes_g, dfluxes_u = phot.data['mag_err']

    #residuals and chi2
    residuals = []
    chi2 = 0

    for model_flux, fluxes, dfluxes in zip([model_r, model_g, model_u], [fluxes_r, fluxes_g, fluxes_u], [dfluxes_r, dfluxes_g, dfluxes_u]):
        res = fluxes - model_flux
        residuals.append(res)
        chi2 += np.sum((res / dfluxes)**2)

    #we return a negative chi2
    return -0.5*chi2

def log_prob (theta):
    """
    This function calculates the log posterioir probability based on def log_prior and def log_likelihood
    """
    startlp = time.time()
    #call the previous two functions
    lp = log_prior(theta)

    # Reject parameters outside the prior immediately 
    if not np.isfinite(lp): 
        return -np.inf

    ll = log_likelihood(theta)
    endlp = time.time()
    # print(f"Time for log_prob: {endlp - startlp}")
    return lp + ll



#set up the initial parameters & emcee sampler 
nwalkers = 18 #each walker produces its own chain (should be AT LEAST twice the number of dimensions) start at 20 lets say
ndim = 9
nsteps = 15000 #number of steps to run


initial_vals = np.array([1.4, 0.3, 0.75, 325000, 0.7, 4000, 5000, 10, 0.15 ]) #based on whatever we know about the system
initial_states = np.array((1e-4*np.random.randn(nwalkers, ndim)) + initial_vals)
print(type(initial_states))
print(initial_states.shape)


with Pool(25) as pool:

    # create the Icarus object to handle photometry

    emcee_sampler = emcee.EnsembleSampler (nwalkers, ndim, log_prob, pool=pool)
    for sample in emcee_sampler.sample (initial_states, iterations=nsteps, progress = True): #progress = True gives a progress bar 
        if emcee_sampler.iteration % 50 == 0:
            print("parameters saved!")
            samples = emcee_sampler.get_chain(flat = True)
            # print ("Final model paramaters:")
            # parameters = ["MP", "MC", "incl", "k1", "filling", "tbase", "tirr", "dm", "av"]
            # for parameter, value in zip (parameters, best_theta):
            #     print (f"{parameter}: {value:.5f}")
            theta = np.median(samples, axis=0)
            MP_mcmc, MC_mcmc, sin_i, k1_mcmc, filling_mcmc, tbase_mcmc, tirr_mcmc, dm_mcmc, av_mcmc = theta
            ip = {'q' : MP_mcmc / MC_mcmc, # mass ratio (M_P / M_C)
        'porb' : 36000, # orbital period in seconds
        'incl' : np.arcsin(sin_i), # inclination in radians
        'k1' : k1_mcmc, # companion's radial velocity in m/s
        'omega': 1., # companion co-rotation rate (keep @ 1)
        'filling' : filling_mcmc, # Roche lobe filling factor (0->1)
        'temp' : {'tbase':tbase_mcmc, # Base temperature
                'tirr' :tirr_mcmc}, # Temperature applied to irradiated face
        'tempgrav' : .25, # gravity darkening coefficient
        'dm' : dm_mcmc, # distance modulus
        'av' : av_mcmc} # visual extinction
            with open('current_params.pk', 'wb') as f:
                pk.dump(ip, f)
        

#inspect the sampler's MCMC results using .get_chain
samples = emcee_sampler.get_chain(flat = True, discard = 5000) #leave blank for now, later we can add: thin =

#save chains 
with open("sampler.pk", "wb") as f:
    pk.dump(emcee_sampler, f)


#median of posteroir and print results
# best_theta = np.median (samples, axis = 0)
loglike = emcee_sampler.get_log_prob(flat=True, discard=5000)
best_idx = np.argmax(loglike)
best_theta = samples[best_idx]
print ("Final model paramaters:")
parameters = ["MP", "MC", "sin(i)", "k1", "filling", "tbase", "tirr", "dm", "av"]
for parameter, value in zip (parameters, best_theta):
    print (f"{parameter}: {value:.5f}")

#generate final model using Icarus 
MP_mcmc, MC_mcmc, sin_i, k1_mcmc, filling_mcmc, tbase_mcmc, tirr_mcmc, dm_mcmc, av_mcmc = best_theta
ip = {'q' : MP_mcmc / MC_mcmc, # mass ratio (M_P / M_C)
        'porb' : 36000, # orbital period in seconds
        'incl' : np.arcsin(sin_i), # inclination in radians
        'k1' : k1_mcmc, # companion's radial velocity in m/s
        'omega': 1., # companion co-rotation rate (keep @ 1)
        'filling' : filling_mcmc, # Roche lobe filling factor (0->1)
        'temp' : {'tbase':tbase_mcmc, # Base temperature
                'tirr' :tirr_mcmc}, # Temperature applied to irradiated face
        'tempgrav' : .25, # gravity darkening coefficient
        'dm' : dm_mcmc, # distance modulus
        'av' : av_mcmc} # visual extinction

final_phases = phot.data['phase']
final_models = phot.Get_flux(ip)
final_fluxes = phot.data['flux']
final_dfluxes = phot.data['flux_err']


# plot the final model and data
for i in range(len(final_models)):
    plt.errorbar(final_phases[i], final_fluxes[i], final_dfluxes[i], marker='o', linestyle='None', alpha=.25)
    plt.plot(np.sort(final_phases[i]), final_models[i][np.argsort(final_phases[i])], color='k')
plt.xlim(0,1)
plt.savefig("final_plot")
plt.close()